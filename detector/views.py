from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.utils import timezone
from django.conf import settings
from .models import Detection, ModelVersion, APIKey, PerformanceMetric
from .ml_models import EfficientNetDetector, EnsembleDetector
from PIL import Image
from io import BytesIO
import base64
import json
import os
import torch
from facenet_pytorch import MTCNN
import asyncio
import concurrent.futures
from functools import wraps
from typing import List, Tuple, Dict
import time
from django.db.models import Avg
import psutil
from asgiref.sync import sync_to_async

# Initialize face detector
mtcnn = MTCNN(keep_all=True, device='cpu')

# Global ensemble detector
ensemble_detector = None

def get_active_models() -> List[EfficientNetDetector]:
    """Get all active model versions and create their detectors"""
    active_versions = ModelVersion.objects.filter(is_active=True)
    return [EfficientNetDetector(version) for version in active_versions]

@sync_to_async
def get_ensemble_detector_async():
    """Async wrapper for getting/creating ensemble detector"""
    global ensemble_detector
    if ensemble_detector is None:
        try:
            detectors = get_active_models()
            if not detectors:
                # Try to create a model if none exists
                from .model_setup import setup_model
                setup_model()
                detectors = get_active_models()
                if not detectors:
                    raise ValueError("Could not initialize ensemble detector - no active models")
            ensemble_detector = EnsembleDetector(detectors)
        except Exception as e:
            print(f"Error creating ensemble detector: {e}")
            raise ValueError(f"Could not initialize ensemble detector: {str(e)}")
    return ensemble_detector

def create_ensemble() -> EnsembleDetector:
    """Create an ensemble detector from all active models"""
    detectors = get_active_models()
    if not detectors:
        raise ValueError("No active model versions found")
    return EnsembleDetector(detectors)

def rate_limit(calls_per_hour=100):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            # Get API key from header
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return JsonResponse({'error': 'API key required'}, status=401)
            
            try:
                key_obj = APIKey.objects.get(key=api_key, is_active=True)
            except APIKey.DoesNotExist:
                return JsonResponse({'error': 'Invalid API key'}, status=401)
            
            # Check rate limit
            cache_key = f'rate_limit_{api_key}'
            calls = cache.get(cache_key, 0)
            
            if calls >= key_obj.rate_limit:
                return JsonResponse({'error': 'Rate limit exceeded'}, status=429)
            
            # Increment call count
            cache.set(cache_key, calls + 1, timeout=3600)  # 1 hour expiry
            
            # Update last used timestamp
            key_obj.last_used = timezone.now()
            key_obj.save()
            
            return view_func(request, *args, **kwargs)
        return wrapped_view
    return decorator

async def process_image(image: Image.Image) -> Tuple[str, float, List[Dict]]:
    """Process image asynchronously"""
    try:
        # Get detector using the async wrapper
        detector = await get_ensemble_detector_async()
        if detector is None:
            raise ValueError("Could not initialize ensemble detector")
        
        # Run prediction directly since it's already async
        result = await detector.predict(image)
        return result
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise

def index(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return render(request, 'detector/index.html', {'error': 'No image file uploaded'})
            
        try:
            image = request.FILES['image']
            img_pil = Image.open(image).convert('RGB')
            
            # Create detection record
            detection = Detection.objects.create(
                image=image,
                label="Processing",
                confidence=0.0,
                status='processing'
            )
            
            # Process image using asyncio.run to handle the async function
            start_time = time.time()
            try:
                label, confidence, boxes = asyncio.run(process_image(img_pil))
            except RuntimeError as e:
                if "cannot schedule new futures after interpreter shutdown" in str(e):
                    # Handle the case where the event loop is closed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    label, confidence, boxes = loop.run_until_complete(process_image(img_pil))
                    loop.close()
                else:
                    raise
                    
            processing_time = time.time() - start_time
            
            # Update detection record
            detection.label = label
            detection.confidence = confidence
            detection.boxes = boxes
            detection.status = 'completed'
            detection.processing_time = processing_time
            detection.save()

            context = {
                'detection': detection,
                'label': label,
                'confidence': confidence,
                'boxes': boxes,
                'processing_time': f"{processing_time:.2f}s"
            }

            return render(request, 'detector/index.html', context)
            
        except Exception as e:
            return render(request, 'detector/index.html', 
                        {'error': f'Error processing image: {str(e)}'})
    
    # Get detection history for display
    history = Detection.objects.filter(status='completed').order_by('-created_at')[:10]
    return render(request, 'detector/index.html', {'history': history})

def webcam_live_view(request):
    """View for the live webcam detection page"""
    try:
        # Create or get development API key
        dev_key, created = APIKey.objects.get_or_create(
            name='development',
            defaults={
                'is_active': True,
                'rate_limit': 1000,  # Higher limit for development
                'is_development': True
            }
        )
        return render(request, 'detector/live.html', {'api_key': dev_key.key})
    except Exception as e:
        print(f"Error creating API key: {e}")
        return render(request, 'detector/live.html', {'error': 'Failed to initialize API key'})

@rate_limit(calls_per_hour=1000)  # Increased limit for development
@csrf_exempt
def live_frame_api(request):
    if request.method != 'POST':
        return JsonResponse({
            'error': 'Invalid request method',
            'success': False
        }, status=400)
        
    try:
        # Parse request body
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data',
                'success': False
            }, status=400)
            
        if 'frame' not in data:
            return JsonResponse({
                'error': 'No frame data provided',
                'success': False
            }, status=400)
            
        # Process frame
        try:
            # Extract base64 data
            frame_data = data['frame'].split(';base64,')[1]
            frame_bytes = base64.b64decode(frame_data)
            frame = Image.open(BytesIO(frame_bytes)).convert('RGB')
            
            # Process image
            try:
                label, confidence, boxes = asyncio.run(process_image(frame))
                
                # Create detection record
                Detection.objects.create(
                    label=label,
                    confidence=confidence,
                    boxes=boxes,
                    status='completed'
                )
                
                return JsonResponse({
                    'success': True,
                    'label': label,
                    'confidence': float(confidence),
                    'boxes': boxes
                })
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                return JsonResponse({
                    'error': f'Failed to process frame: {str(e)}',
                    'success': False
                }, status=500)
                
        except Exception as e:
            return JsonResponse({
                'error': f'Invalid frame data: {str(e)}',
                'success': False
            }, status=400)
            
    except Exception as e:
        print(f"Unexpected error in live_frame_api: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error',
            'success': False
        }, status=500)

@rate_limit(calls_per_hour=100)
@csrf_exempt
def batch_process(request):
    """API endpoint for batch processing multiple images"""
    if request.method != 'POST':
        return JsonResponse({
            'error': 'Invalid request method',
            'success': False
        }, status=400)
        
    try:
        if not request.FILES.getlist('images'):
            return JsonResponse({
                'error': 'No images provided',
                'success': False
            }, status=400)
            
        results = []
        for image in request.FILES.getlist('images'):
            img_pil = Image.open(image).convert('RGB')
            
            # Process image
            label, confidence, boxes = asyncio.run(process_image(img_pil))
            
            # Create detection record
            detection = Detection.objects.create(
                image=image,
                label=label,
                confidence=confidence,
                boxes=boxes,
                status='completed'
            )
            
            results.append({
                'id': detection.id,
                'filename': image.name,
                'label': label,
                'confidence': confidence,
                'boxes': boxes
            })
            
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)

def model_metrics(request):
    """View for displaying model performance metrics"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
        
    metrics = PerformanceMetric.objects.select_related('model_version')\
        .order_by('-timestamp')[:1000]
        
    # Calculate aggregate metrics
    model_stats = {}
    for metric in metrics:
        version = metric.model_version
        if version.id not in model_stats:
            model_stats[version.id] = {
                'name': version.name,
                'avg_latency': [],
                'avg_memory': [],
                'success_rate': [],
                'total_predictions': 0
            }
        stats = model_stats[version.id]
        stats['avg_latency'].append(metric.latency)
        stats['avg_memory'].append(metric.memory_usage)
        stats['success_rate'].append(metric.success_rate)
        stats['total_predictions'] += metric.batch_size
        
    # Calculate final averages
    for stats in model_stats.values():
        stats['avg_latency'] = sum(stats['avg_latency']) / len(stats['avg_latency'])
        stats['avg_memory'] = sum(stats['avg_memory']) / len(stats['avg_memory'])
        stats['success_rate'] = sum(stats['success_rate']) / len(stats['success_rate']) * 100
        
    return JsonResponse({'model_stats': model_stats})

def model_dashboard(request):
    """View for the model performance dashboard"""
    if not request.user.is_authenticated:
        return redirect('login')
        
    # Get model versions and their metrics
    model_versions = []
    for version in ModelVersion.objects.filter(is_active=True):
        metrics = PerformanceMetric.objects.filter(model_version=version)\
            .order_by('-timestamp')[:1000]
            
        if metrics:
            avg_metrics = metrics.aggregate(
                avg_latency=Avg('latency'),
                avg_success=Avg('success_rate')
            )
            
            model_versions.append({
                'name': version.name,
                'accuracy': version.accuracy,
                'avg_latency': avg_metrics['avg_latency'],
                'success_rate': avg_metrics['avg_success'] * 100
            })
            
    # Get recent detections
    recent_detections = Detection.objects.filter(status='completed')\
        .order_by('-created_at')[:20]
        
    # Get performance metrics for chart
    latest_metrics = PerformanceMetric.objects.order_by('-timestamp')[:100]
    timestamps = [m.timestamp.strftime('%H:%M:%S') for m in latest_metrics]
    latencies = [float(m.latency) for m in latest_metrics]
    success_rates = [float(m.success_rate * 100) for m in latest_metrics]
    
    # Get system health metrics
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    # Get API usage metrics
    hour_ago = timezone.now() - timezone.timedelta(hours=1)
    api_requests = Detection.objects.filter(created_at__gte=hour_ago).count()
    active_users = len(set(Detection.objects.filter(
        created_at__gte=hour_ago
    ).values_list('user_id', flat=True)))
    
    context = {
        'model_versions': model_versions,
        'recent_detections': recent_detections,
        'timestamps': timestamps,
        'latencies': latencies,
        'success_rates': success_rates,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'api_requests_hour': api_requests,
        'active_users': active_users
    }
    
    return render(request, 'detector/dashboard.html', context)
