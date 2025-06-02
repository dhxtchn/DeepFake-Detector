import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import time
import psutil
from .models import ModelVersion, PerformanceMetric
import os
from asgiref.sync import sync_to_async
import asyncio
import cv2
from skimage.feature import local_binary_pattern

class BaseDeepfakeDetector(ABC):
    @abstractmethod
    def predict(self, image) -> Tuple[str, float, List[Dict]]:
        pass

    @abstractmethod
    def preprocess(self, image) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def version_info(self) -> ModelVersion:
        pass

class EfficientNetDetector(BaseDeepfakeDetector):
    def __init__(self, model_version: ModelVersion):
        super().__init__()
        self.model_version = model_version
        try:
            self.model = self._create_model()
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to initialize model: {str(e)}")

    @sync_to_async
    def _create_performance_metric(self, latency, memory_usage, success_rate):
        """Create performance metric in database"""
        return PerformanceMetric.objects.create(
            model_version=self.model_version,
            latency=latency,
            memory_usage=memory_usage,
            batch_size=1,
            success_rate=success_rate
        )
        
    def _create_model(self) -> nn.Module:
        try:
            # Use EfficientNet-B0 with updated weights
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            # Freeze early layers
            for param in base_model.features[:4].parameters():
                param.requires_grad = False
                
            # Create a custom feature extractor
            class FeatureExtractor(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.features = base_model.features
                    self.avg_pool = nn.AdaptiveAvgPool2d(1)
                    self.max_pool = nn.AdaptiveMaxPool2d(1)
                    
                def forward(self, x):
                    features = self.features(x)
                    avg_pooled = self.avg_pool(features)
                    max_pooled = self.max_pool(features)
                    return torch.cat([avg_pooled, max_pooled], dim=1)
            
            # Create attention module
            class AttentionModule(nn.Module):
                def __init__(self, in_features):
                    super().__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(in_features, in_features // 4),
                        nn.ReLU(),
                        nn.Linear(in_features // 4, in_features),
                        nn.Sigmoid()
                    )
                    
                def forward(self, x):
                    attention_weights = self.attention(x)
                    return x * attention_weights
            
            # Create the complete model
            class DeepfakeDetector(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.feature_extractor = FeatureExtractor(base_model)
                    num_features = 2560  # 1280 * 2 (from concatenation)
                    
                    self.classifier = nn.Sequential(
                        nn.Dropout(p=0.4),
                        nn.Linear(num_features, 1024),
                        nn.ReLU(),
                        nn.BatchNorm1d(1024),
                        AttentionModule(1024),
                        nn.Dropout(p=0.4),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.BatchNorm1d(512),
                        nn.Dropout(p=0.3),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.BatchNorm1d(256),
                        nn.Linear(256, 2)
                    )
                    
                def forward(self, x):
                    features = self.feature_extractor(x)
                    features = features.view(features.size(0), -1)
                    return self.classifier(features)
            
            # Create the model instance
            model = DeepfakeDetector(base_model)
            
            # Load weights if available
            if os.path.exists(self.model_version.file_path):
                file_size = os.path.getsize(self.model_version.file_path)
                if file_size > 10:
                    try:
                        state_dict = torch.load(self.model_version.file_path, map_location='cpu')
                        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        model.load_state_dict(state_dict, strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load weights: {e}")
            
            return model
                
        except Exception as e:
            raise ValueError(f"Failed to create model: {str(e)}")

    def preprocess(self, image) -> torch.Tensor:
        """Enhanced preprocessing with advanced image quality analysis"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Extract image quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Calculate noise level using multiple methods
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Noise estimation using multiple methods
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        noise = np.std(gray - blur)
        
        # Calculate local binary pattern for texture analysis
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))[0]
        lbp_hist = lbp_hist.astype('float') / np.sum(lbp_hist)
        
        # Store metrics for prediction
        self.image_stats = {
            'brightness': brightness,
            'contrast': contrast,
            'noise': noise,
            'lbp_hist': lbp_hist.tolist(),
            'texture_uniformity': np.sum(lbp_hist ** 2)
        }
        
        # Simplified preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)

    def _analyze_image_characteristics(self, stats: Dict[str, float]) -> Tuple[float, float]:
        """Advanced image characteristics analysis with strict fake detection"""
        fake_score = 0.0
        real_score = 0.0
        
        # Analyze brightness distribution
        brightness = stats['brightness']
        if 40 < brightness < 220:  # Normal range
            real_score += 0.3
        elif brightness < 20 or brightness > 235:  # Extreme values
            fake_score += 0.4
            
        # Analyze contrast
        contrast = stats['contrast']
        if 20 < contrast < 100:  # Normal range
            real_score += 0.3
        elif contrast < 10 or contrast > 120:  # Extreme values
            fake_score += 0.4
            
        # Analyze noise patterns
        noise = stats['noise']
        if 3 < noise < 25:  # Normal range
            real_score += 0.3
        elif noise > 35:  # High noise
            fake_score += 0.5
            
        # Analyze texture uniformity
        texture_uniformity = stats['texture_uniformity']
        if 0.05 < texture_uniformity < 0.35:  # Natural texture range
            real_score += 0.3
        elif texture_uniformity > 0.45:  # Too uniform
            fake_score += 0.5
        
        return real_score, fake_score

    async def predict(self, image) -> Tuple[str, float, List[Dict]]:
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            input_tensor = self.preprocess(image)
            real_score, fake_score = self._analyze_image_characteristics(self.image_stats)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                if len(output.shape) < 2 or output.shape[1] != 2:
                    raise ValueError("Invalid model output shape")
                    
                probs = torch.softmax(output, dim=1)[0]
                if len(probs) != 2:
                    raise ValueError("Invalid probability distribution")
                
                real_prob = float(probs[0])
                fake_prob = float(probs[1])
            
            # Calculate weighted probabilities
            model_weight = 0.7  # Give more weight to model prediction
            characteristics_weight = 0.3  # Less weight to image characteristics
            
            # Calculate final probabilities
            final_real_prob = (real_prob * model_weight) + ((real_score / 1.2) * characteristics_weight)
            final_fake_prob = (fake_prob * model_weight) + ((fake_score / 1.8) * characteristics_weight)
            
            # Normalize probabilities
            total_prob = final_real_prob + final_fake_prob
            final_real_prob = final_real_prob / total_prob
            final_fake_prob = final_fake_prob / total_prob
            
            # Decision making with strict criteria for fake detection
            is_fake = False
            if final_fake_prob > 0.65:  # High confidence fake
                is_fake = True
            elif final_fake_prob > 0.45 and fake_score > 0.6:  # Medium confidence but strong characteristics
                is_fake = True
            elif fake_prob > 0.8:  # Very high model confidence
                is_fake = True
            
            if is_fake:
                confidence = min(98.0, final_fake_prob * 100)
                final_label = 'Fake'
            else:
                confidence = min(98.0, final_real_prob * 100)
                final_label = 'Real'
            
            # Record metrics
            end_time = time.time()
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            await self._create_performance_metric(
                latency=end_time - start_time,
                memory_usage=memory_end - memory_start,
                success_rate=1.0
            )
            
            return final_label, confidence, []
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            await self._create_performance_metric(
                latency=time.time() - start_time,
                memory_usage=0,
                success_rate=0.0
            )
            raise ValueError(f"Failed to process image: {str(e)}")

    @property
    def version_info(self) -> ModelVersion:
        return self.model_version

class EnsembleDetector(BaseDeepfakeDetector):
    def __init__(self, detectors: List[BaseDeepfakeDetector]):
        self.detectors = detectors
        
    def preprocess(self, image) -> torch.Tensor:
        """Preprocess image using the first detector's preprocessing method"""
        if not self.detectors:
            raise ValueError("No detectors available in ensemble")
        return self.detectors[0].preprocess(image)
        
    @sync_to_async
    def _create_performance_metric(self, detector, latency, memory_usage, success_rate):
        """Create performance metric in database"""
        return PerformanceMetric.objects.create(
            model_version=detector.version_info,
            latency=latency,
            memory_usage=memory_usage,
            batch_size=1,
            success_rate=success_rate
        )
        
    def _calibrate_confidence(self, confidence: float, is_fake: bool) -> float:
        """Balanced confidence calibration"""
        if is_fake:
            # More conservative for fake predictions
            scaled = (confidence - 60) * 0.1  # Higher threshold
            calibrated = 50 + (50 * np.tanh(scaled))
        else:
            # More generous for real predictions
            scaled = (confidence - 40) * 0.12  # Lower threshold
            calibrated = 50 + (50 * np.tanh(scaled))
        return float(calibrated)
        
    async def predict(self, image) -> Tuple[str, float, List[Dict]]:
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        predictions = []
        confidences = []
        fake_votes = 0
        total_fake_prob = 0.0
        total_real_prob = 0.0
        
        try:
            input_tensor = self.preprocess(image)
            
            # Get predictions from all detectors
            detector_results = []
            for detector in self.detectors:
                with torch.no_grad():
                    output = detector.model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    
                fake_prob = float(probs[1])
                real_prob = float(probs[0])
                
                # Get image characteristics scores
                real_score, fake_score = detector._analyze_image_characteristics(detector.image_stats)
                
                detector_results.append({
                    'fake_prob': fake_prob,
                    'real_prob': real_prob,
                    'fake_score': fake_score,
                    'real_score': real_score
                })
                
                total_fake_prob += fake_prob
                total_real_prob += real_prob
            
            # Calculate averages
            avg_fake_prob = total_fake_prob / len(self.detectors)
            avg_real_prob = total_real_prob / len(self.detectors)
            
            # Calculate characteristics scores
            avg_fake_score = sum(r['fake_score'] for r in detector_results) / len(detector_results)
            avg_real_score = sum(r['real_score'] for r in detector_results) / len(detector_results)
            
            # Weighted ensemble decision
            model_weight = 0.7
            characteristics_weight = 0.3
            
            final_fake_prob = (avg_fake_prob * model_weight) + ((avg_fake_score / 1.8) * characteristics_weight)
            final_real_prob = (avg_real_prob * model_weight) + ((avg_real_score / 1.2) * characteristics_weight)
            
            # Normalize probabilities
            total = final_fake_prob + final_real_prob
            final_fake_prob = final_fake_prob / total
            final_real_prob = final_real_prob / total
            
            # Strict criteria for fake detection
            is_fake = False
            if final_fake_prob > 0.65:  # High confidence fake
                is_fake = True
            elif final_fake_prob > 0.45 and avg_fake_score > 0.6:  # Medium confidence but strong characteristics
                is_fake = True
            elif avg_fake_prob > 0.8:  # Very high average model confidence
                is_fake = True
            
            if is_fake:
                final_label = 'Fake'
                final_confidence = min(98.0, final_fake_prob * 100)
            else:
                final_label = 'Real'
                final_confidence = min(98.0, final_real_prob * 100)
            
            # Record performance
            end_time = time.time()
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            await asyncio.gather(*[
                self._create_performance_metric(
                    detector=detector,
                    latency=end_time - start_time,
                    memory_usage=(memory_end - memory_start) / len(self.detectors),
                    success_rate=1.0
                )
                for detector in self.detectors
            ])
            
            return final_label, final_confidence, []
            
        except Exception as e:
            await asyncio.gather(*[
                self._create_performance_metric(
                    detector=detector,
                    latency=time.time() - start_time,
                    memory_usage=0,
                    success_rate=0.0
                )
                for detector in self.detectors
            ])
            raise ValueError(f"Failed to process image: {str(e)}")

    @property
    def version_info(self) -> ModelVersion:
        # Return the latest version among all detectors
        return max((d.version_info for d in self.detectors), key=lambda x: x.created_at) 