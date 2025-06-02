from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Avg, Count
from .models import ModelVersion, Detection, PerformanceMetric, APIKey
from django.utils import timezone
from django.core.cache import cache

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('name', 'version_id', 'is_active', 'created_at', 
                   'accuracy', 'false_positive_rate', 'false_negative_rate',
                   'avg_latency', 'success_rate')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = ('version_id', 'created_at')
    
    def avg_latency(self, obj):
        avg = PerformanceMetric.objects.filter(model_version=obj)\
            .aggregate(Avg('latency'))['latency__avg']
        return f"{avg:.2f}ms" if avg else "N/A"
    
    def success_rate(self, obj):
        avg = PerformanceMetric.objects.filter(model_version=obj)\
            .aggregate(Avg('success_rate'))['success_rate__avg']
        return f"{avg*100:.1f}%" if avg else "N/A"

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'thumbnail', 'label', 'confidence', 'status',
                   'processing_time', 'created_at')
    list_filter = ('status', 'label', 'created_at')
    search_fields = ('id', 'label')
    readonly_fields = ('id', 'created_at', 'processing_time')
    
    def thumbnail(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="max-height: 50px;"/>', 
                             obj.image.url)
        return "No image"

@admin.register(PerformanceMetric)
class PerformanceMetricAdmin(admin.ModelAdmin):
    list_display = ('model_version', 'timestamp', 'latency', 'memory_usage',
                   'batch_size', 'success_rate')
    list_filter = ('model_version', 'timestamp')
    date_hierarchy = 'timestamp'

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'is_active', 'created_at', 'last_used',
                   'rate_limit', 'usage_count')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'user__username')
    readonly_fields = ('key', 'created_at', 'last_used')
    
    def usage_count(self, obj):
        hour_ago = timezone.now() - timezone.timedelta(hours=1)
        count = cache.get(f'rate_limit_{obj.key}', 0)
        return f"{count} requests/hour"
