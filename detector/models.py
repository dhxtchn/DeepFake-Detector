from django.db import models
from django.contrib.auth.models import User
import uuid

class ModelVersion(models.Model):
    version_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField()
    file_path = models.CharField(max_length=255)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField(default=0.0)
    false_positive_rate = models.FloatField(default=0.0)
    false_negative_rate = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.version_id})"

class Detection(models.Model):
    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='detections/')
    video = models.FileField(upload_to='videos/', null=True, blank=True)
    label = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    processing_time = models.FloatField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    model_version = models.ForeignKey(ModelVersion, on_delete=models.SET_NULL, null=True)
    
    # Detailed analysis results
    boxes = models.JSONField(blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.label} ({self.confidence:.2f}%)"

class PerformanceMetric(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    latency = models.FloatField()
    memory_usage = models.FloatField()
    batch_size = models.IntegerField()
    success_rate = models.FloatField()
    
    class Meta:
        ordering = ['-timestamp']

class APIKey(models.Model):
    key = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    rate_limit = models.IntegerField(default=100)  # requests per hour
    is_development = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.name} ({self.user.username if self.user else 'system'})"

    class Meta:
        verbose_name = 'API Key'
        verbose_name_plural = 'API Keys'
        ordering = ['-created_at']