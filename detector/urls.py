from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='home'),
    path('live/', views.webcam_live_view, name='live'),
    path('api/detect/', views.live_frame_api, name='live_detect_api'),
    path('api/metrics/', views.model_metrics, name='model_metrics'),
    path('api/batch/', views.batch_process, name='batch_process'),
    path('dashboard/', views.model_dashboard, name='dashboard'),
] 