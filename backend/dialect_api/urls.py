from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_audio, name='analyze_audio'),
    path('health/', views.health_check, name='health_check'),
]
