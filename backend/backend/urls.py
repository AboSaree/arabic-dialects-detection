"""
URL configuration for the Arabic Dialect Identifier backend.
"""
from django.urls import path, include

urlpatterns = [
    path('api/', include('dialect_api.urls')),
]
