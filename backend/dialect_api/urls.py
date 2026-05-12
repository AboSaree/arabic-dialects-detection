from django.urls import path
from . import views

urlpatterns = [
    path('analyze/',    views.analyze_audio,   name='analyze_audio'),
    path('analyze-mix/', views.analyze_mixed_audio, name='analyze_mixed_audio'),
    path('transcribe/', views.transcribe_audio, name='transcribe_audio'),
    path('health/',         views.health_check,     name='health_check'),
    path('convert-dialect/', views.convert_dialect,  name='convert_dialect'),
    path('voices/', views.list_voices, name='list_voices'),
    path('tts/', views.text_to_speech, name='text_to_speech'),
]
