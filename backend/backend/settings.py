"""
Django settings for the Arabic Dialect Identifier backend.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-arabic-dialect-identifier-secret-key-2024'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'dialect_api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS settings - allow Angular dev server
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.JSONParser',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
    'UNAUTHENTICATED_USER': None,
    'UNAUTHENTICATED_TOKEN': None,
}

# Model paths
MODEL_PATH = BASE_DIR.parent / 'dialect_model.pkl'
SCALER_PATH = BASE_DIR.parent / 'scaler.pkl'

# ── HuggingFace token (for Fanar dialect conversion) ──────────────────────────
# Reads from backend/.env first, then falls back to system env HF_TOKEN.
_env_file = BASE_DIR / '.env'
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

HF_TOKEN = os.environ.get('HF_TOKEN', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', '')
# File upload settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
