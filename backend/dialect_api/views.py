"""
Core views for the Arabic Dialect Identifier API.

POST /api/analyze/  — accepts audio file, returns dialect prediction + visualizations
GET  /api/health/   — health check
"""
import os
import io
import base64
import tempfile
import traceback

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

# ── Load model & scaler once at module import ──────────────────────────────────
_model = None
_scaler = None

DIALECT_LABELS = {
    'Egyptian':  {'code': 'EGY', 'color': '#E74C3C', 'flag': '🇪🇬'},
    'Moroccan':  {'code': 'MOR', 'color': '#27AE60', 'flag': '🇲🇦'},
    'Jordanian': {'code': 'JOR', 'color': '#2980B9', 'flag': '🇯🇴'},
    'Lebanese':  {'code': 'LEB', 'color': '#8E44AD', 'flag': '🇱🇧'},
}

DIALECT_INFO = {
    'Egyptian': {
        'description': 'Egyptian Arabic (Masri) is spoken by ~100 million people and is the most widely understood dialect across the Arab world due to Egypt\'s influential media industry.',
        'region': 'Egypt, North Africa',
        'speakers': '~100 million',
        'characteristics': 'Known for "g" replacing "q", soft intonation, and rich colloquial expressions.',
    },
    'Moroccan': {
        'description': 'Moroccan Arabic (Darija) is a unique dialect heavily influenced by Berber, French, and Spanish. It\'s often considered the most distinct Arabic dialect.',
        'region': 'Morocco, North Africa',
        'speakers': '~35 million',
        'characteristics': 'Rapid speech, consonant clusters, heavy Berber and French loanwords.',
    },
    'Jordanian': {
        'description': 'Jordanian Arabic is a Levantine dialect spoken in Jordan. It shares many features with Palestinian and Syrian Arabic.',
        'region': 'Jordan, Levant',
        'speakers': '~10 million',
        'characteristics': 'Clear pronunciation, "q" often pronounced as a glottal stop, similar to Palestinian Arabic.',
    },
    'Lebanese': {
        'description': 'Lebanese Arabic is a vibrant Levantine dialect known for its musicality and heavy French influence, spoken in Lebanon.',
        'region': 'Lebanon, Levant',
        'speakers': '~4 million native + diaspora',
        'characteristics': 'Musical intonation, French loanwords, distinctive vowel sounds.',
    },
}


def _load_models():
    global _model, _scaler
    if _model is None:
        model_path = str(settings.MODEL_PATH)
        scaler_path = str(settings.SCALER_PATH)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _model = joblib.load(model_path)
            _scaler = joblib.load(scaler_path)
            print(f"[DialectAPI] Model loaded from {model_path}")
        else:
            print(f"[DialectAPI] WARNING: Model files not found at {model_path}")
    return _model, _scaler


def _fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _extract_features(y, sr):
    """Extract the same 34 features used during training."""
    # MFCCs (13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Spectral Contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    # Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    # RMS Energy (1)
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    features = np.hstack([mfccs_mean, chroma_mean, contrast_mean, zcr_mean, rms_mean])
    return features, {
        'mfccs': mfccs,
        'mfccs_mean': mfccs_mean.tolist(),
        'chroma': chroma,
        'chroma_mean': chroma_mean.tolist(),
        'contrast': contrast,
        'contrast_mean': contrast_mean.tolist(),
        'zcr': zcr,
        'zcr_mean': zcr_mean,
        'rms': rms,
        'rms_mean': rms_mean,
    }


def _make_spectrogram(y, sr, dialect_color='#D4AF37'):
    """Generate mel spectrogram figure (dark theme)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                   ax=ax, cmap='magma')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#AAAAAA')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#AAAAAA')
    cbar.set_label('Power (dB)', color='#AAAAAA')

    ax.set_title('Mel Spectrogram', color=dialect_color, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Time (s)', color='#AAAAAA')
    ax.set_ylabel('Frequency (Hz)', color='#AAAAAA')
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    return _fig_to_b64(fig)


def _make_mfcc_plot(mfccs, mfccs_mean, sr, dialect_color='#D4AF37'):
    """Generate MFCC heatmap + mean bar chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0D1117')

    # MFCC heatmap
    ax1.set_facecolor('#0D1117')
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax1, cmap='RdYlBu_r')
    cbar = fig.colorbar(img, ax=ax1)
    cbar.ax.yaxis.set_tick_params(color='#AAAAAA')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#AAAAAA')
    ax1.set_title('MFCC Heatmap (13 Coefficients)', color=dialect_color,
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('MFCC Coefficient', color='#AAAAAA')
    ax1.tick_params(colors='#AAAAAA')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')

    # MFCC means bar chart
    ax2.set_facecolor('#111827')
    x = np.arange(len(mfccs_mean))
    colors = [dialect_color if v >= 0 else '#FF6B6B' for v in mfccs_mean]
    bars = ax2.bar(x, mfccs_mean, color=colors, alpha=0.85, edgecolor='#333333')
    ax2.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'M{i+1}' for i in x], color='#AAAAAA', fontsize=8)
    ax2.set_title('Mean MFCC Values (Decision Features)', color='#CCCCCC', fontsize=11)
    ax2.set_ylabel('Mean Value', color='#AAAAAA', fontsize=9)
    ax2.tick_params(colors='#AAAAAA')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout(pad=1.5)
    return _fig_to_b64(fig)


def _make_features_plot(feature_data, dialect_name, dialect_color='#D4AF37'):
    """Generate chroma + spectral contrast + ZCR/RMS overview."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle(f'Distinguishing Features — {dialect_name} Dialect',
                 color=dialect_color, fontsize=13, fontweight='bold', y=1.02)

    # ── Chroma features ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#111827')
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma_vals = feature_data['chroma_mean']
    bars = ax.barh(notes, chroma_vals, color='#00D4AA', alpha=0.8, edgecolor='#1A2A3A')
    ax.set_title('Chroma Features\n(Tonal Fingerprint)', color='#CCCCCC', fontsize=11)
    ax.set_xlabel('Mean Energy', color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA')
    ax.set_xlim(0, max(chroma_vals) * 1.2 if max(chroma_vals) > 0 else 1)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    # ── Spectral Contrast ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('#111827')
    contrast_vals = feature_data['contrast_mean']
    bands = [f'Band {i+1}' for i in range(len(contrast_vals))]
    ax.bar(bands, contrast_vals, color='#FF9500', alpha=0.85, edgecolor='#333333')
    ax.set_title('Spectral Contrast\n(Peak vs Valley Energy)', color='#CCCCCC', fontsize=11)
    ax.set_xlabel('Sub-band', color='#AAAAAA', fontsize=9)
    ax.set_ylabel('Contrast (dB)', color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA', axis='x', rotation=30, labelsize=8)
    ax.tick_params(colors='#AAAAAA', axis='y')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    # ── ZCR + RMS gauges ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor('#111827')
    labels = ['ZCR\n(Sharpness)', 'RMS\n(Loudness)']
    values = [feature_data['zcr_mean'], feature_data['rms_mean']]
    colors_g = ['#A855F7', '#EC4899']
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors_g, alpha=0.85, edgecolor='#333333', width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#AAAAAA', fontsize=10)
    ax.set_title('Speech Dynamics\n(ZCR & Energy)', color='#CCCCCC', fontsize=11)
    ax.set_ylabel('Mean Value', color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', color='#CCCCCC', fontsize=9)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _make_waveform_plot(y, sr, dialect_color='#D4AF37'):
    """Generate audio waveform figure."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    times = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(times, y, alpha=0.6, color=dialect_color)
    ax.plot(times, y, color=dialect_color, linewidth=0.5, alpha=0.9)
    ax.axhline(0, color='#444444', linewidth=0.5)
    ax.set_title('Audio Waveform', color=dialect_color, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', color='#AAAAAA')
    ax.set_ylabel('Amplitude', color='#AAAAAA')
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    return _fig_to_b64(fig)


@api_view(['GET'])
def health_check(request):
    """Simple health check endpoint."""
    model, scaler = _load_models()
    return Response({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
    })


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_audio(request):
    """
    Analyze an uploaded audio file and return dialect prediction + visualizations.

    Expected: multipart/form-data with 'audio' file field
    Returns: JSON with prediction, probabilities, and base64-encoded plots
    """
    if 'audio' not in request.FILES:
        return Response(
            {'error': 'No audio file provided. Use the "audio" field.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    audio_file = request.FILES['audio']
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg',
                     'audio/flac', 'audio/x-wav', 'audio/wave', 'application/octet-stream']
    # We allow any content type and let librosa handle it

    try:
        model, scaler = _load_models()
        if model is None or scaler is None:
            return Response(
                {'error': 'ML model not found. Please ensure dialect_model.pkl and scaler.pkl exist.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Save uploaded file to a temp file for librosa
        suffix = os.path.splitext(audio_file.name)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            # Load audio with librosa
            y, sr = librosa.load(tmp_path, sr=22050, duration=35)
        finally:
            os.unlink(tmp_path)

        if len(y) < sr:  # Less than 1 second
            return Response(
                {'error': 'Audio file too short. Please provide at least 1 second of audio.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract features
        features, raw = _extract_features(y, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        predicted_label = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        class_names = model.classes_

        prob_dict = {cls: float(prob) for cls, prob in zip(class_names, probabilities)}
        confidence = float(prob_dict.get(predicted_label, 0))

        dialect_meta = DIALECT_LABELS.get(predicted_label, {
            'code': '???', 'color': '#D4AF37', 'flag': '🌍'
        })
        dialect_color = dialect_meta['color']

        # Generate visualizations
        spectrogram_b64 = _make_spectrogram(y, sr, dialect_color)
        mfcc_plot_b64 = _make_mfcc_plot(raw['mfccs'], raw['mfccs_mean'], sr, dialect_color)
        features_plot_b64 = _make_features_plot(raw, predicted_label, dialect_color)
        waveform_b64 = _make_waveform_plot(y, sr, dialect_color)

        # Dialect info
        info = DIALECT_INFO.get(predicted_label, {})

        return Response({
            'predicted_dialect': predicted_label,
            'dialect_code': dialect_meta['code'],
            'dialect_flag': dialect_meta['flag'],
            'dialect_color': dialect_color,
            'confidence': round(confidence * 100, 1),
            'probabilities': {k: round(v * 100, 1) for k, v in prob_dict.items()},
            'audio_info': {
                'duration': round(len(y) / sr, 2),
                'sample_rate': sr,
                'filename': audio_file.name,
            },
            'feature_data': {
                'mfcc_means': raw['mfccs_mean'],
                'chroma_means': raw['chroma_mean'],
                'spectral_contrast_means': raw['contrast_mean'],
                'zcr_mean': raw['zcr_mean'],
                'rms_mean': raw['rms_mean'],
            },
            'plots': {
                'waveform': waveform_b64,
                'spectrogram': spectrogram_b64,
                'mfcc': mfcc_plot_b64,
                'features': features_plot_b64,
            },
            'dialect_info': info,
        })

    except Exception as e:
        traceback.print_exc()
        return Response(
            {'error': f'Analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
