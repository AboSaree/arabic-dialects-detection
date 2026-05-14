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
    'Egyptian Arabic':  {'code': 'EGY', 'color': '#CE1126', 'flag': '', 'display': 'Egyptian'},
    'Gulf Arabic':      {'code': 'GLF', 'color': '#009736', 'flag': '', 'display': 'Gulf'},
    'Levantine Arabic': {'code': 'LEV', 'color': '#007A3D', 'flag': '', 'display': 'Levantine'},
    'Maghrebi Arabic':  {'code': 'MAG', 'color': '#C1272D', 'flag': '', 'display': 'Maghrebi'},
}

DIALECT_INFO = {
    'Egyptian Arabic': {
        'description': 'Egyptian Arabic is the most widely spoken and understood dialect in the Arab world, largely due to the historic influence of Egyptian cinema and media.',
        'region': 'Egypt, North Africa',
        'speakers': '~100 million',
        'characteristics': 'Pronunciation of "jeem" as a hard "g", distinct vowel shortenings, and unique vocabulary.',
    },
    'Gulf Arabic': {
        'description': 'Gulf Arabic is spoken across the Arabian Peninsula countries including Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, and Oman. It retains many classical Arabic features.',
        'region': 'Arabian Peninsula (Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, Oman)',
        'speakers': '~36 million',
        'characteristics': 'Preservation of classical sounds, distinct intonation, Bedouin vocabulary, and loanwords from Persian and English.',
    },
    'Levantine Arabic': {
        'description': 'Levantine Arabic is spoken in the Levant region including Syria, Lebanon, Palestine, and Jordan. It is known for its musicality and soft consonant pronunciation.',
        'region': 'Syria, Lebanon, Palestine, Jordan',
        'speakers': '~35 million',
        'characteristics': 'Soft consonant sounds, musical intonation, French and Aramaic loanwords, and distinctive vowel shifts.',
    },
    'Maghrebi Arabic': {
        'description': 'Maghrebi Arabic (Darija) is spoken across North Africa including Morocco, Algeria, Tunisia, and Libya. It is heavily influenced by Berber, French, and Spanish.',
        'region': 'Morocco, Algeria, Tunisia, Libya',
        'speakers': '~75 million',
        'characteristics': 'Rapid speech, consonant clusters, heavy Berber and French loanwords, often mutually unintelligible with eastern dialects.',
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
    """Extract the same 222 features used during MADIS5 training."""
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 40 MFCCs mean + std (80)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std  = np.std(mfccs.T, axis=0)

    # Delta MFCCs mean (40)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mean  = np.mean(delta_mfccs.T, axis=0)

    # Delta-Delta MFCCs mean (40)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mean  = np.mean(delta2_mfccs.T, axis=0)

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Spectral Contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    # Mel Spectrogram (40)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = np.mean(librosa.power_to_db(mel).T, axis=0)

    # Spectral Rolloff (1)
    rolloff_mean = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    # RMS Energy (1)
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    # Total: 40+40+40+40+12+7+40+1+1+1 = 222 features
    features = np.hstack([
        mfccs_mean, mfccs_std, delta_mean, delta2_mean,
        chroma_mean, contrast_mean, mel_mean,
        [rolloff_mean, zcr_mean, rms_mean]
    ])

    return features, {
        'mfccs': mfccs,
        'mfccs_mean': mfccs_mean.tolist(),
        'chroma': chroma,
        'chroma_mean': chroma_mean.tolist(),
        'contrast': contrast,
        'contrast_mean': contrast_mean.tolist(),
        'zcr': zcr,
        'zcr_raw': (zcr[0] if zcr.ndim > 1 else zcr).tolist(),
        'zcr_mean': zcr_mean,
        'rms': rms,
        'rms_raw': (rms[0] if rms.ndim > 1 else rms).tolist(),
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
    ax1.set_title('MFCC Heatmap (40 Coefficients)', color=dialect_color,
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


def _make_feature_line_plot(values, title, ylabel, dialect_color='#D4AF37'):
    """Generate a line plot for frame-based features such as RMS and ZCR."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#111827')

    frames = np.arange(len(values))
    ax.plot(frames, values, color=dialect_color, linewidth=1.8)
    ax.fill_between(frames, values, color=dialect_color, alpha=0.16)
    ax.set_title(title, color='#F8FAFC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frames', color='#AAAAAA')
    ax.set_ylabel(ylabel, color='#AAAAAA')
    ax.grid(True, color='#333333', alpha=0.45, linewidth=0.7)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    return _fig_to_b64(fig)


def _make_feature_heatmap(matrix, title, ylabel, y_labels=None):
    """Generate a heatmap for matrix features such as chroma and spectral contrast."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#111827')

    matrix = np.asarray(matrix)
    img = ax.imshow(matrix, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
    ax.set_title(title, color='#F8FAFC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frames', color='#AAAAAA')
    ax.set_ylabel(ylabel, color='#AAAAAA')
    ax.tick_params(colors='#AAAAAA')

    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors='#AAAAAA')
    cbar.outline.set_edgecolor('#333333')

    plt.tight_layout()
    return _fig_to_b64(fig)


def _save_upload_to_temp(audio_file):
    suffix = os.path.splitext(audio_file.name)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        return tmp.name


def _trim_and_normalize(y):
    trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(trimmed) == 0:
        return trimmed
    return trimmed / (np.max(np.abs(trimmed)) + 1e-9)


def _build_analysis_payload(y, sr, filename):
    model, scaler = _load_models()
    features, raw = _extract_features(y, sr)
    features_scaled = scaler.transform(features.reshape(1, -1))

    predicted_label = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    class_names = model.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(class_names, probabilities)}
    confidence = float(prob_dict.get(predicted_label, 0))

    dialect_meta = DIALECT_LABELS.get(predicted_label, {
        'code': '???', 'color': '#D4AF37', 'flag': '🌍', 'display': predicted_label
    })
    dialect_color = dialect_meta['color']
    display_name = dialect_meta.get('display', predicted_label)

    display_map = {k: v.get('display', k) for k, v in DIALECT_LABELS.items()}

    spectrogram_b64 = _make_spectrogram(y, sr, dialect_color)
    mfcc_plot_b64 = _make_mfcc_plot(raw['mfccs'], raw['mfccs_mean'], sr, dialect_color)
    features_plot_b64 = _make_features_plot(raw, display_name, dialect_color)
    waveform_b64 = _make_waveform_plot(y, sr, dialect_color)
    rms_plot_b64 = _make_feature_line_plot(raw['rms_raw'], f'RMS Energy - {display_name} Arabic', 'Energy', dialect_color)
    zcr_plot_b64 = _make_feature_line_plot(raw['zcr_raw'], f'Zero Crossing Rate - {display_name} Arabic', 'ZCR', dialect_color)
    spectral_contrast_plot_b64 = _make_feature_heatmap(
        raw['contrast'],
        f'Spectral Contrast - {display_name} Arabic',
        'Frequency Bands',
        [f'Band {i + 1}' for i in range(raw['contrast'].shape[0])]
    )
    chroma_plot_b64 = _make_feature_heatmap(
        raw['chroma'],
        f'Chroma Features - {display_name} Arabic',
        'Pitch class',
        ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    )

    info = DIALECT_INFO.get(predicted_label, {})

    return {
        'predicted_dialect': display_name,
        'dialect_code': dialect_meta['code'],
        'dialect_flag': dialect_meta['flag'],
        'dialect_color': dialect_color,
        'confidence': round(confidence * 100, 1),
        'probabilities': {display_map.get(k, k): round(v * 100, 1) for k, v in prob_dict.items()},
        'audio_info': {
            'duration': round(len(y) / sr, 2),
            'sample_rate': sr,
            'filename': filename,
        },
        'feature_data': {
            'mfcc_means': raw['mfccs_mean'],
            'chroma_means': raw['chroma_mean'],
            'spectral_contrast_means': raw['contrast_mean'],
            'zcr_mean': raw['zcr_mean'],
            'zcr_raw': raw['zcr_raw'],
            'rms_mean': raw['rms_mean'],
            'rms_raw': raw['rms_raw'],
        },
        'plots': {
            'waveform': waveform_b64,
            'spectrogram': spectrogram_b64,
            'mfcc': mfcc_plot_b64,
            'features': features_plot_b64,
            'rms': rms_plot_b64,
            'zcr': zcr_plot_b64,
            'spectral_contrast': spectral_contrast_plot_b64,
            'chroma': chroma_plot_b64,
        },
        'dialect_info': info,
    }


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
            'code': '???', 'color': '#D4AF37', 'flag': '🌍', 'display': predicted_label
        })
        dialect_color = dialect_meta['color']
        display_name = dialect_meta.get('display', predicted_label)

        # Build a mapping from raw model labels to display names
        display_map = {k: v.get('display', k) for k, v in DIALECT_LABELS.items()}

        # Generate visualizations
        spectrogram_b64 = _make_spectrogram(y, sr, dialect_color)
        mfcc_plot_b64 = _make_mfcc_plot(raw['mfccs'], raw['mfccs_mean'], sr, dialect_color)
        features_plot_b64 = _make_features_plot(raw, display_name, dialect_color)
        waveform_b64 = _make_waveform_plot(y, sr, dialect_color)
        rms_plot_b64 = _make_feature_line_plot(raw['rms_raw'], f'RMS Energy - {display_name} Arabic', 'Energy', dialect_color)
        zcr_plot_b64 = _make_feature_line_plot(raw['zcr_raw'], f'Zero Crossing Rate - {display_name} Arabic', 'ZCR', dialect_color)
        spectral_contrast_plot_b64 = _make_feature_heatmap(
            raw['contrast'],
            f'Spectral Contrast - {display_name} Arabic',
            'Frequency Bands',
            [f'Band {i + 1}' for i in range(raw['contrast'].shape[0])]
        )
        chroma_plot_b64 = _make_feature_heatmap(
            raw['chroma'],
            f'Chroma Features - {display_name} Arabic',
            'Pitch class',
            ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        )

        # Dialect info
        info = DIALECT_INFO.get(predicted_label, {})

        return Response({
            'predicted_dialect': display_name,
            'dialect_code': dialect_meta['code'],
            'dialect_flag': dialect_meta['flag'],
            'dialect_color': dialect_color,
            'confidence': round(confidence * 100, 1),
            'probabilities': {display_map.get(k, k): round(v * 100, 1) for k, v in prob_dict.items()},
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
                'zcr_raw': raw['zcr_raw'],
                'rms_mean': raw['rms_mean'],
                'rms_raw': raw['rms_raw'],
            },
            'plots': {
                'waveform': waveform_b64,
                'spectrogram': spectrogram_b64,
                'mfcc': mfcc_plot_b64,
                'features': features_plot_b64,
                'rms': rms_plot_b64,
                'zcr': zcr_plot_b64,
                'spectral_contrast': spectral_contrast_plot_b64,
                'chroma': chroma_plot_b64,
            },
            'dialect_info': info,
        })

    except Exception as e:
        traceback.print_exc()
        return Response(
            {'error': f'Analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_mixed_audio(request):
    if 'audio1' not in request.FILES or 'audio2' not in request.FILES:
        return Response(
            {'error': 'Two audio files are required. Use the "audio1" and "audio2" fields.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if 'weight' not in request.POST:
        return Response(
            {'error': 'Mix weight is required. Use the "weight" field.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    audio_file_1 = request.FILES['audio1']
    audio_file_2 = request.FILES['audio2']
    mix_method = request.POST.get('mix_method', 'weighted')

    try:
        weight = float(request.POST['weight'])
    except ValueError:
        return Response(
            {'error': 'Invalid mix weight. Expected a number between 0.0 and 1.0.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if weight < 0.0 or weight > 1.0:
        return Response(
            {'error': 'Invalid mix weight. Expected a number between 0.0 and 1.0.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        model, scaler = _load_models()
        if model is None or scaler is None:
            return Response(
                {'error': 'ML model not found. Please ensure dialect_model.pkl and scaler.pkl exist.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        tmp_path_1 = _save_upload_to_temp(audio_file_1)
        tmp_path_2 = _save_upload_to_temp(audio_file_2)

        try:
            y1, sr = librosa.load(tmp_path_1, sr=22050, duration=35)
            y2, _ = librosa.load(tmp_path_2, sr=22050, duration=35)
        finally:
            os.unlink(tmp_path_1)
            os.unlink(tmp_path_2)

        y1 = _trim_and_normalize(y1)
        y2 = _trim_and_normalize(y2)

        if len(y1) == 0 or len(y2) == 0:
            return Response(
                {'error': 'One of the audio files is silent after trimming. Please upload audible recordings.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]

        if mix_method == 'splice':
            splice_len_1 = int(min_length * weight)
            splice_len_2 = min_length - splice_len_1
            mixed = np.concatenate([y1[:splice_len_1], y2[:splice_len_2]])
        else:
            mixed = weight * y1 + (1 - weight) * y2

        mixed = mixed / (np.max(np.abs(mixed)) + 1e-9)

        if len(mixed) < sr:
            return Response(
                {'error': 'Mixed audio is too short. Please provide at least 1 second of audio in each file.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response(_build_analysis_payload(mixed, sr, f'{audio_file_1.name} + {audio_file_2.name}'))

    except Exception as e:
        traceback.print_exc()
        return Response(
            {'error': f'Analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )






# ── Dialect name → Arabic label used in prompts ───────────────────────────────
_DIALECT_AR = {
    'Egyptian':  'المصري',
    'Moroccan':  'المغربي (الدارجة)',
    'Iraqi':     'العراقي',
    'Lebanese':  'اللبناني',
    'Gulf':      'الخليجي',
    'MSA':       'الفصحى (اللغة العربية الفصحى)',
}


@api_view(['POST'])
def convert_dialect(request):
    """
    Convert Arabic text from one dialect to another using Google Gemini 2.5 Flash.

    Body (JSON):
        text           - Arabic text to convert
        source_dialect - e.g. "Egyptian"
        target_dialect - e.g. "Moroccan"

    Returns:
        { converted_text: str, source_dialect: str, target_dialect: str }
    """
    from groq import Groq

    text           = request.data.get('text', '').strip()
    source_dialect = request.data.get('source_dialect', '').strip()
    target_dialect = request.data.get('target_dialect', '').strip()

    if not text:
        return Response({'error': 'No text provided.'}, status=status.HTTP_400_BAD_REQUEST)
    if not target_dialect:
        return Response({'error': 'target_dialect is required.'}, status=status.HTTP_400_BAD_REQUEST)

    api_key = getattr(settings, 'GROQ_API_KEY', '').strip()
    if not api_key:
        return Response(
            {'error': 'GROQ_API_KEY is not set. Add it to backend/.env'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    src_ar = _DIALECT_AR.get(source_dialect, source_dialect) if source_dialect else 'غير محدد'
    tgt_ar = _DIALECT_AR.get(target_dialect, target_dialect)

    system_prompt = (
        "أنت خبير متخصص في اللهجات العربية. مهمتك الوحيدة هي تحويل النصوص العربية من لهجة إلى أخرى.\n"
        "قواعد صارمة يجب الالتزام بها:\n"
        "1. اكتب النص المحوَّل باللغة العربية فقط — لا إنجليزية أبداً.\n"
        "2. حافظ على المعنى الأصلي تماماً مع تغيير المفردات والتعبيرات لتناسب اللهجة المطلوبة.\n"
        "3. لا تضف أي تفسير أو تعليق أو مقدمة — النص المحوَّل فقط.\n"
        "4. إذا لم يكن للكلمة مقابل في اللهجة المطلوبة، استخدم الأقرب إليها."
    )

    user_prompt = (
        f"حوِّل النص التالي من اللهجة {src_ar} إلى اللهجة {tgt_ar}.\n"
        f"اكتب النص المحوَّل فقط بدون أي مقدمة أو تعليق.\n\n"
        f"النص ({src_ar}):\n{text}\n\n"
        f"النص بعد التحويل إلى {tgt_ar}:"
    )

    try:
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        converted_text = completion.choices[0].message.content.strip()

        return Response({
            "converted_text":  converted_text,
            "source_dialect":  source_dialect,
            "target_dialect":  target_dialect,
        })

    except Exception as e:
        err_str = str(e)
        traceback.print_exc()
        if '401' in err_str or 'invalid_api_key' in err_str.lower():
            return Response(
                {'error': 'Invalid Groq API key. Check GROQ_API_KEY in backend/.env'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        if '429' in err_str:
            return Response(
                {'error': 'Groq rate limit reached. Please wait a moment and try again.'},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        return Response(
            {'error': f'Dialect conversion failed: {err_str}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


    """
    Convert Arabic text from one dialect to another using Claude claude-sonnet-4-20250514.

    Body (JSON):
        text           - Arabic text to convert
        source_dialect - e.g. "Egyptian"
        target_dialect - e.g. "Moroccan"

    Returns:
        { converted_text: str, source_dialect: str, target_dialect: str }
    """
    import anthropic

    text           = request.data.get('text', '').strip()
    source_dialect = request.data.get('source_dialect', '').strip()
    target_dialect = request.data.get('target_dialect', '').strip()

    if not text:
        return Response({'error': 'No text provided.'}, status=status.HTTP_400_BAD_REQUEST)
    if not target_dialect:
        return Response({'error': 'target_dialect is required.'}, status=status.HTTP_400_BAD_REQUEST)

    api_key = getattr(settings, 'ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        return Response(
            {'error': 'ANTHROPIC_API_KEY is not set. Add it to backend/.env'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    src_ar = _DIALECT_AR.get(source_dialect, source_dialect) if source_dialect else 'غير محدد'
    tgt_ar = _DIALECT_AR.get(target_dialect, target_dialect)

    system_prompt = (
        "أنت خبير متخصص في اللهجات العربية. مهمتك الوحيدة هي تحويل النصوص العربية من لهجة إلى أخرى.\n"
        "قواعد صارمة يجب الالتزام بها:\n"
        "1. اكتب النص المحوَّل باللغة العربية فقط — لا إنجليزية أبداً.\n"
        "2. حافظ على المعنى الأصلي تماماً مع تغيير المفردات والتعبيرات لتناسب اللهجة المطلوبة.\n"
        "3. لا تضف أي تفسير أو تعليق أو مقدمة — النص المحوَّل فقط.\n"
        "4. إذا لم يكن للكلمة مقابل في اللهجة المطلوبة، استخدم الأقرب إليها."
    )

    user_prompt = (
        f"حوِّل النص التالي من اللهجة {src_ar} إلى اللهجة {tgt_ar}.\n"
        f"اكتب النص المحوَّل فقط بدون أي مقدمة أو تعليق.\n\n"
        f"النص ({src_ar}):\n{text}\n\n"
        f"النص بعد التحويل إلى {tgt_ar}:"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        converted_text = message.content[0].text.strip()

        return Response({
            "converted_text":  converted_text,
            "source_dialect":  source_dialect,
            "target_dialect":  target_dialect,
        })

    except anthropic.AuthenticationError:
        return Response(
            {'error': 'Invalid Anthropic API key. Check ANTHROPIC_API_KEY in backend/.env'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    except anthropic.RateLimitError:
        return Response(
            {'error': 'Anthropic rate limit reached. Please wait and try again.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )
    except Exception as e:
        traceback.print_exc()
        return Response(
            {'error': f'Dialect conversion failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


    """
    Convert Arabic text from one dialect to another using Fanar-1-9B-Instruct
    via the HuggingFace Inference Router (OpenAI-compatible endpoint).

    Body (JSON):
        text          - Arabic text to convert
        source_dialect - e.g. "Egyptian"
        target_dialect - e.g. "Moroccan"

    Returns:
        { converted_text: str, source_dialect: str, target_dialect: str }
    """
    text           = request.data.get('text', '').strip()
    source_dialect = request.data.get('source_dialect', '').strip()
    target_dialect = request.data.get('target_dialect', '').strip()

    if not text:
        return Response({'error': 'No text provided.'}, status=status.HTTP_400_BAD_REQUEST)
    if not target_dialect:
        return Response({'error': 'target_dialect is required.'}, status=status.HTTP_400_BAD_REQUEST)

    hf_token = getattr(settings, 'HF_TOKEN', '').strip()
    if not hf_token:
        return Response(
            {'error': 'HF_TOKEN environment variable is not set. '
                      'Please add your Hugging Face token to the backend environment.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    src_ar = _DIALECT_AR.get(source_dialect, source_dialect) if source_dialect else 'غير محدد'
    tgt_ar = _DIALECT_AR.get(target_dialect, target_dialect)

    system_prompt = (
        "You are an expert Arabic dialect translator. "
        "Your ONLY task is to rewrite Arabic text from one dialect into another dialect. "
        "STRICT RULES you must never break:\n"
        "1. Output ONLY Arabic text. Never write English or any other language.\n"
        "2. Do NOT translate to English. Do NOT explain. Do NOT add comments.\n"
        "3. Keep the exact same meaning but change vocabulary and expressions to match the target dialect.\n"
        "4. If you are unsure about a word, keep the closest Arabic equivalent in the target dialect.\n"
        "5. Your entire response must be Arabic script only."
    )

    user_prompt = (
        f"Convert the following Arabic text from {source_dialect} dialect to {target_dialect} dialect.\n"
        f"Output ONLY the converted Arabic text. No English. No explanation.\n\n"
        f"Input ({source_dialect}):\n{text}\n\n"
        f"Output ({target_dialect} Arabic):"
    )

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            provider="featherless-ai",   # Fanar is hosted here via HF router
            api_key=hf_token,
        )

        completion = client.chat.completions.create(
            model="QCRI/Fanar-1-9B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
        )

        converted_text = completion.choices[0].message.content.strip()

        # Sanity check: if less than 30% of letters are Arabic script, the model hallucinated
        arabic_chars = sum(1 for c in converted_text if '\u0600' <= c <= '\u06FF')
        total_letters = sum(1 for c in converted_text if c.isalpha())
        if total_letters > 0 and arabic_chars / total_letters < 0.3:
            return Response(
                {'error': 'Model returned non-Arabic output. Please try again — this is a known '
                          'occasional issue with the Fanar model on long texts.'},
                status=status.HTTP_502_BAD_GATEWAY
            )

        return Response({
            "converted_text":  converted_text,
            "source_dialect":  source_dialect,
            "target_dialect":  target_dialect,
        })

    except Exception as e:
        err_str = str(e)
        traceback.print_exc()
        if '401' in err_str or 'Unauthorized' in err_str:
            return Response(
                {'error': 'Invalid HuggingFace token. Check HF_TOKEN in backend/.env'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        if '429' in err_str:
            return Response(
                {'error': 'HuggingFace rate limit reached. Please wait and try again.'},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        return Response(
            {'error': f'Dialect conversion failed: {err_str}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def transcribe_audio(request):
    """
    Transcribe an uploaded Arabic audio file using Deepgram Nova-3.

    Expected: multipart/form-data with 'audio' file field
    Returns:  JSON  { words: [], full_text: str, text: str }
    """
    if 'audio' not in request.FILES:
        return Response(
            {'error': 'No audio file provided. Use the "audio" field.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    audio_file = request.FILES['audio']

    api_key = getattr(settings, 'DEEPGRAM_API_KEY', '').strip()
    if not api_key:
        return Response(
            {'error': 'DEEPGRAM_API_KEY is not set. Add it to backend/.env'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        import requests

        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': audio_file.content_type or 'audio/wav',
        }

        audio_bytes = b''.join(chunk for chunk in audio_file.chunks())
        resp = requests.post(
            'https://api.deepgram.com/v1/listen?model=nova-3&language=ar&punctuate=true&words=true',
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )

        if resp.status_code == 401:
            return Response(
                {'error': 'Invalid Deepgram API key. Check DEEPGRAM_API_KEY in backend/.env'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        if resp.status_code == 429:
            return Response(
                {'error': 'Deepgram rate limit reached. Please wait a moment and try again.'},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )

        resp.raise_for_status()

        alternative = resp.json()['results']['channels'][0]['alternatives'][0]
        transcript = alternative.get('transcript', '').strip()
        raw_words = alternative.get('words', [])

        words = [
            {
                'word':  w.get('word', ''),
                'start': float(w.get('start', 0)),
                'end':   float(w.get('end', 0)),
            }
            for w in raw_words
        ]

        if not transcript:
            return Response({'error': 'Transcription failed: empty response from Deepgram.'}, status=status.HTTP_502_BAD_GATEWAY)

        return Response({'words': words, 'full_text': transcript, 'text': transcript})

    except requests.RequestException as e:
        traceback.print_exc()
        return Response({'error': f'Transcription failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        traceback.print_exc()
        return Response({'error': f'Transcription failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



ELEVENLABS_VOICES = [
    {"id": "nPczCjzI2devNBz1zQrb", "name": "Mohamed",   "description": "Deep, Resonant and Comforting"},
    {"id": "cjVigY5qzO86Huf0OWal", "name": "Hussain",    "description": "Smooth, Trustworthy"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "3awadein",  "description": "Warm, Captivating Storyteller"},
    {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Emad",  "description": "Steady Broadcaster"},
    {"id": "XrExE9yKIg1WjnnlVkGX", "name": "samiha", "description": "Knowledgeable, Professional"},
]

_ELEVENLABS_VOICE_IDS = {voice["id"] for voice in ELEVENLABS_VOICES}


@api_view(['GET'])
def list_voices(request):
    return Response(ELEVENLABS_VOICES)


@api_view(['POST'])
def text_to_speech(request):
    """
    Convert Arabic text to speech using ElevenLabs API.

    Body (JSON):
        text   - Arabic text to synthesize
        dialect - Dialect name (Egyptian, Gulf, Levantine, Maghrebi)

    Returns:
        Audio stream (audio/mpeg)
    """
    from elevenlabs.client import ElevenLabs
    from django.http import HttpResponse

    text = request.data.get('text', '').strip()
    voice_id = request.data.get('voice_id', '').strip()

    if not text:
        return Response({'error': 'No text provided.'}, status=status.HTTP_400_BAD_REQUEST)

    api_key = getattr(settings, 'ELEVENLABS_API_KEY', '').strip()
    if not api_key:
        return Response(
            {'error': 'ELEVENLABS_API_KEY is not set. Add it to backend/.env'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    if voice_id not in _ELEVENLABS_VOICE_IDS:
        voice_id = 'nPczCjzI2devNBz1zQrb'

    try:
        client = ElevenLabs(api_key=api_key)

        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id='eleven_multilingual_v2'
        )

        # Convert generator to bytes
        audio_bytes = b''.join(audio_stream)

        return HttpResponse(audio_bytes, content_type='audio/mpeg')

    except Exception as e:
        err_str = str(e)
        traceback.print_exc()
        if '401' in err_str or 'invalid_api_key' in err_str.lower():
            return Response(
                {'error': 'Invalid ElevenLabs API key. Check ELEVENLABS_API_KEY in backend/.env'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        if '429' in err_str:
            return Response(
                {'error': 'ElevenLabs rate limit reached. Please wait a moment and try again.'},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        return Response(
            {'error': f'Text-to-speech failed: {err_str}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
