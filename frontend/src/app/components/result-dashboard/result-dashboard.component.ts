import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NgChartsModule } from 'ng2-charts';
import { ChartConfiguration, TooltipItem } from 'chart.js';
import 'chart.js/auto';
import { AnalysisResult, DialectConversionResult } from '../../models/analysis-result.interface';
import { DialectService, ElevenLabsVoice } from '../../services/dialect.service';

@Component({
  selector: 'app-result-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, NgChartsModule],
  templateUrl: './result-dashboard.component.html',
  styleUrls: ['./result-dashboard.component.css']
})
export class ResultDashboardComponent implements OnInit {
  @Input() result!: AnalysisResult;
  @Input() transcribedText: string = '';   // full_text from TranscriptionComponent
  @Output() reset = new EventEmitter<void>();

  activeTab: 'spectrogram' | 'mfcc' | 'features' | 'waveform' = 'spectrogram';
  activeFeatureTab: FeaturePlotTab = 'rms';
  probabilityEntries: { dialect: string; value: number; color: string }[] = [];
  animatedConfidence = 0;
  availableDialects = ['Egyptian', 'Gulf', 'Levantine', 'Maghrebi'];
  selectedTargetDialect = '';
  conversionState: 'idle' | 'loading' | 'done' | 'error' = 'idle';
  conversionResult: DialectConversionResult | null = null;
  conversionError = '';
  ttsState: 'idle' | 'loading' | 'playing' = 'idle';
  voices: ElevenLabsVoice[] = [];
  selectedVoiceId = 'nPczCjzI2devNBz1zQrb'; // Brian default
  currentAudio: HTMLAudioElement | null = null;
  currentAudioUrl: string | null = null;
  ttsDownloadUrl: string | null = null;
  ttsDownloadName = 'arabic-tts.mp3';

  // Chart data
  zcrChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  spectralChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  chromaChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };

  zcrChartOptions: ChartConfiguration['options'] = {};
  spectralChartOptions: ChartConfiguration['options'] = {};
  chromaChartOptions: ChartConfiguration['options'] = {};

  private dialectColors: Record<string, string> = {
    'Egyptian Arabic':  '#CE1126',
    'Gulf Arabic':      '#009736',
    'Levantine Arabic': '#007A3D',
    'Maghrebi Arabic':  '#C1272D',
  };

  constructor(private dialectService: DialectService) {}

  ngOnInit(): void {
    this.dialectService.getVoices().subscribe((voices) => {
      this.voices = voices;
    });

    // Build sorted probability entries
    this.probabilityEntries = Object.entries(this.result.probabilities)
      .map(([dialect, value]) => ({
        dialect,
        value,
        color: this.dialectColors[dialect] || '#D4AF37'
      }))
      .sort((a, b) => b.value - a.value);

    // Animate confidence ring
    setTimeout(() => {
      this.animatedConfidence = this.result.confidence;
    }, 200);

    // Initialize charts
    this.initializeCharts();
    this.initializeFeatureTab();
  }

  private initializeFeatureTab(): void {
    const tabs: FeaturePlotTab[] = ['rms', 'zcr', 'spectral_contrast', 'chroma'];
    this.activeFeatureTab = tabs.find((tab) => Boolean(this.result.plots[tab])) || 'rms';
  }

  private initializeCharts(): void {
    const dialectColor = this.result.dialect_color;
    
    // ── Zero Crossing Rate Chart ──
    if (this.result.feature_data?.zcr_raw && Array.isArray(this.result.feature_data.zcr_raw)) {
      this.zcrChartData = {
        labels: this.result.feature_data.zcr_raw.map((_: number, i: number) => `${i}`),
        datasets: [{
          label: 'Zero Crossing Rate',
          data: this.result.feature_data.zcr_raw,
          borderColor: dialectColor,
          backgroundColor: `${dialectColor}20`,
          fill: true,
          tension: 0.4,
          pointRadius: 2,
          pointBackgroundColor: dialectColor,
        }]
      };

      this.zcrChartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: '#333333' },
            ticks: { color: '#AAAAAA' }
          },
          x: { grid: { color: '#333333' }, ticks: { color: '#AAAAAA' } }
        }
      };
    }

    // ── Spectral Contrast Chart ──
    if (this.result.feature_data?.spectral_contrast_means && Array.isArray(this.result.feature_data.spectral_contrast_means)) {
      this.spectralChartData = {
        labels: Array.from({ length: this.result.feature_data.spectral_contrast_means.length }, (_, i) => `Band ${i + 1}`),
        datasets: [{
          label: 'Spectral Contrast (dB)',
          data: this.result.feature_data.spectral_contrast_means,
          backgroundColor: dialectColor,
          borderColor: dialectColor,
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: dialectColor,
        }]
      };

      this.spectralChartOptions = {
        indexAxis: 'y' as const,
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (context: TooltipItem<'bar'>) => `${(context.parsed.x ?? 0).toFixed(2)} dB` } }
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: '#333333' },
            ticks: { color: '#AAAAAA' }
          },
          y: { grid: { color: '#333333' }, ticks: { color: '#AAAAAA' } }
        }
      };
    }

    // ── Chroma Features Chart ──
    if (this.result.feature_data?.chroma_means && Array.isArray(this.result.feature_data.chroma_means)) {
      const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      this.chromaChartData = {
        labels: notes,
        datasets: [{
          label: 'Chroma Features',
          data: this.result.feature_data.chroma_means,
          backgroundColor: '#00D4AA',
          borderColor: '#00D4AA',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: '#00D4AA',
        }]
      };

      this.chromaChartOptions = {
        indexAxis: 'y' as const,
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (context: TooltipItem<'bar'>) => `Energy: ${(context.parsed.x ?? 0).toFixed(3)}` } }
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: '#333333' },
            ticks: { color: '#AAAAAA' }
          },
          y: { grid: { color: '#333333' }, ticks: { color: '#AAAAAA' } }
        }
      };
    }
  }

  setTab(tab: 'spectrogram' | 'mfcc' | 'features' | 'waveform'): void {
    this.activeTab = tab;
  }

  setFeatureTab(tab: FeaturePlotTab): void {
    this.activeFeatureTab = tab;
  }

  getDialectColor(dialect: string): string {
    return this.dialectColors[dialect] || '#D4AF37';
  }

  /** Convert 0-360 arc for SVG confidence ring */
  getArcPath(percent: number): string {
    const r = 54;
    const cx = 64, cy = 64;
    const angle = (percent / 100) * 360 - 90;
    const rad = (angle * Math.PI) / 180;
    const x = cx + r * Math.cos(rad);
    const y = cy + r * Math.sin(rad);
    const largeArc = percent > 50 ? 1 : 0;
    return `M ${cx} ${cy - r} A ${r} ${r} 0 ${largeArc} 1 ${x} ${y}`;
  }

  getCircumference(): number {
    return 2 * Math.PI * 54;
  }

  getDashOffset(): number {
    const c = this.getCircumference();
    return c - (this.animatedConfidence / 100) * c;
  }

  // ── Dialect Conversion ─────────────────────────────────────────────────────

  convertDialect(): void {
    const text = this.transcribedText || '';
    if (!text.trim() || !this.selectedTargetDialect) return;

    this.conversionState  = 'loading';
    this.conversionResult = null;
    this.conversionError  = '';
    this.ttsState         = 'idle';

    this.dialectService.convertDialect(
      text,
      this.result.predicted_dialect,
      this.selectedTargetDialect
    ).subscribe({
      next: (res) => {
        this.conversionResult = res;
        this.conversionState  = 'done';
      },
      error: (err) => {
        this.conversionError = err.message || 'Conversion failed.';
        this.conversionState = 'error';
      }
    });
  }

  selectVoice(voiceId: string): void {
    this.selectedVoiceId = voiceId;
    if (this.ttsState === 'playing' && this.currentAudio) {
      this.currentAudio.pause();
      this.ttsState = 'idle';
    }
  }

  /** TTS using ElevenLabs — synthesize and play converted text */
  speakConverted(): void {
    if (!this.conversionResult?.converted_text) return;

    if (this.ttsState === 'playing' && this.currentAudio) {
      this.currentAudio.pause();
      this.ttsState = 'idle';
      return;
    }

    this.ttsState = 'loading';

    if (this.currentAudioUrl) {
      URL.revokeObjectURL(this.currentAudioUrl);
      this.currentAudioUrl = null;
    }

    this.dialectService.synthesize(
      this.conversionResult.converted_text,
      this.selectedVoiceId
    ).subscribe({
      next: (audioBlob) => {
        const audioUrl = URL.createObjectURL(audioBlob);
        this.currentAudioUrl = audioUrl;
        this.ttsDownloadUrl = audioUrl;
        this.ttsDownloadName = `arabic-tts-${this.selectedVoiceId}.mp3`;

        const audio = new Audio(audioUrl);
        this.currentAudio = audio;
        this.ttsState = 'playing';

        audio.onended = () => {
          this.ttsState = 'idle';
        };

        audio.onerror = () => {
          this.ttsState = 'idle';
        };

        audio.play().catch((err) => {
          console.error('Failed to play audio:', err);
          this.ttsState = 'idle';
        });
      },
      error: (err) => {
        console.error('TTS synthesis failed:', err);
        this.ttsState = 'idle';
      }
    });
  }

  resetConversion(): void {
    this.conversionState     = 'idle';
    this.conversionResult    = null;
    this.conversionError     = '';
    this.selectedTargetDialect = '';
    this.ttsState            = 'idle';
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    if (this.currentAudioUrl) {
      URL.revokeObjectURL(this.currentAudioUrl);
      this.currentAudioUrl = null;
    }
    this.ttsDownloadUrl = null;
    this.ttsDownloadName = 'arabic-tts.mp3';
  }

  get canConvert(): boolean {
    return !!this.selectedTargetDialect &&
           !!this.transcribedText.trim() &&
           this.conversionState !== 'loading';
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  formatDuration(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }
}

type FeaturePlotTab = 'rms' | 'zcr' | 'spectral_contrast' | 'chroma';
