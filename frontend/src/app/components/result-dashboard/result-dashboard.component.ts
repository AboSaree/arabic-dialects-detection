import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AnalysisResult, DialectConversionResult } from '../../models/analysis-result.interface';
import { DialectService } from '../../services/dialect.service';

@Component({
  selector: 'app-result-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './result-dashboard.component.html',
  styleUrls: ['./result-dashboard.component.css']
})
export class ResultDashboardComponent implements OnInit {
  @Input() result!: AnalysisResult;
  @Input() transcribedText: string = '';   // full_text from TranscriptionComponent
  @Output() reset = new EventEmitter<void>();

  activeTab: 'spectrogram' | 'mfcc' | 'features' | 'waveform' = 'spectrogram';
  probabilityEntries: { dialect: string; value: number; color: string }[] = [];
  animatedConfidence = 0;

  // ── Dialect Converter state ────────────────────────────────────────────────
  readonly availableDialects = ['Egyptian', 'Moroccan', 'Iraqi', 'Lebanese', 'Gulf', 'MSA'];
  selectedTargetDialect = '';
  conversionState: 'idle' | 'loading' | 'done' | 'error' = 'idle';
  conversionResult: DialectConversionResult | null = null;
  conversionError = '';
  ttsState: 'idle' | 'loading' | 'playing' = 'idle';   // TTS placeholder

  private dialectColors: Record<string, string> = {
    'Lebanese':  '#FFFFFF',
    'Moroccan':  '#C1272D',
    'Iraqi':     '#CE1126',
    'Egyptian':  '#000000',
  };

  constructor(private dialectService: DialectService) {}

  ngOnInit(): void {
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
  }

  setTab(tab: 'spectrogram' | 'mfcc' | 'features' | 'waveform'): void {
    this.activeTab = tab;
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

  /** TTS placeholder — wire to your chosen model here */
  pronounce(): void {
    if (!this.conversionResult?.converted_text) return;
    // TODO: Call your TTS endpoint, e.g.:
    //   this.dialectService.synthesize(this.conversionResult.converted_text, this.selectedTargetDialect)
    //     .subscribe(audioBlob => { ... play blob ... });
    this.ttsState = 'loading';
    setTimeout(() => {
      // Simulated response — remove this block once TTS is wired
      alert('TTS not connected yet.\n\nText to pronounce:\n' + this.conversionResult!.converted_text);
      this.ttsState = 'idle';
    }, 500);
  }

  resetConversion(): void {
    this.conversionState     = 'idle';
    this.conversionResult    = null;
    this.conversionError     = '';
    this.selectedTargetDialect = '';
    this.ttsState            = 'idle';
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
