import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AnalysisResult } from '../../models/analysis-result.interface';

@Component({
  selector: 'app-result-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './result-dashboard.component.html',
  styleUrls: ['./result-dashboard.component.css']
})
export class ResultDashboardComponent implements OnInit {
  @Input() result!: AnalysisResult;
  @Output() reset = new EventEmitter<void>();

  activeTab: 'spectrogram' | 'mfcc' | 'features' | 'waveform' = 'spectrogram';
  probabilityEntries: { dialect: string; value: number; color: string }[] = [];
  animatedConfidence = 0;

  private dialectColors: Record<string, string> = {
    'Lebanese':  '#FFFFFF',
    'Moroccan':  '#C1272D',
    'Iraqi':     '#CE1126',
    'Sudanese':  '#00A651',
  };

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

  formatDuration(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }
}
