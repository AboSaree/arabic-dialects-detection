import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UploadComponent } from './components/upload/upload.component';
import { ResultDashboardComponent } from './components/result-dashboard/result-dashboard.component';
import { TranscriptionComponent } from './components/transcription/transcription.component';
import { DialectService } from './services/dialect.service';
import { AnalysisResult } from './models/analysis-result.interface';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, UploadComponent, ResultDashboardComponent, TranscriptionComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  result: AnalysisResult | null = null;
  isLoading = false;
  error: string | null = null;
  backendOnline = false;
  selectedFile: File | null = null;
  audioUrl: string | null = null;

  constructor(private dialectService: DialectService) {}

  ngOnInit(): void {
    this.checkBackend();
  }

  checkBackend(): void {
    this.dialectService.healthCheck().subscribe({
      next: () => { this.backendOnline = true; },
      error: () => { this.backendOnline = false; }
    });
  }

  onFileSelected(file: File): void {
    this.selectedFile = file;
    this.result = null;
    this.error = null;
  }

  onAudioUrlChanged(url: string | null): void {
    this.audioUrl = url;
  }

  onAnalyze(): void {
    if (!this.selectedFile) return;
    this.isLoading = true;
    this.error = null;
    this.result = null;

    this.dialectService.analyzeAudio(this.selectedFile).subscribe({
      next: (res) => {
        this.result = res;
        this.isLoading = false;
        setTimeout(() => {
          document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      },
      error: (err) => {
        this.error = err.message;
        this.isLoading = false;
      }
    });
  }

  onReset(): void {
    this.result = null;
    this.error = null;
    this.selectedFile = null;
    this.audioUrl = null;
  }
}
