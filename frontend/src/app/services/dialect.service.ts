import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { AnalysisResult, TranscriptionResult } from '../models/analysis-result.interface';

@Injectable({
  providedIn: 'root'
})
export class DialectService {
  private readonly apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  analyzeAudio(file: File): Observable<AnalysisResult> {
    const formData = new FormData();
    formData.append('audio', file, file.name);
    return this.http
      .post<AnalysisResult>(`${this.apiUrl}/analyze/`, formData)
      .pipe(catchError(this.handleError));
  }

  analyzeMixedAudio(fileA: File, fileB: File, weightA: number, mixMethod: string): Observable<AnalysisResult> {
    const formData = new FormData();
    formData.append('audio1', fileA, fileA.name);
    formData.append('audio2', fileB, fileB.name);
    formData.append('weight', weightA.toString());
    formData.append('mix_method', mixMethod);
    return this.http
      .post<AnalysisResult>(`${this.apiUrl}/analyze-mix/`, formData)
      .pipe(catchError(this.handleError));
  }

  transcribeAudio(file: File): Observable<TranscriptionResult> {
    const formData = new FormData();
    formData.append('audio', file, file.name);
    return this.http
      .post<TranscriptionResult>(`${this.apiUrl}/transcribe/`, formData)
      .pipe(catchError(this.handleError));
  }

  healthCheck(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health/`).pipe(catchError(this.handleError));
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let message = 'An unexpected error occurred.';
    if (error.error?.error) {
      message = error.error.error;
    } else if (error.status === 0) {
      message = 'Cannot connect to the backend. Is Django running on port 8000?';
    } else if (error.status === 503) {
      message = 'ML model not loaded. Check that dialect_model.pkl exists.';
    }
    return throwError(() => new Error(message));
  }
}
