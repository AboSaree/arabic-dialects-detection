import {
  Component, Input, Output, EventEmitter, OnChanges, OnDestroy, SimpleChanges,
  ViewChild, ElementRef, NgZone, ChangeDetectorRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { DialectService } from '../../services/dialect.service';
import { TranscriptionWord } from '../../models/analysis-result.interface';
import { Subscription } from 'rxjs';

type TranscriptionState = 'idle' | 'loading' | 'ready' | 'error';

@Component({
  selector: 'app-transcription',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './transcription.component.html',
  styleUrls: ['./transcription.component.css']
})
export class TranscriptionComponent implements OnChanges, OnDestroy {
  /** The audio File object provided by the upload component */
  @Input() audioFile: File | null = null;
  /** Blob URL for the <audio> element — provided by the upload component */
  @Input() audioUrl: string | null = null;
  /** Emits the full transcribed text once ready — consumed by result-dashboard */
  @Output() transcribed = new EventEmitter<string>();

  @ViewChild('audioRef') audioRef!: ElementRef<HTMLAudioElement>;

  state: TranscriptionState = 'idle';
  errorMsg = '';

  words: TranscriptionWord[] = [];
  fullText = '';
  currentTime = 0;

  private sub: Subscription | null = null;
  private rafId: number | null = null;

  constructor(
    private dialectService: DialectService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    // Kick off transcription as soon as we receive a new file
    if (changes['audioFile'] && this.audioFile) {
      this.startTranscription();
    }
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
    this.stopRaf();
  }

  // ── Transcription ──────────────────────────────────────────────────────────

  private startTranscription(): void {
    this.sub?.unsubscribe();
    this.stopRaf();
    this.state   = 'loading';
    this.words   = [];
    this.fullText = '';
    this.currentTime = 0;

    this.sub = this.dialectService.transcribeAudio(this.audioFile!).subscribe({
      next: (res) => {
        this.words    = res.words;
        this.fullText = res.full_text;
        this.state    = 'ready';
        this.transcribed.emit(res.full_text);   // ← bubble up to app component
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.state    = 'error';
        this.errorMsg = err.message || 'Transcription failed.';
        this.cdr.detectChanges();
      }
    });
  }

  // ── Audio sync ─────────────────────────────────────────────────────────────

  onAudioPlay(): void {
    this.startRaf();
  }

  onAudioPause(): void {
    this.stopRaf();
  }

  onAudioEnded(): void {
    this.stopRaf();
  }

  onAudioTimeUpdate(): void {
    // Fallback for browsers that don't fire rAF smoothly
    if (this.rafId === null) {
      this.syncTime();
    }
  }

  private startRaf(): void {
    this.stopRaf();
    const tick = () => {
      this.syncTime();
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  private stopRaf(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private syncTime(): void {
    const audio = this.audioRef?.nativeElement;
    if (!audio) return;
    // Run outside Angular zone for performance, then trigger CD only when needed
    const t = audio.currentTime;
    if (Math.abs(t - this.currentTime) > 0.05) {
      this.zone.run(() => {
        this.currentTime = t;
      });
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  /**
   * A word is "active" if the audio playhead is inside its [start, end] window.
   * We add a tiny look-ahead (0.15 s) so the highlight fires slightly before
   * the word is fully spoken, which feels more natural.
   */
  isActive(word: TranscriptionWord): boolean {
    return this.currentTime >= word.start - 0.15 &&
           this.currentTime <  word.end;
  }

  /** A word is "spoken" once the audio has passed its end timestamp */
  isSpoken(word: TranscriptionWord): boolean {
    return this.currentTime >= word.end;
  }

  get hasWords(): boolean {
    return this.words.length > 0;
  }
}
