import {
  Component, EventEmitter, Input, Output,
  ElementRef, ViewChild, HostListener
} from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent {
  @Input()  isLoading = false;
  @Input()  hasResult = false;
  @Output() fileSelected = new EventEmitter<File>();
  @Output() analyze     = new EventEmitter<void>();
  @Output() reset       = new EventEmitter<void>();

  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  selectedFile: File | null = null;
  isDragOver = false;
  audioPreviewUrl: string | null = null;
  fileError: string | null = null;

  readonly ACCEPTED = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'];
  readonly MAX_MB   = 50;

  @HostListener('dragover', ['$event'])
  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.isDragOver = true;
  }

  @HostListener('dragleave', ['$event'])
  onDragLeave(e: DragEvent): void {
    this.isDragOver = false;
  }

  @HostListener('drop', ['$event'])
  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.isDragOver = false;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      this.processFile(files[0]);
    }
  }

  onFileInputChange(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.processFile(input.files[0]);
    }
  }

  processFile(file: File): void {
    this.fileError = null;

    // Validate extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!this.ACCEPTED.includes(ext)) {
      this.fileError = `Unsupported format. Please use: ${this.ACCEPTED.join(', ')}`;
      return;
    }

    // Validate size
    if (file.size > this.MAX_MB * 1024 * 1024) {
      this.fileError = `File too large. Maximum size is ${this.MAX_MB}MB.`;
      return;
    }

    this.selectedFile = file;

    // Create preview URL for audio element
    if (this.audioPreviewUrl) {
      URL.revokeObjectURL(this.audioPreviewUrl);
    }
    this.audioPreviewUrl = URL.createObjectURL(file);

    this.fileSelected.emit(file);
  }

  openFilePicker(): void {
    this.fileInput.nativeElement.click();
  }

  onAnalyze(): void {
    if (this.selectedFile && !this.isLoading) {
      this.analyze.emit();
    }
  }

  onReset(): void {
    this.selectedFile = null;
    if (this.audioPreviewUrl) {
      URL.revokeObjectURL(this.audioPreviewUrl);
      this.audioPreviewUrl = null;
    }
    this.fileError = null;
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
    this.reset.emit();
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024)        return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  getFileIcon(name: string): string {
    const ext = name.split('.').pop()?.toLowerCase();
    const icons: Record<string, string> = {
      wav: '🎵', mp3: '🎶', ogg: '🎼', flac: '🎹', m4a: '🎧', aac: '🔊'
    };
    return icons[ext || ''] || '🎵';
  }
}
