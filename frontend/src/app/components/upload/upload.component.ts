import {
  Component, EventEmitter, Input, Output,
  ElementRef, ViewChild
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

export type UploadMode = 'single' | 'mix';
export type MixTarget = 'a' | 'b';

export interface MixPayload {
  fileA: File;
  fileB: File;
  weightA: number;
  weightB: number;
  mixMethod: 'weighted' | 'splice';
}

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent {
  @Input()  isLoading = false;
  @Input()  hasResult = false;
  @Output() fileSelected    = new EventEmitter<File>();
  @Output() audioUrlChanged = new EventEmitter<string | null>();
  @Output() analyze         = new EventEmitter<void>();
  @Output() analyzeMix      = new EventEmitter<MixPayload>();
  @Output() mixedAudioReady = new EventEmitter<string>();
  @Output() reset           = new EventEmitter<void>();

  @ViewChild('fileInputSingle') fileInputSingle!: ElementRef<HTMLInputElement>;
  @ViewChild('fileInputA') fileInputA!: ElementRef<HTMLInputElement>;
  @ViewChild('fileInputB') fileInputB!: ElementRef<HTMLInputElement>;

  mode: UploadMode = 'single';
  selectedFile: File | null = null;
  fileA: File | null = null;
  fileB: File | null = null;
  isDragOver = false;
  audioPreviewUrl: string | null = null;
  audioPreviewUrlA: string | null = null;
  audioPreviewUrlB: string | null = null;
  mixedAudioUrl: string | null = null;
  fileError: string | null = null;
  weightA = 70;
  weightB = 30;
  mixMethod: 'weighted' | 'splice' = 'weighted';

  readonly ACCEPTED = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'];
  readonly MAX_MB   = 50;

  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(e: DragEvent): void {
    this.isDragOver = false;
  }

  setMode(nextMode: UploadMode): void {
    if (this.mode === nextMode) {
      return;
    }
    this.mode = nextMode;
    this.resetSelections();
    this.reset.emit();
  }

  onModeChange(e: Event): void {
    const input = e.target as HTMLSelectElement;
    const nextMode = input.value === 'mix' ? 'mix' : 'single';
    this.setMode(nextMode);
  }

  onDrop(e: DragEvent, target: 'single' | MixTarget = 'single'): void {
    e.preventDefault();
    this.isDragOver = false;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      this.processFile(files[0], target);
    }
  }

  onFileInputChange(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.processFile(input.files[0], 'single');
    }
  }

  onMixFileInputChange(e: Event, target: MixTarget): void {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.processFile(input.files[0], target);
    }
  }

  processFile(file: File, target: 'single' | MixTarget): void {
    this.fileError = null;

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!this.ACCEPTED.includes(ext)) {
      this.fileError = `Unsupported format. Please use: ${this.ACCEPTED.join(', ')}`;
      return;
    }

    if (file.size > this.MAX_MB * 1024 * 1024) {
      this.fileError = `File too large. Maximum size is ${this.MAX_MB}MB.`;
      return;
    }

    if (target === 'single') {
      this.selectedFile = file;
      if (this.audioPreviewUrl) {
        URL.revokeObjectURL(this.audioPreviewUrl);
      }
      this.audioPreviewUrl = URL.createObjectURL(file);
      this.fileSelected.emit(file);
      this.audioUrlChanged.emit(this.audioPreviewUrl);
      return;
    }

    if (target === 'a') {
      this.fileA = file;
      if (this.audioPreviewUrlA) {
        URL.revokeObjectURL(this.audioPreviewUrlA);
      }
      this.audioPreviewUrlA = URL.createObjectURL(file);
      this.clearMixedAudioUrl();
      return;
    }

    this.fileB = file;
    if (this.audioPreviewUrlB) {
      URL.revokeObjectURL(this.audioPreviewUrlB);
    }
    this.audioPreviewUrlB = URL.createObjectURL(file);
    this.clearMixedAudioUrl();
  }

  openFilePicker(target: 'single' | MixTarget = 'single'): void {
    if (target === 'a') {
      this.fileInputA?.nativeElement.click();
      return;
    }
    if (target === 'b') {
      this.fileInputB?.nativeElement.click();
      return;
    }
    this.fileInputSingle?.nativeElement.click();
  }

  async onAnalyze(): Promise<void> {
    if (this.mode === 'mix') {
      if (this.fileA && this.fileB) {
        await this.buildMixedAudioUrl();
        if (this.mixedAudioUrl) {
          this.mixedAudioReady.emit(this.mixedAudioUrl);
        }
        this.analyzeMix.emit({
          fileA: this.fileA,
          fileB: this.fileB,
          weightA: this.weightA,
          weightB: this.weightB,
          mixMethod: this.mixMethod,
        });
      }
      return;
    }
    if (this.selectedFile) {
      this.analyze.emit();
    }
  }

  onReset(): void {
    this.mode = 'single';
    this.resetSelections();
    this.audioUrlChanged.emit(null);
    this.reset.emit();
  }

  onWeightAChange(): void {
    this.weightB = 100 - this.weightA;
    this.clearMixedAudioUrl();
  }

  onWeightBChange(): void {
    this.weightA = 100 - this.weightB;
    this.clearMixedAudioUrl();
  }

  onMixMethodChange(e: Event): void {
    const input = e.target as HTMLSelectElement;
    this.mixMethod = input.value === 'splice' ? 'splice' : 'weighted';
    this.clearMixedAudioUrl();
  }

  swapWeights(): void {
    const tmp = this.weightA;
    this.weightA = this.weightB;
    this.weightB = tmp;

    const tmpFile = this.fileA;
    this.fileA = this.fileB;
    this.fileB = tmpFile;

    const tmpUrl = this.audioPreviewUrlA;
    this.audioPreviewUrlA = this.audioPreviewUrlB;
    this.audioPreviewUrlB = tmpUrl;

    this.clearMixedAudioUrl();
  }

  get canAnalyze(): boolean {
    if (this.mode === 'mix') {
      return this.fileA !== null && this.fileB !== null;
    }
    return this.selectedFile !== null;
  }

  getMixFile(target: MixTarget): File | null {
    return target === 'a' ? this.fileA : this.fileB;
  }

  getMixAudioUrl(target: MixTarget): string | null {
    return target === 'a' ? this.audioPreviewUrlA : this.audioPreviewUrlB;
  }

  getMixTitle(target: MixTarget): string {
    return target === 'a' ? 'Audio File 1' : 'Audio File 2';
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024)        return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  async buildMixedAudioUrl(): Promise<void> {
    if (!this.fileA || !this.fileB) return;

    this.clearMixedAudioUrl();

    const ctx = new AudioContext();

    const [buf1, buf2] = await Promise.all([
      this.fileA.arrayBuffer().then(ab => ctx.decodeAudioData(ab)),
      this.fileB.arrayBuffer().then(ab => ctx.decodeAudioData(ab)),
    ]);

    const sr = ctx.sampleRate;
    const w = this.weightA / 100;

    if (this.mixMethod === 'weighted') {
      const minLen = Math.min(buf1.length, buf2.length);
      const out = ctx.createBuffer(1, minLen, sr);
      const outData = out.getChannelData(0);
      const d1 = buf1.getChannelData(0);
      const d2 = buf2.getChannelData(0);
      for (let i = 0; i < minLen; i++) {
        outData[i] = w * d1[i] + (1 - w) * d2[i];
      }
      this.mixedAudioUrl = await this.audioBufferToUrl(out, ctx);
    } else {
      const minLen = Math.min(buf1.length, buf2.length);
      const splice1 = Math.floor(minLen * w);
      const splice2 = minLen - splice1;
      const out = ctx.createBuffer(1, minLen, sr);
      const outData = out.getChannelData(0);
      const d1 = buf1.getChannelData(0);
      const d2 = buf2.getChannelData(0);
      for (let i = 0; i < splice1; i++) outData[i] = d1[i];
      for (let i = 0; i < splice2; i++) outData[splice1 + i] = d2[i];
      this.mixedAudioUrl = await this.audioBufferToUrl(out, ctx);
    }

    await ctx.close();
  }

  private audioBufferToUrl(buffer: AudioBuffer, ctx: AudioContext): Promise<string> {
    return new Promise((resolve) => {
      const length = buffer.length;
      const wav = new DataView(new ArrayBuffer(44 + length * 2));
      const writeStr = (offset: number, str: string) => {
        for (let i = 0; i < str.length; i++) wav.setUint8(offset + i, str.charCodeAt(i));
      };
      writeStr(0, 'RIFF');
      wav.setUint32(4, 36 + length * 2, true);
      writeStr(8, 'WAVE');
      writeStr(12, 'fmt ');
      wav.setUint32(16, 16, true);
      wav.setUint16(20, 1, true);
      wav.setUint16(22, 1, true);
      wav.setUint32(24, ctx.sampleRate, true);
      wav.setUint32(28, ctx.sampleRate * 2, true);
      wav.setUint16(32, 2, true);
      wav.setUint16(34, 16, true);
      writeStr(36, 'data');
      wav.setUint32(40, length * 2, true);
      const data = buffer.getChannelData(0);
      let offset = 44;
      for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, data[i]));
        wav.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
      const blob = new Blob([wav.buffer], { type: 'audio/wav' });
      resolve(URL.createObjectURL(blob));
    });
  }

  getFileIcon(name: string): string {
    const ext = name.split('.').pop()?.toLowerCase();
    const icons: Record<string, string> = {
      wav: '🎵', mp3: '🎶', ogg: '🎼', flac: '🎹', m4a: '🎧', aac: '🔊'
    };
    return icons[ext || ''] || '🎵';
  }

  private resetSelections(): void {
    this.selectedFile = null;
    this.fileA = null;
    this.fileB = null;
    this.fileError = null;
    this.weightA = 70;
    this.weightB = 30;
    this.mixMethod = 'weighted';
    this.clearMixedAudioUrl();

    if (this.audioPreviewUrl) {
      URL.revokeObjectURL(this.audioPreviewUrl);
      this.audioPreviewUrl = null;
    }
    if (this.audioPreviewUrlA) {
      URL.revokeObjectURL(this.audioPreviewUrlA);
      this.audioPreviewUrlA = null;
    }
    if (this.audioPreviewUrlB) {
      URL.revokeObjectURL(this.audioPreviewUrlB);
      this.audioPreviewUrlB = null;
    }

    if (this.fileInputSingle) {
      this.fileInputSingle.nativeElement.value = '';
    }
    if (this.fileInputA) {
      this.fileInputA.nativeElement.value = '';
    }
    if (this.fileInputB) {
      this.fileInputB.nativeElement.value = '';
    }
  }

  private clearMixedAudioUrl(): void {
    if (this.mixedAudioUrl) {
      URL.revokeObjectURL(this.mixedAudioUrl);
      this.mixedAudioUrl = null;
    }
  }
}
