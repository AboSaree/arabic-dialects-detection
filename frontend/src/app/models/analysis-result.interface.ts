// Analysis result model matching Django API response

export interface AudioInfo {
  duration: number;
  sample_rate: number;
  filename: string;
}

export interface FeatureData {
  mfcc_means: number[];
  chroma_means: number[];
  spectral_contrast_means: number[];
  zcr_mean: number;
  rms_mean: number;
}

export interface DialectInfo {
  description: string;
  region: string;
  speakers: string;
  characteristics: string;
}

export interface Plots {
  waveform: string;
  spectrogram: string;
  mfcc: string;
  features: string;
}

export interface AnalysisResult {
  predicted_dialect: string;
  dialect_code: string;
  dialect_flag: string;
  dialect_color: string;
  confidence: number;
  probabilities: { [dialect: string]: number };
  audio_info: AudioInfo;
  feature_data: FeatureData;
  plots: Plots;
  dialect_info: DialectInfo;
}

export interface ApiError {
  error: string;
}

// Transcription models
export interface TranscriptionWord {
  word: string;
  start: number;
  end: number;
}

export interface TranscriptionResult {
  words: TranscriptionWord[];
  full_text: string;
}

// Dialect conversion (Fanar-1-9B-Instruct)
export interface DialectConversionResult {
  converted_text: string;
  source_dialect: string;
  target_dialect: string;
}
