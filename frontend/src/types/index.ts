// src/types/index.ts — NeuroNova shared TypeScript types

export type AppState = "idle" | "file-selected" | "loading" | "result" | "error";

export interface ClassProbabilities {
  glioma: number;
  meningioma: number;
  notumor: number;
  pituitary: number;
  [key: string]: number;  // allow iteration with Object.entries
}

export interface ModelPrediction {
  prediction: string;
  confidence: number;
  confidence_percent: number;
  probabilities: ClassProbabilities;
}

export interface EnsembleWeights {
  mobilenet: number;
  resnet50: number;
  vgg16: number;
}

export interface DiagnosticMetrics {
  tumor_detected: boolean;
  tumor_probability_percent: number;
  healthy_probability_percent: number;
  certainty_index_percent: number;
  model_agreement_percent: number;
  risk_level: string;
  normalized_entropy: number;
  latency_ms?: number;
  scan_metadata?: {
    original_size?: number[];
    original_mode?: string;
    roi_coverage_percent?: number;
    cropped_roi?: boolean;
    contrast_enhanced?: boolean;
    inverted_background?: boolean;
  };
}

export interface PredictionResult {
  success: true;
  prediction: {
    class: string;
    confidence: number;
    confidence_percent: number;
  };
  probabilities: ClassProbabilities;
  metrics?: DiagnosticMetrics;
  models: {
    mobilenet: ModelPrediction;
    resnet50: ModelPrediction;
    vgg16: ModelPrediction;
  };
  ensemble: {
    weights: EnsembleWeights;
  };
}

export interface ApiError {
  success: false;
  error: string;
}

export type ApiResponse = PredictionResult | ApiError;

export interface FileInfo {
  file: File;
  previewUrl: string;
  dimensions?: { width: number; height: number };
}

export const CLASS_DISPLAY: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
};

export const CLASS_COLORS: Record<string, string> = {
  glioma: "#f43f5e",
  meningioma: "#f59e0b",
  notumor: "#22c55e",
  pituitary: "#818cf8",
};

export const CLASS_ORDER: (keyof ClassProbabilities)[] = [
  "glioma",
  "meningioma",
  "notumor",
  "pituitary",
];
