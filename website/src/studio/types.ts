/**
 * Types mirroring inference/prediction_schema.py (the backend's response contract).
 * Every value rendered by the Studio comes from a response of these shapes; nothing is
 * computed or invented client-side except formatting.
 */

export type Dataset = "cremad" | "utd_mhad";
export type Status = "compatible" | "potential_distribution_shift" | "low_quality" | "unsupported";
export type Severity = "unsupported" | "low_quality" | "info";

export interface ClassProb {
  index: number;
  code: string;
  name: string | null;
  probability: number;
}

export interface Prediction {
  source: "joint_head" | "frozen_encoder_probe";
  modalities_observed: string[];
  description: string;
  predicted_index: number;
  predicted_code: string;
  predicted_name: string | null;
  probabilities: ClassProb[];
  max_probability: number;
  entropy_nats: number;
  normalized_entropy: number;
  reference_accuracy: { validation_accuracy: number; test_accuracy_evaluation_only: number; chance: number };
}

export interface Issue {
  code: string;
  severity: Severity;
  modality: string;
  message: string;
  measured_false_flag_rate_test: number | null;
}

export interface ShiftAssessment {
  detector: string;
  score: number;
  threshold: number;
  exceeds: boolean;
  ratio_to_threshold: number;
  validation_flag_rate: number;
  test_flag_rate_evaluation_only: number;
  used_for_status: boolean;
}

export interface TraceLayer {
  name: string;
  type: string;
  output_shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface Step {
  name: string;
  duration_ms: number;
  detail: string;
}

export interface PredictionResponse {
  request_id: string;
  dataset: Dataset;
  mode: string;
  status: Status;
  status_reasons: string[];
  primary: Prediction | null;
  unimodal: Record<string, Prediction>;
  issues: Issue[];
  shift: ShiftAssessment[];
  input_report: Record<string, Record<string, unknown>>;
  preprocessed: {
    audio_logmel?: string;
    audio_tensor_shape?: number[];
    audio_logmel_range?: [number, number];
    video_frames?: string[];
    video_tensor_shape?: number[];
    depth_frames?: string[];
    depth_tensor_shape?: number[];
    inertial_zscored?: number[][];
    inertial_tensor_shape?: number[];
    inertial_channels?: string[];
  };
  trace: {
    executed: boolean;
    layers: TraceLayer[];
    representations: Record<string, number[]>;
    notes: string[];
  };
  steps: Step[];
  total_ms: number;
  model: Record<string, any>;
  limitations: string[];
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  device: string;
  models: Record<Dataset, { loaded: boolean; checkpoint_id?: string; error?: string }>;
  uptime_s: number;
}

export interface StudioErrorBody {
  code: string;
  message: string;
  field: string | null;
}

/** POST /preview/video: the 8 frames the model would sample, decoded on the server. */
export interface VideoPreviewResponse {
  request_id: string;
  n_frames: number;
  fps: number;
  width: number;
  height: number;
  duration_s: number;
  crop: { x: number; y: number; w: number; h: number } | null;
  frames: { index: number; time_s: number | null; image: string }[];
  audio_stream: { codec: string; sample_rate: number; duration_s: number | null } | null;
}
