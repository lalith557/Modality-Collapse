"""Request contract: multipart field names, accepted content and limits (documentation + reuse)."""
from __future__ import annotations

from inference.input_validator import AUDIO_KINDS, DEPTH_KINDS, INERTIAL_KINDS, LIMITS, VIDEO_KINDS

ACCEPTED = {
    "audio": {"kinds": sorted(AUDIO_KINDS), "max_bytes": LIMITS["audio"],
              "note": "Speech recording. WAV is read exactly as in training; other containers are decoded to "
                      "16-bit PCM WAV at their native rate first. Only the first 3.0 s are used."},
    "video": {"kinds": sorted(VIDEO_KINDS), "max_bytes": LIMITS["video"], "max_seconds": 30,
              "note": "Close-up, frontal video of one speaking face. The centre square of each frame is used."},
    "depth": {"kinds": sorted(DEPTH_KINDS), "max_bytes": LIMITS["depth"],
              "note": "Kinect-style depth stack in millimetres: UTD .mat (d_depth, H x W x T) or .npy (T x H x W)."},
    "inertial": {"kinds": sorted(INERTIAL_KINDS), "max_bytes": LIMITS["inertial"],
                 "note": "T x 6 signal: acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (deg/s); UTD .mat (d_iner) or CSV."},
}

ENDPOINTS = {
    "POST /predict/cremad/audio": {"fields": {"audio": "required"}},
    "POST /predict/cremad/video": {"fields": {"video": "required"}},
    "POST /predict/cremad/multimodal": {"fields": {"video": "required", "audio": "optional",
                                                   "use_video_audio": "bool, default true: if no audio file is sent, "
                                                                      "use the video's own audio track"}},
    "POST /predict/utd_mhad/multimodal": {"fields": {"depth": "required", "inertial": "required"}},
    "POST /preview/video": {"fields": {"video": "required"},
                            "note": "no model is run: returns the 8 frames the model would sample, for "
                                    "videos the browser cannot play (e.g. .flv)"},
}
