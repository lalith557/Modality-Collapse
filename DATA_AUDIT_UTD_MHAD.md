# UTD-MHAD data integrity audit

_Git `3f66e72afaa654ff6baa4e855fedf80c17cc293f`. Raw dir `../UTD-MHAD`. Modalities: depth + inertial (RGB/skeleton NOT used)._

**STOP flags:** NONE — audit clean

## Trial counts
- depth trials: **861**
- inertial trials: **861**
- **intersection (usable, both modalities): 861** (expected 861)
- depth-only (missing inertial): 0 
- inertial-only (missing depth): 0 
- assigned to a split: 861

## Subjects & split (subject-disjoint)
- subjects present: [1, 2, 3, 4, 5, 6, 7, 8]
- TRAIN subjects [1, 3, 5]: 323 clips
- VAL subjects [7]: 108 clips
- TEST subjects [2, 4, 6, 8]: 430 clips
- overlaps: train∩val=[], train∩test=[], val∩test=[]
- **subject-disjoint: True**

## Per-subject clip counts
- subject 1: 107
- subject 2: 108
- subject 3: 108
- subject 4: 108
- subject 5: 108
- subject 6: 107
- subject 7: 108
- subject 8: 107

## Per-action clip counts
- actions present: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] (n=27)
- min/median/max per action: 31/32/32

## Sample decode (shape check)
- depth_clip: [16, 1, 32, 32]
- depth_dtype: float16
- depth_range: [0.0, 1.0]
- inertial: [6, 128]
- inertial_dtype: float32