# YOLO (You Only Look Once)

YOLO is a **family of one-stage object detection architectures** for images and videos. Given an input, it predicts bounding boxes, class labels and confidence scores in a **single forward pass**.

```
Image → Backbone → Neck → Head → Boxes + Classes
```

## 1. One-Stage vs Two-Stage

| | Two-stage (R-CNN family) | One-stage (YOLO) |
|---|---|---|
| Pipeline | 1. propose regions, 2. classify each | Predict everything directly |
| Speed | Slower | Real-time friendly |
| Context | Looks at crops | Sees the whole image at once |

The original idea (Redmon et al., 2016): split the image into an S×S grid, and each cell predicts the objects whose center falls inside it. Today the "grid" is the feature maps themselves (P3–P5): **every position predicts what is there**.

## 2. Architecture (YOLOv8 as reference)

### 2.1 Backbone

CSPDarknet: a stack of `Conv` (stride 2) and **C2f** blocks, ending in **SPPF**. It outputs feature maps at three scales:

| Level | Stride | Size (640 input) | Role |
|---|---|---|---|
| C3 | 8 | 80×80 | small objects, fine detail |
| C4 | 16 | 40×40 | medium objects |
| C5 | 32 | 20×20 | large objects, strong semantics |

**C2f** (CSP bottleneck, 2 convs, faster): 1×1 conv → split channels in two halves → one half goes through N sequential bottlenecks → **concat of the untouched half plus every bottleneck output** → 1×1 conv. Benefits: cheap (only half the channels are processed), many short gradient paths, and features from different depths are mixed.

**SPPF**: stacked 5×5 max-pools (applied sequentially) concatenated to enlarge the receptive field cheaply.

### 2.2 Neck

Fuses the backbone's multi-scale maps so each level has both **semantics** ("what") and **localization** ("where").

**FPN (top-down):** semantics flow from deep to shallow levels.

```
M5 = lat(C5)
M4 = lat(C4) + up(M5)
M3 = lat(C3) + up(M4)
Pi = conv3x3(Mi)
```

- `lat` (1×1 conv) aligns **channels**; `up` (nearest) aligns **H×W**.
- Limitation: information only flows down, so P5 only knows C5.

**PAN (adds bottom-up):** fine detail flows from shallow to deep levels.

```
N3 = P3
N4 = down(N3) + P4
N5 = down(N4) + P5       # down = 3×3 conv, stride 2
```

**YOLO's version (v4+: CSP-PAN):** same top-down + bottom-up structure, but **concatenation + C2f** replaces addition, there are no lateral 1×1 convs, channels differ per level, and only P3–P5 are used.

```
Top-down:   up(C5) → concat(C4) → C2f → T4
            up(T4) → concat(C3) → C2f → P3_out
Bottom-up:  conv_s2(P3_out) → concat(T4) → C2f → P4_out
            conv_s2(P4_out) → concat(C5) → C2f → P5_out
```

| | FPN | YOLO (v5/v8) |
|---|---|---|
| Fusion | add | concat + conv block |
| Lateral 1×1 | yes | no |
| Channels | same N everywhere | different per level |
| Direction | top-down | top-down + bottom-up |
| Levels | P2–P6 | P3–P5 |

### 2.3 Head

**Decoupled** (separate branches for box and class) and **anchor-free**, one head per level. Each position predicts:

| Output | Shape | Meaning |
|---|---|---|
| Box | H×W×(4·reg_max) | distance to the 4 box edges as a distribution (DFL, reg_max=16) |
| Class | H×W×K | class scores (K = number of classes) |

No objectness score in v8 (it existed in v5 and earlier). For a 640×640 input: 80² + 40² + 20² = **8400 predictions**.

Post-processing: decode boxes → score threshold → NMS (YOLOv10 removes NMS).

## 3. Training

- **Assigner:** TAL (Task-Aligned Assigner) decides which predictions are responsible for each ground truth, based on a mix of classification score and IoU.
- **Losses:** BCE (class) + CIoU (box) + DFL (box distribution).
- **Augmentations:** mosaic, mixup, HSV, flip, scale.
- **Extras:** EMA of weights, AMP, warmup.

## 4. Evolution

| Version | Key change |
|---|---|
| v1 | S×S grid, direct regression |
| v2 | Anchor boxes, batch norm |
| v3 | Multi-scale predictions (FPN-like), Darknet-53 |
| v4 | CSPDarknet53, SPP, PANet |
| v5 | PyTorch, C3 blocks, SPPF, CSP-PAN |
| v8 | C2f, anchor-free decoupled head, TAL |
| v9 | PGI + GELAN |
| v10 | NMS-free training |
| v11 | C3k2 blocks, C2PSA (spatial attention) |

## 5. The Ultralytics Library

`ultralytics` ships the full pipeline (architecture, training, inference, export), so you don't implement the modules, FPN/PAN, assigner, losses, augmentations or NMS yourself.

```bash
pip install ultralytics
```


FOR MORE INFORMATIONS ABOUT YOLO LIB YOU CAN SEE: https://github.com/tevoshw/yolo