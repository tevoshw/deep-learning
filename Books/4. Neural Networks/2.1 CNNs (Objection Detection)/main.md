# Object Detection

Locating objects in images and drawing a bounding box around each one, with a class label and confidence score.

Image → Backbone → Neck → Head → Boxes + Classes


## 1. Backbone

Stack of convolutional blocks that extracts features at multiple scales (C2–C5).

| Level | Stride | Resolution | Semantics | Detail |
|---|---|---|---|---|
| C2 | 4 | high | weak | high |
| C5 | 32 | low | strong | low |

## 2. Neck

Fuses the backbone's multi-scale feature maps so every level gets both **semantics** ("what") and **localization** ("where").

### Feature Pyramid Network (FPN)

Top-down path (C5 → C2):

1. **Lateral:** 1×1 conv aligns channels of every Ci to N (e.g. 256).
2. **Upsample + add:** upsample 2× the level above (H×W only), then add element-wise.
3. **Smooth:** 3×3 conv on each M to produce the output P.
4. Send P2–P5 to the head.

M5 = lat(C5)
M4 = lat(C4) + up(M5) # C4 + C5
M3 = lat(C3) + up(M4) # C3 + C4 + C5
M2 = lat(C2) + up(M3) # C2 + C3 + C4 + C5

Pi = conv3x3(Mi)


- `lat` changes **channels**, `up` changes **H×W**.
- The chain propagates **M**, not P.
- Limitation: information only flows down, so P5 only knows C5.

### Path Aggregation Network (PAN)

FPN + a bottom-up path (P2 → P5):

N2 = P2
N3 = down(N2) + P3
N4 = down(N3) + P4
N5 = down(N4) + P5


- `down` = 3×3 conv with stride 2.
- Now every level has both semantics and fine detail.
- **YOLO (v4+):** same idea, but uses **concat + C2f/CSP block** instead of add, and only P3–P5.

| | Direction | Carries | Resizes with |
|---|---|---|---|
| FPN | top → down | semantics | upsample |
| PAN | bottom → up | localization | strided conv |

## 3. Head

Convolutions over each neck output. Each position in the feature map predicts:

| Channels | Meaning |
|---|---|
| 4 | box (x, y, w, h) |
| 1 | objectness  |
| K | class probabilities (K = number of classes) |

Post-processing: decode boxes → score threshold → NMS.