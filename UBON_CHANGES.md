# UBON Changes to Ultralytics YOLO

This document describes the additions made to the upstream Ultralytics YOLO26 codebase in
the current `ubon26_wip` branch compared with upstream `main`. The branch is organised around
two feature areas plus follow-up fixes from the current work-in-progress commits:

1. **Binary attribute head** — per-detection multi-label attribute prediction
2. **PoseReID head** — per-detection L2-normalised embedding for re-identification
3. **E2E/ReID follow-ups** — fp16-safe ReID adapter V2 and post-top-k ReID computation

Both features are fully opt-in and backward-compatible: all new config keys default to off,
and models without `attributes: True` / `attr_nc > 0` behave identically to upstream.

---

## 1. Binary Attribute Head

### What it does

A parallel attribute head is added to the Detect (and Pose/Segment/OBB via inheritance)
head that produces a **per-detection sigmoid attribute vector** of length `attr_nc`. Main
class detection is unchanged — NMS is performed on main class scores only; the attribute
vector is then carried through for the selected rows.

Use-cases: gender, age group, clothing colour, occlusion flags, face-visible, carrying
object — any binary or soft labels that apply to a detection but are not the detection class.

### Configuration

Add these keys to your training YAML or command line:

```yaml
attributes: True          # enable attribute head (required)
attr_nc: 6                # number of attribute outputs (required)
attr_label_format: combined  # 'combined' or 'split' (see Dataset below)
attr: 0.25                # BCE loss weight for attribute outputs (default 0.25)
```

Default values in `ultralytics/cfg/default.yaml`:

```yaml
attributes: False
attr_nc: 0
attr_label_format: combined
attr: 0.25
```

`attr_nc` and `attributes` are also registered in `cfg/__init__.py` (INT and BOOL keys
respectively) so they work from the CLI and the Python API.

### Dataset formats

Two label formats are supported:

#### `combined` (default)

Each unique bounding box appears **once per active label**. Multiple rows may share the
same bbox coordinates but have different class IDs. `YOLOAttributeDataset` aggregates rows
with identical bboxes into a single detection with a multi-hot class+attribute vector, then
splits that vector into:

- `cls`: the first active class column (integer class ID)
- `attr`: columns `main_nc … main_nc+attr_nc-1` (float, may be soft 0–1)

Example label file (4 main classes + 3 attribute classes = 7 columns after bbox):
```
# class x_c y_c w h   → combined: two rows for same box, classes 0 and 5
0 0.5 0.3 0.2 0.4
5 0.5 0.3 0.2 0.4
```
`YOLOAttributeDataset` would produce cls=0, attr=[0,1,0] (class 5 → attribute index 1).

#### `split`

Each row has the standard YOLO layout with extra trailing attribute columns:
```
cls x_c y_c w h attr0 attr1 attr2 ...
```
Set `attr_label_format: split` and `attr_nc: N` to use this format.

### Head architecture

**`ultralytics/nn/modules/head.py`**

`Detect.__init__` gains an `attr_nc` parameter (default 0, also declared as a class-level
attribute for backward compatibility with old checkpoints that pre-date `attr_nc`).

When `attr_nc > 0`, **cv3's final 1×1 layer outputs `nc + attr_nc` channels** (rather than
having a separate `cv_attr` ModuleList). `forward_head` splits the trailing `attr_nc` slice
and places it in `preds["attr"]` (logits, `[B, attr_nc, A]`). The `one2many`/`one2one` dicts
contain only `box_head` and `cls_head` — no separate `attr_head` key. Everything else —
`_inference`, E2E `postprocess` top-K gather, row-index column, `bias_init` — is the same as
before.

This layout **matches the `multilabel` branch exactly**: cv3's last 1×1 already outputs
`nc + attr_nc` channels in that branch. The key names and shapes are identical, so weights
can be copied between multilabel and attribute-head checkpoints with zero key remapping.

`Pose` and `Pose26` pass `attr_nc` to `super().__init__`. `Segment` and `OBB` pass
`attr_nc=0` explicitly.

**`ultralytics/nn/tasks.py`**

`parse_model` reads `attr_nc` from the YAML dict and injects it as the third positional
argument of `Pose` / `Pose26` / `PoseReID` heads (position 2, after `kpt_shape`).

### Dataset implementation

**`ultralytics/data/dataset.py`**

`YOLOAttributeDataset` extends `YOLODataset`:
- `cache_labels` handles both `combined` and `split` formats
- `_split_combined_labels(labels, main_nc)`: vectorised numpy split (uses `np.argmax` on
  the multi-hot mask; warns on missing/multiple main classes)
- Cache hash includes `attr_nc` and `attr_label_format` so stale caches are invalidated

`YOLODataset` itself gains `attributes`, `attr_nc`, `attr_label_format` init params and
includes `attr` in the `collate_fn` concatenation set.

**`ultralytics/data/build.py`**

`build_yolo_dataset` selects `YOLOAttributeDataset` when `cfg.attributes` is True; also
falls back to data YAML for attribute settings so a dataset YAML alone is sufficient.

**`ultralytics/data/augment.py`**

All mix/spatial transforms propagate `labels["attr"]` alongside `labels["cls"]`:
- `Mosaic._cat_labels`: concatenates and filters attr with good-bbox mask
- `MixUp`, `CutMix`, `CopyPaste`: concatenate attr when present in both operands
- `RandomPerspective`: filters attr with spatial keep mask
- `Albumentations`: passes `attr_labels` as an extra field to `A.Compose`
- `Format`: converts attr from numpy to torch; handles missing attr gracefully

### Loss

**`ultralytics/utils/loss.py`**

`v8DetectionLoss` gains:
- `attr_enabled = self.attributes and self.attr_nc > 0`
- `assigner_multilabel`: a second `TaskAlignedAssigner` with `num_classes=nc+attr_nc` used
  when `attr_enabled`. The multilabel path calls `_build_multilabel_gt` to produce a
  `[B, max_targets, nc+attr_nc]` multi-hot GT tensor; TAL score gathering uses
  `amax()` over active class channels.
- `get_assigned_targets_and_loss` branches on `attr_enabled`:
  - **multilabel path**: assigner_multilabel → split `target_scores_full` into
    `target_scores_cls` (first `nc` channels) and `target_scores_attr` (last `attr_nc`
    channels); cls BCE uses cls scores; attr BCE uses attr scores; **bbox loss uses
    cls-only scores** for weight/normalization (so box gradient is driven by detection
    quality, independent of attr channel count).
  - **standard path**: existing single-assigner, class-only BCE.
- Attribute BCE loss: `bce(pred_attrs, target_scores_attr).sum() / target_scores_attr.sum().clamp(min=1)`
  scaled by `hyp.attr`
- Loss tensor extended from 3 → 4 slots when `attr_enabled`

`v8PoseLoss`, `PoseLoss26`, and `v8SegmentationLoss` propagate the attribute loss slot
(index 5/6 for pose depending on whether RLE is present, index 3 for detection) and expose
`attr_loss` in `loss_names`.

**E2E uniform scaling** (`E2ELoss.__call__`): All o2m losses — including attr — are scaled
uniformly by `self.o2m`. The o2o path never produces an attr loss (since `include_attr=False`
suppresses attr output), so the attr loss is effectively `attr_o2m * o2m + 0 * o2o`. This
preserves the gradient ratio in cv3's shared intermediate layers between cls and attr, matching
the non-e2e gradient balance. The alternative (keeping attr at weight 1.0 while cls/box/dfl
decay with o2m) causes ~10x gradient distortion late in training, degrading TAL target
assignment quality for attributes.

**`ultralytics/utils/tal.py`**

`TaskAlignedAssigner` extended for multi-label `gt_labels` (last dim > 1):
- Score gathering uses `amax()` over active label channels instead of direct class-index lookup
- The multi-label score path is chunked over GT rows to avoid a Python inner loop while keeping
  temporary memory bounded
- `get_targets` produces a multi-hot `target_scores` tensor instead of one-hot

### Inference and validation

**NMS** (`ultralytics/utils/nms.py`): `return_idxs=True` path in the E2E branch strips the
row-index trailer and returns kept indices for feature alignment.

**Predictors**: detect, pose, segment, OBB predictors:
- Pass `nc=len(model.names)` to NMS (was task-conditional; this is always correct)
- Detect head's `attr_nc` is read via `getattr(head, "attr_nc", 0)` after resolving
  `AutoBackend.model → nn.Sequential[-1]`
- `DetectionPredictor.construct_result`: strips the row-index column (`pred[:, :-1]`) when
  `attr_nc > 0` and `pred.shape[1] >= 6 + attr_nc + 1` (defensive `>=` check)
- `PoseReIDPredictor.construct_result`: uses **exact equality** (`pred.shape[1] == 6 +
  attr_nc + nk + reid_dim + 1`) to avoid accidentally stripping a reid column in non-E2E
  mode where no row-index is appended
- Slice `pred[:, 6 : 6+attr_nc]` and store in `result.attributes`

**Validators**: similar head resolution, strip row-index, pass attr through `"extra"` dict.
`ConfusionMatrix` and `ap_per_class` / `DetMetrics` extended for multi-label `target_cls`
(handles both squeezed-int and multi-hot tensor formats).

**`ultralytics/engine/results.py`**:
- `_keys` includes `"attributes"` (so slicing and iteration work correctly)
- `update(attributes=...)` stores raw sigmoid scores per detection
- `Keypoints.__init__` zeros x,y for keypoints with visibility < 0.01 (display fix)

### Training

**`ultralytics/engine/trainer.py`**:
- Syncs `attributes`, `attr_nc`, `attr_label_format` from data YAML → `self.args` so the
  right dataset class is used even when those keys are only in the data file
- `Pose26.forward_head` includes a zero-scaled dummy `flow_model.log_prob()` call during
  training so that `flow_model` parameters participate in the forward autograd graph. This
  prevents DDP from double-marking flow_model's gradient-ready hooks when `torch.compile`
  is active (previously compile was blanket-disabled under DDP to work around this)
- E2E resume rebuilds the criterion and restores the o2m/o2o loss schedule from the resumed
  epoch, so the decayed one-to-many weight does not restart at the initial value.

**`ultralytics/models/yolo/detect/train.py`**:
- Passes `attr_nc=self.args.attr_nc` when constructing the detection model
- Attaches `model.attr_names` from data YAML (or auto-generates `attr_0 … attr_N`)

**`ultralytics/models/yolo/pose/train.py`**:
- Passes `attr_nc` to `PoseModel`
- Builds correct `loss_names` tuple depending on RLE and attribute flags

### Infrastructure improvements (bundled with this commit)

These are standalone fixes unrelated to attributes but needed in the same codebase:

- **DDP file** (`utils/dist.py`): wraps `main()` with `@record` from
  `torch.distributed.elastic` for better distributed error tracing
- **`model_info()`** (`utils/torch_utils.py`): returns stats dict even when `verbose=False`
  (logging remains conditional)
- **Face keypoint OKS sigmas** (`utils/metrics.py`): `FACEPOSE_SIGMA` (22 kpts, face+pose)
  and `FACEPOSEBOX_SIGMA` (23 kpts, pose+face+face-box corners) for correct OKS computation
  in face+pose validation / loss

---

## 2. PoseReID Head

### What it does

`PoseReID`/`Pose26ReID` extend the pose heads with a **FiLM-modulated MLP**
(`ReIDAdapter` or `ReIDAdapterV2`) that produces a per-detection L2-normalised embedding
vector. The embedding fuses:
- Per-anchor **class logits + attribute logits** from the detection head
- Per-anchor **backbone spatial features** (padded/truncated to a fixed width)
- A **per-scale one-hot code** (FiLM conditioning signal)

This gives the re-identification head access to both visual appearance and detection
semantics (what the person is wearing, whether they are carrying something, etc.).

**Training strategy**: During training `PoseReID.forward()` skips the adapter entirely
(`if self.training: return r`) and trains identically to a standard Pose model. The adapter
weights are then trained *separately* using the companion `reid` repository (triplet loss on
backbone features extracted from the trained YOLO checkpoint), and the trained adapter weights
are **fused** back into the YOLO checkpoint via weight merging. This decoupled approach means:

- The backbone/neck/pose-head are trained with standard YOLO losses
- The adapter is optimised with ReID-appropriate metric learning (triplet loss, hard mining)
- Both can be iterated independently without re-training from scratch

### Architecture

**`ReIDAdapter` and `ReIDAdapterV2`** (`ultralytics/nn/modules/head.py`):

```
Input:  [class_scores (nc + attr_nc) | backbone_feats (FEAT_WIDTH) | scale_code (CODE_LEN)]
FiLM:   scale_code → Linear(CODE_LEN, 2*(nc+FEAT_WIDTH)) → gamma, beta
Feats:  feats = feats * (1 + gamma) + beta
MLP:    Linear(nc+FEAT_WIDTH, hidden1) → ReLU → Dropout(0.05) →
        Linear(hidden1, hidden2) → ReLU → Dropout(0.05) →
        Linear(hidden2, emb_dim) → LayerNorm(emb_dim)
Output v1: F.normalize(mlp_out, p=2) * scale  (learnable temperature)
Output v2: F.normalize(mlp_out, p=2, eps=1e-4)
```

Default sizes: `FEAT_WIDTH=512`, `CODE_LEN=8`, `hidden1=160`, `hidden2=192`. V1 uses
`emb_dim=80`; V2 uses `emb_dim=96`.

`ReIDAdapterV2` is the new preferred adapter for quantized/fp16 export. It keeps the same
input contract as V1, but adds LayerNorm after every Linear, uses SiLU, zero-initialises the
FiLM layer so it starts as identity, removes the learnable output scale, and normalises with
`eps=1e-4` to avoid fp16 underflow. `build_reid_adapter_from_state_dict()` sniffs the state
dict keys and instantiates either V1 or V2, preserving compatibility with existing fused
checkpoints.

The class-score input width is `nc + attr_nc` — both detection class scores and attribute
scores are concatenated before the adapter. This is important when models are converted from
the old multilabel format: a multilabel model with `nc=55` (all classes) and a converted
model with `nc=5, attr_nc=50` both present 55 values to the adapter, so the same adapter
weights work with both checkpoint formats. The FiLM conditioning is applied to
`nc + FEAT_WIDTH` (not `nc + attr_nc + FEAT_WIDTH`) — the scale/shift parameters are sized
to match the backbone feature width, which does not change with attr_nc.

**Architecture constraint**: `FEAT_WIDTH=512` is a fixed model parameter — backbone maps
are zero-padded (if narrower) or channel-truncated (if wider) to exactly 512 before
concatenation. This keeps the `ReIDAdapter` weight shape constant across backbone variants.
For a different backbone width, change `PoseReID.FEAT_WIDTH` and retrain from scratch.

**`_PoseReIDMixin` / PoseReID heads**:
- `forward()` clones the incoming feature maps before delegating to the parent pose head.
  Training mode returns the parent output unchanged, so the adapter is not trained by YOLO
  losses.
- Non-E2E anchor-aligned inference can append ReID embeddings directly to the raw prediction
  tensor.
- E2E inference now computes ReID **after** `postprocess()`/top-k. `_postprocess_with_reid_after_topk`
  runs normal pose postprocessing, recomputes the same top-k indices from the anchor scores,
  gathers only the selected rows' class/attr scores and backbone features, runs the adapter on
  those rows, and appends the embeddings to the final detections.
- Export has a dedicated E2E path that rebuilds one2one inference, injects trained attributes
  from the o2m `cv3` path when needed, and runs the same post-top-k ReID computation. This
  avoids running the adapter across every anchor in exported E2E PoseReID models.

### Model task routing

```
task: posereid
  model:     PoseReIDModel  (tasks.py)
  trainer:   PoseTrainer    (existing)
  validator: PoseValidator  (existing)
  predictor: PoseReIDPredictor (pose/predict.py)
```

`PoseReIDModel` is a thin wrapper around `DetectionModel` that sets `kpt_shape` from the
YAML and uses `v8PoseLoss` (ReID has no separate loss term). In the current workflow this is
mainly for loading/exporting fused checkpoints; normal YOLO pose training still routes through
`PoseTrainer`/`PoseModel`.

`guess_model_task` identifies `PoseReID` heads and returns `"posereid"` so saved
checkpoints load correctly.

### Inference

**`PoseReIDPredictor`** (`ultralytics/models/yolo/pose/predict.py`):
1. Reads `head.reid.emb` to determine embedding width dynamically (no hardcoded size)
2. Strips the row-index trailer **only** when the prediction width exactly equals
   `6 + attr_nc + nk + reid_dim + 1`. Using exact equality (not `>=`) prevents incorrectly
   stripping a reid embedding column in non-E2E inference where no row-index is appended.
3. Slices trailing `emb_dim` channels as `reid_emb`, passes remainder to `PosePredictor`
4. Stores embeddings in `result.reid_embeddings` (shape `[N, emb_dim]`)

**`model.task` in checkpoints**: For `.pt` files, `YOLO._load` reads `model.task` directly
from the stored checkpoint rather than calling `guess_model_task`. A model produced by the
conversion tool must therefore have `model.task = "posereid"` patched onto the saved object,
otherwise `YOLO` loads a `DetectionPredictor` instead of `PoseReIDPredictor` and keypoints
are never extracted. `guess_model_task` does correctly identify PoseReID heads and returns
`"posereid"`, but it is bypassed for `.pt` checkpoints.

**`Results.reid_embeddings`**: registered in `_keys`; slicing/indexing propagates.

### Expanded feature path (get_obj_feats)

`DetectionPredictor.get_obj_feats` gains an `expanded_feats=True` path used by the
companion `reid` repository when extracting training data from a base (non-fused) YOLO model:

```
per anchor: [class_logits + attr_logits | padded_backbone | scale_one_hot]
```

This is the same vector layout that `PoseReID.forward()` constructs at inference time,
ensuring the separately-trained adapter can be fused into the model without any feature
reformatting. The path is activated by the `reid` repo's `on_predict_start` callback:

```python
predictor.expanded_feats = True
predictor._feats = None  # reset; pre-hook overwrites before postprocess
```

When activated, per-detection expanded feature vectors are available in `result.feats`.
During normal PoseReID inference this path is not used — `result.reid_embeddings` is read
directly from the head output.

### Companion repository: `ubonpartners/reid`

The `reid` repository implements the three-phase adapter training workflow:

1. **Feature extraction** — runs a base YOLO model over grid images of person crops; captures
   backbone feature maps via `predictor._feats` (the expanded path described above) and saves
   per-detection `[class_logits | backbone_feats | scale_one_hot]` vectors to a `.npz` file.
2. **Triplet training** — trains `ReIDAdapter` or `ReIDAdapterV2` on the saved features using
   hard triplet mining, margin annealing, and warmup+cosine LR; evaluates Recall@K (FAISS)
   each epoch.
3. **Fusion** — instantiates an empty PoseReID model from `reid_yaml`, copies base YOLO weights
   (exact key+shape match) and adapter weights (suffix-based match), patches task/names/kpt_shape,
   saves the fused `.pt`, and exports ONNX.

See the `reid` repository README for full configuration, dataset loaders, and CLI usage.

---

## Bugs fixed (applied on top of the feature commits)

These bugs were found during review and fixed in `ultralytics/utils/loss.py`:

### 1. GPU-CPU sync in loss normalisation (×4)

`max(tensor.item(), 1)` and `max(tensor, 1)` called `.item()` or compared directly, forcing
a GPU→CPU synchronisation every forward pass and breaking CUDA graph capture.

Fixed by replacing all four occurrences with `.clamp(min=1)`:
```python
# Before
target_scores_sum = max(target_scores.sum(), 1)
# After
target_scores_sum = target_scores.sum().clamp(min=1)
```

### 2. Dead `select_target_attrs` method

A `select_target_attrs(attrs, batch_idx, target_gt_idx, fg_mask)` method was defined but
never called anywhere in the codebase. The actual multilabel path uses the `assigner_multilabel`
path (TAL assigns multi-hot targets directly via `_build_multilabel_gt`). The dead method was
removed.

### 3. `bbox_loss` weight used full `nc+attr_nc` scores in multilabel path

In the multilabel branch, `target_scores` and `target_scores_sum` were computed from
`target_scores_full` (shape `[..., nc+attr_nc]`). `BboxLoss` sums over the last dimension,
so the per-anchor weight included all `attr_nc` channels, coupling box gradient magnitude to
the number of active attributes. Fixed to use `target_scores_cls` (first `nc` channels only)
for bbox loss weight and normalisation — semantically cleaner; numerically equivalent in
expectation since the extra attr channels cancel across a balanced batch.

### 4. Missing `self.class_weights` initialisation

The `v8DetectionLoss.__init__` had `self.class_weights = getattr(model, "class_weights", None)`
removed from a prior refactor, but the usage at lines 500–501 (multilabel path) and 536–537
(standard path) remained, causing `AttributeError` for any model with per-class weights.
Restored the initialisation (and the `.to(device).view(1, 1, -1)` reshape) in `__init__`.

### 5. Pose cls/dfl gain double-scaling

`v8PoseLoss.loss()` and `PoseLoss26.loss()` previously copied `det_loss[1]` and `det_loss[2]`
after `get_assigned_targets_and_loss()` had already applied `hyp.cls`/`hyp.dfl`, then applied
those gains again in the pose wrapper. The current branch keeps cls/dfl single-scaled and
only applies pose-specific gains (`hyp.pose`, `hyp.kobj`, and `hyp.rle`) in the pose loss.

---

## Known limitations / future work

1. **`YOLOAttributeDataset.cache_labels`** duplicates most of `YOLODataset.cache_labels`.
   A future refactor could add a `_process_label(lb, keypoint)` hook to the parent class.

2. **TAL multi-label path** (`utils/tal.py`) still loops over batch rows and chunks GT rows.
   This is much better than the original nested `(bs, n_max_boxes)` loop, but a fully
   vectorised gather/reduce could still help very large batches or dense scenes.

3. **`PoseReID.FEAT_WIDTH=512`** is chosen for the yolo26s backbone. Other backbone widths
   require changing the class constant and retraining.

4. **Adapter trained separately**: the `ReIDAdapter`/`ReIDAdapterV2` is trained with triplet
   loss by the companion `reid` repository (`ubonpartners/reid`), not jointly with the YOLO backbone.
   The adapter participates in inference only (skipped during `model.train()`) and is fused
   into the checkpoint after separate triplet training. Joint end-to-end fine-tuning would
   require a ReID loss term summed with `v8PoseLoss`.

5. **Export**: attributes pass through ONNX/TensorRT export as normal output columns. E2E
   PoseReID export has special handling to compute ReID after top-k so export-time FLOPs stay
   proportional to `max_det` rather than the number of anchors.

6. **Attr bias initialisation**: `Detect.bias_init` initialises cls logit biases (indices
   `0:nc`) but does not explicitly initialise attr biases (indices `nc:nc+attr_nc`). They
   remain at zero (sigmoid → 0.5), which is a reasonable starting point for binary
   attributes but may benefit from a task-specific prior.

7. **E2E attributes are trained only through one-to-many**: this is intentional because the
   one-to-one assignment is unstable early in training, but it means the attribute signal
   decays with the o2m schedule and never receives o2o reinforcement.

---

## Suggestions from current review

1. **Mirror `PoseModel.init_criterion()` in `PoseReIDModel` before enabling direct PoseReID training.**
   `PoseReIDModel.init_criterion()` currently returns plain `v8PoseLoss(self)` regardless of
   `end2end` or whether the head is `Pose26ReID`. That is fine for the current load/export
   workflow where the ReID adapter is trained separately, but direct `task=posereid` training
   would miss `E2ELoss` and `PoseLoss26`/RLE behavior. If direct PoseReID training is intended,
   make it select `PoseLoss26` for `Pose26*` heads and wrap with `E2ELoss` when `end2end=True`.

2. **Watch attribute under-training late in E2E schedules.** Attributes are deliberately
   excluded from one-to-one loss and get `attr_o2m * self.o2m`. With `final_o2m=0.1`, the
   attribute gradient falls to 10% while box/cls/dfl are partly replaced by o2o signal. If
   validation shows attribute AP flattening or regressing late, consider an attribute-specific
   o2m floor or a late attr-only fine-tune while keeping the current default conservative.

3. **Add a small e2e regression test for output layout and gradients.** The critical contract is
   that training `one2one` has no `attr`, inference injects trained o2m attributes, postprocess
   appends a row index only when attrs are present, and PoseReID embeddings sit before that row
   index. A CPU smoke test with a tiny E2E attribute pose model would catch many future shape
   regressions without needing a full training run.

4. **Consider removing the first-batch `.item()` attr-label warning from hot loss code.** The
   one-time check is useful while debugging datasets, but it still introduces a CPU sync on the
   first attribute batch. If compile/cuda-graph cleanliness becomes important, move this check
   to dataset validation or guard it behind a debug flag.

---

## File index

| File | Change category |
|------|----------------|
| `ultralytics/cfg/default.yaml` | Config: attributes, attr_nc, attr_label_format, attr |
| `ultralytics/cfg/__init__.py` | Config: register attr_nc (INT) and attributes (BOOL) |
| `ultralytics/data/dataset.py` | Dataset: YOLOAttributeDataset, cache hash, collate_fn |
| `ultralytics/data/augment.py` | Dataset: propagate attr through all augmentation transforms |
| `ultralytics/data/base.py` | Dataset: filter attr in update_labels |
| `ultralytics/data/build.py` | Dataset: select YOLOAttributeDataset when attributes=True |
| `ultralytics/data/utils.py` | Dataset: verify_image_label accepts attr_len |
| `ultralytics/nn/modules/head.py` | Head: attr_nc in Detect/Pose/Pose26; ReIDAdapter/V2; PoseReID/Pose26ReID; post-top-k E2E ReID |
| `ultralytics/nn/modules/__init__.py` | Head: export PoseReID/Pose26ReID variants |
| `ultralytics/nn/tasks.py` | Model: parse_model attr_nc injection; PoseReIDModel |
| `ultralytics/engine/results.py` | Results: attributes, reid_embeddings; kpt visibility fix |
| `ultralytics/engine/trainer.py` | Train: sync attr args from data YAML; DDP compile fix; resume E2E o2m/o2o schedule |
| `ultralytics/engine/validator.py` | Val: minor attr pass-through |
| `ultralytics/engine/exporter.py` | Export: ReID/attribute compatibility plumbing |
| `ultralytics/engine/model.py` | Model: minor |
| `ultralytics/models/yolo/model.py` | Model: posereid task routing |
| `ultralytics/models/yolo/detect/predict.py` | Predict: attr in construct_result; expanded get_obj_feats |
| `ultralytics/models/yolo/detect/train.py` | Train: attr_nc in model build; attr_names |
| `ultralytics/models/yolo/detect/val.py` | Val: attr_nc row-index strip; nc fix |
| `ultralytics/models/yolo/pose/predict.py` | Predict: attr slice; dynamic PoseReID embedding extraction |
| `ultralytics/models/yolo/pose/train.py` | Train: attr_nc; loss_names |
| `ultralytics/models/yolo/pose/val.py` | Val: attr_nc strip; FACEPOSE_SIGMA; pbatch rows |
| `ultralytics/models/yolo/pose/__init__.py` | Init: export PoseReIDPredictor |
| `ultralytics/models/yolo/segment/predict.py` | Predict: attr minor |
| `ultralytics/models/yolo/segment/train.py` | Train: attr minor |
| `ultralytics/models/yolo/segment/val.py` | Val: attr minor |
| `ultralytics/models/yolo/obb/predict.py` | Predict: attr minor |
| `ultralytics/models/yolo/obb/val.py` | Val: attr minor |
| `ultralytics/models/nas/val.py` | Val: nc fix |
| `ultralytics/utils/loss.py` | Loss: attr BCE; multilabel assigner path; FACEPOSE sigmas import; bug fixes |
| `ultralytics/utils/metrics.py` | Metrics: FACEPOSE/FACEPOSEBOX sigmas; multi-label ConfusionMatrix |
| `ultralytics/utils/nms.py` | NMS: return_idxs E2E row-index path |
| `ultralytics/utils/tal.py` | TAL: chunked multi-label score gathering and target assignment |
| `ultralytics/utils/plotting.py` | Plot: handle 2D cls array from attribute dataset |
| `ultralytics/utils/dist.py` | Infra: @record DDP wrapper |
| `ultralytics/utils/torch_utils.py` | Infra: model_info always returns stats |
| `ultralytics/utils/callbacks/comet.py` | Infra: minor logging update |
| `ultralytics/__init__.py` | Init: `__attributes__ = True` marker; export PoseReID |
