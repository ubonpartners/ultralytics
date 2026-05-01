# UBON QAT — Quantization-Aware Training

QAT (Quantization-Aware Training) is wired into the UBON ultralytics fork as a
targeted backport of the upstream `qat-nvidia` design, adapted to preserve
UBON's attribute head, ReID adapter, end2end NMS path, and engine-metadata
changes.

The pipeline:

1. Take an existing `yolo26 + attr (+ reid)` FP32 checkpoint.
2. Insert NVIDIA Model Optimizer Q/DQ wrappers, calibrate on real data.
3. Fine-tune for a few epochs (`int8=True` in `model.train`).
4. Round-trip the Q/DQ state through a `.pt` checkpoint.
5. Optionally re-fuse a ReID adapter on top of the Q/DQ checkpoint.
6. Export ONNX with explicit Q/DQ nodes; build a TensorRT INT8 engine
   without any PTQ calibration step.

---

## How to use

### 1. Install requirements

QAT depends on NVIDIA Model Optimizer; it is loaded lazily and asserts the
runtime supports it (Torch ≥ 2.6, Python ≥ 3.10).

```bash
pip install nvidia-modelopt
```

`setup_modelopt()` (in `ultralytics/utils/torch_utils.py`) will auto-install
via `check_requirements` on first use, so manual install is optional.

### 2. Train with QAT

Pass `int8=True` to a normal `model.train(...)` call. The trainer detects this
and inserts Q/DQ wrappers (calibrating on `qat_calib_data` or the train split)
before the training loop starts.

```python
from ultralytics import YOLO

model = YOLO("yolo26l-attr.pt")            # your existing FP32/FP16 checkpoint
model.train(
    data="coco.yaml",
    epochs=5,                              # short fine-tune is normal for QAT
    imgsz=640,
    batch=8,
    int8=True,                             # ← enables QAT
    lr0=1e-4,                              # QAT typically wants a lower LR
    val_at_start=True,                     # baseline metrics before any step
    qat_calib_method="max",                # max | percentile | mse | entropy
    qat_calib_samples=4096,                # per rank
    qat_exclude=["model.23"],              # optional: skip head module Q/DQ
)
```

What happens in `_setup_train()`:

1. `setup_model()` builds/loads the model normally.
2. If `int8=True` and the model has no `_modelopt_state` yet,
   `build_quantized_model()` runs.
   - Calls `setup_modelopt()` (asserts Torch/Python versions, suppresses noisy logs).
   - Builds a calibration dataloader from `qat_calib_data` (default: `data["train"]`)
     using `mode="val"` so no augmentation runs and rect-batching is consistent.
   - Calls `mtq.quantize(...)` with INT8 per-channel weights, per-tensor activations,
     DFL excluded.
   - For `max`: modelopt's built-in `forward_loop` handles calibration end-to-end.
   - For `percentile`/`mse`/`entropy`: histogram calibrators are inserted, our
     forward_loop runs, then `load_calib_amax(method, percentile=...)` is called
     per quantizer.
   - DDP: each rank calibrates on its own slice; cross-rank `all_reduce(MAX)`
     restores the global amax (mathematically equivalent to single-process max
     over the union of inputs).
   - Quantizers that saw no data (NaN / negative amax) are disabled to avoid
     downstream CUDA assert failures.
3. Mosaic / mixup / cutmix / copy_paste / close_mosaic are forced off — fake-quant
   is sensitive to distribution shift, and these augmentations defeat that.
4. The E2E loss schedule (`o2m`/`o2o`) is pinned at the final ratio
   (`o2m = 0.1`) so QAT does not re-run the warmup that the source checkpoint
   already finished.
5. Each `save_model()` writes a QAT-aware checkpoint (see layout below) instead
   of pickling the live module (TensorQuantizer wrappers are not picklable).

### 3. Export to TensorRT

Re-load the QAT `best.pt` and export as usual:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # QAT checkpoint
model.export(format="engine", int8=True, imgsz=640) # → best.engine (INT8)
```

Or, equivalently, ONNX first then `trtexec`:

```bash
yolo export model=best.pt format=onnx int8=True simplify=False
trtexec --onnx=best.onnx --int8 --saveEngine=best.engine
```

`simplify=False` is recommended for the first run — `onnxslim` can fold Q/DQ
nodes in unhelpful ways depending on version.

`half=True` is automatically disabled on QAT models everywhere it would cause
silent corruption (PyTorch backend inference, training-mode validation).
TensorRT engine builds run with `BuilderFlag.INT8` only; FP16 is not stacked
on top because Q/DQ scales are FP32 by spec.

### 4. ReID fusion on a QAT checkpoint

ReID training only updates a small adapter head; the backbone is frozen. To
fold a trained adapter onto a QAT checkpoint, the REID fusion script
(`reid/src/reid_model.py`) detects `modelopt_state` in the source `.pt` and
takes a Q/DQ-preserving re-save path:

* The fused model's full `state_dict` (now containing `head.reid.*` weights)
  replaces the saved one.
* A `reid_promotion` descriptor `{head_cls_name, adapter_cls_name, in_dim, emb}`
  is added so the QAT loader can class-swap the head and graft the adapter
  *before* `load_state_dict`.
* The original `modelopt_state` is preserved verbatim so TensorQuantizer
  wrappers stay valid.

---

## New CLI / YAML keys

Defined in `ultralytics/cfg/default.yaml`:

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `val_at_start` | bool | `False` | Run validation on initial weights before any training step (epoch 0). Useful baseline for fine-tune / QAT. Skipped on resume. |
| `qat_freeze_bn` | bool | `False` | Keep BatchNorm running statistics frozen during QAT fine-tuning. Recommended when the source checkpoint is fully converged. |
| `qat_exclude` | list[str] | _unset_ | Glob-substring patterns matching module name fragments to skip Q/DQ insertion (e.g. `["model.23"]` to keep the detection head FP). |
| `qat_calib_samples` | int | `4096` | Per-rank calibration sample cap (total = `qat_calib_samples * world_size`). |
| `qat_calib_data` | str | _unset_ | Override calibration data source. Defaults to the training data with val-mode preprocessing. Use to point at a dedicated calibration image dir. |
| `qat_calib_method` | str | `max` | Activation calibration method: `max` \| `percentile` \| `mse` \| `entropy`. |
| `qat_calib_percentile` | float | `99.9` | Percentile clip for activation amax (used when `qat_calib_method=percentile`). |

Existing keys reused:

| Key | Use |
| --- | --- |
| `int8` | Set to `True` to enable QAT (when used with `train`) or INT8 export (when used with `export`). |
| `data` | Same as normal training. The trainer reads `data["train"]` for calibration unless `qat_calib_data` overrides it. |
| `train_subsample` | New `data:` dict key (not a CLI key). When `train_subsample > 1` the dataloader keeps every Nth image. Stride-based sampling is unbiased across multi-source train lists, unlike `fraction` which is a deterministic prefix slice. |

---

## Files modified

| File | Change |
| --- | --- |
| `ultralytics/cfg/default.yaml` | Add `val_at_start`, `qat_freeze_bn`, `qat_exclude`, `qat_calib_samples`, `qat_calib_data`, `qat_calib_method`, `qat_calib_percentile`. |
| `ultralytics/utils/torch_utils.py` | Add `TORCH_2_6` constant. Add `setup_modelopt()` (Torch/Python asserts + log suppression + lazy install). Make `strip_optimizer` skip `.half()` and `requires_grad=False` when the checkpoint dict has a `state_dict` key (QAT layout). |
| `ultralytics/engine/trainer.py` | Add `build_quantized_model()`. Hook into `_setup_train()` to insert Q/DQ + disable mosaic family augmentations + disable `close_mosaic` when `int8=True`. Skip the E2E `criterion.update()` call for QAT. Add optional epoch-0 `val_at_start` validation. Add `qat_freeze_bn` to BatchNorm freeze logic. Modify `save_model()` to detect QAT (`hasattr(model, "_modelopt_state")`) and persist `modelopt_state`, `state_dict`, `model_class`, `yaml`, `names`, `nc`, plus UBON head metadata (`attr_nc`, `attr_names`, `attr_ncs`, `kpt_shape`, `task`); EMA payload is set to `None` for QAT. |
| `ultralytics/engine/validator.py` | Lazy-init `self.loss` in training-mode val so `trainer.loss_items` doesn't have to exist yet (needed for `val_at_start`). Force FP32 when validating a QAT model with `args.half=True`. |
| `ultralytics/engine/exporter.py` | Detect QAT via `model._modelopt_state[0][0] == "quantize"`. ONNX export moves model + dummy input to CPU when QAT (modelopt's tracing path crashes without a CUDA toolkit). TRT engine path passes `dataset=None` when QAT, skipping PTQ calibration. |
| `ultralytics/utils/export/engine.py` | Only assign `int8_calibrator` when a calibration dataset is provided, so explicit-Q/DQ ONNX flows through TensorRT untouched. |
| `ultralytics/utils/loss.py` | `E2ELoss.__init__`: pin `o2m = final_o2m` when `int8=True` so QAT skips the o2m→final_o2m warmup the source checkpoint already finished. |
| `ultralytics/nn/tasks.py` | `BaseModel.fuse()` early-out when `_modelopt_state` is present. `load_checkpoint()` rebuilds the bare module from `model_class(yaml)` + UBON head metadata, then `mto.restore_from_modelopt_state` + (optional `reid_promotion` graft) + `load_state_dict` under `torch.no_grad()`. |
| `ultralytics/nn/backends/pytorch.py` | Force FP32 on QAT models in `PyTorchBackend.load_model()` even if the caller asked for FP16. |
| `ultralytics/data/build.py`, `ultralytics/data/dataset.py` | New `train_subsample` data-yaml key for stride-based sampling on multi-source train lists. Cache file name includes the stride to avoid label-cache thrashing. |
| `ultralytics/models/yolo/detect/train.py` | Apply `args.end2end` to the model when set (needed so QAT calibration sees a consistent end2end branch on every rank). |

No changes to `nn/modules/head.py` — UBON's attr/reid heads work with Q/DQ
insertion since they are standard `nn.Conv2d` / `nn.Linear` chains.

---

## Checkpoint layout (QAT)

```
{
  "epoch": int,
  "best_fitness": float,
  "model": None,
  "ema": None,                           # ← QAT: ema slot is empty
  "updates": int,
  "optimizer": dict,
  "scaler": dict,
  "train_args": dict,
  "train_metrics": dict,
  ...

  # QAT extras (only present when `_modelopt_state` was on the EMA model):
  "modelopt_state": list,                # mto.modelopt_state(model)
  "state_dict": dict[str, Tensor],       # FP32 weights including quantizer scales
  "model_class": type,                   # e.g. ultralytics.nn.tasks.DetectionModel
  "yaml": dict,                          # cfg used to rebuild the module
  "names": dict,
  "nc": int,
  "attr_nc": int,                        # UBON-only, when present
  "attr_names": list,                    # UBON-only, when present
  "attr_ncs": list,                      # UBON-only, when present
  "kpt_shape": tuple,                    # UBON-only, when present
  "task": str,                           # UBON-only, when present
  "reid_promotion": dict,                # Only set by reid/src/reid_model.py after QAT REID fusion
}
```

Standard (non-QAT) checkpoints are byte-for-byte unchanged.

---

## QAT config (defaults)

Inserted by `build_quantized_model()`:

```python
{
  "quant_cfg": {
    "*weight_quantizer": {"num_bits": 8, "axis": 0, "trt_high_precision_dtype": "Float"},
    "*input_quantizer":  {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Float"},
    "*output_quantizer": {"enable": False, "trt_high_precision_dtype": "Float"},
    "*.dfl*weight_quantizer": {"enable": False},
    "*.dfl*input_quantizer":  {"enable": False},
    "*.dfl*output_quantizer": {"enable": False},
    # plus user `qat_exclude` patterns expanded over all three quantizer kinds
    # plus calibrator="histogram" on *input_quantizer when method != max
  },
  "algorithm": "max" | None,             # None → manual two-step calibration
}
```

`trt_high_precision_dtype = "Float"` keeps the high-precision side of every
Q/DQ pair as FP32 in the exported ONNX. TensorRT then chooses INT8/FP32 per
layer at engine-build time.

---

## DDP behaviour

* Calibration runs per-rank on a disjoint slice of the calibration data
  (modelopt's `mtq.quantize` is invoked from each rank).
* After local calibration, every TensorQuantizer's `amax` is reduced across
  ranks with `all_reduce(MAX)`. `max(max(A), max(B), …) == max(A ∪ B)`
  exactly, so this is equivalent to running calibration on the union of
  per-rank slices.
* Conditional branches that some ranks never exercise (e.g. the `one2one_*`
  heads when `end2end=False`) leave their amax uninitialised. Those
  quantizers are detected (NaN / negative amax) and disabled.
* Total calibration sample budget = `qat_calib_samples * world_size`. With
  the default 4096 and an 8-GPU run, 32 768 calibration images are seen,
  matching the previous PTQ recipe in `quant/make_int8.py`.

---

## Known caveats

1. **`half=True` + QAT.** Auto-disabled in the PyTorch backend, in
   training-mode validation, and at export. Don't pass it explicitly.
2. **`onnxslim` + Q/DQ.** Some onnxslim versions reorder Q/DQ pairs into
   shapes TensorRT no longer recognises. Run with `simplify=False` first; if
   the engine builds, re-enable simplification.
3. **Dynamic shapes + NMS + QAT.** Triple combination is not yet validated.
   Start static, then add axes one at a time.
4. **EMA disabled for QAT.** `ema_payload=None` in `save_model()` because the
   live model is what carries the modelopt state. If you depended on EMA for
   final accuracy on FP32 runs, expect QAT runs to track the live weights
   directly.
5. **`strip_optimizer` on a QAT ckpt** keeps the `state_dict` key intact and
   skips `.half()` / `requires_grad=False`, so Q/DQ scales survive the strip.
6. **CPU export move.** ONNX export drops the model to CPU when QAT to avoid a
   modelopt tracing failure on hosts without a CUDA toolkit. This mutates
   `self.model` / `self.im` — fine because the exporter tears down after.
7. **Initial fake-quant mAP.** PyTorch fake-quant is numerically lossier than
   a deployed TensorRT INT8 engine (no kernel fusion, no INT32 accumulators).
   The `val_at_start` mAP on a freshly calibrated QAT model can therefore be
   substantially lower than the PTQ-engine baseline; what matters is the
   delta after fine-tuning and the final exported engine's mAP, not the
   epoch-0 fake-quant number.
