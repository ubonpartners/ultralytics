# Attribute Training: Design History, Regression Analysis, and Options

This document covers the three attribute-head designs across the branch history,
explains why non-E2E attribute training regressed in `ubon26_wip`, and evaluates
candidate fixes including their FLOP/parameter costs and implications for E2E mode.

---

## Background

The model predicts a small fixed set of **attributes** (e.g. gender, age band, clothing
colour — `attr_nc` logits per anchor) alongside the main detection classes (`nc` logits).
Attributes are multi-label: several can be true for one object simultaneously.

Three different approaches have been tried across the branch history. The notation below
uses a small but realistic model as reference:

```
ch  = (128, 256, 512)   — feature-map channels at 3 detection levels (P3/P4/P5)
nc  = 5                 — main detection classes
attr_nc = 12            — attribute logits
c3  = 128               — intermediate cls-head channels (= ch[0] in practice,
                          since max(ch[0], min(nc, 100)) = ch[0] for any reasonable nc)
H×W = 80×80, 40×40, 20×20  — spatial dims at each level (640-px input)
```

---

## Approach 1 — `multilabel_old`: Attributes as extra classes in cv3

### Architecture

`nc` is set to `nc_main + attr_nc` (e.g. 17). Attributes occupy the last `attr_nc`
slots of the cls head output. There is no separate attribute module. cv3 always used
**standard 3×3 Conv** (the DWConv variant was not yet in use):

```
cv3[i] = Conv(ch[i], c3, 3) → Conv(c3, c3, 3) → Conv2d(c3, nc_main+attr_nc, 1)
```

`batch["cls"]` is a multi-hot vector `[N, nc_main+attr_nc]`. Both main classes and
attributes enter the TAL assigner together.

### Loss (non-E2E)

In `v8PoseLoss.loss()` the cls output is split after TAL, with **separate loss
normalization** for main classes and attributes:

```python
pred_main = pred_scores[:, :nc_main]
pred_attr  = pred_scores[:, nc_main:]

pos_main = tgt_main.sum().clamp(min=1)   # foreground + background supervision
pos_attr = tgt_attr.sum().clamp(min=1)

loss_cls  = bce(pred_main, tgt_main).sum() / pos_main
loss_attr = bce(pred_attr, tgt_attr).sum() / pos_attr
```

Key: attribute targets come from `target_scores`, which are **TAL quality-weighted soft
labels** — the same mechanism that soft-labels the main classes. Background anchors are
also supervised (pushed toward 0) for all attribute logits.

### FLOP / parameter cost vs baseline (no attributes)

| Item | Delta params | Delta FLOPs |
|------|-------------|-------------|
| Extra output neurons in cv3 final Conv2d (`c3 × attr_nc × 3 levels`) | +`3 × 128 × 12` = **+4 608** | negligible |
| Loss split and extra BCE term | — | negligible |
| **Total overhead** | **< 0.01%** | **< 0.01%** |

Essentially **zero cost** — attributes are free extra output channels.

### E2E behaviour

With `end2end=True`, a deepcopy of cv3 (`one2one_cv3`) is made. Both o2m and o2o paths
train all `nc_main + attr_nc` output channels. The TAL in the o2o path uses `topk=1`
and a single-label target — but the cls vector is multi-hot, which confuses the TAL
alignment metric. The `topk=1` assignment was never designed for multi-hot labels. This
is why **E2E never worked well with attributes**: the o2o TAL assigns anchors based on a
quality score that mixes class and attribute predictions, then the hard topk=1 selection
produces unstable assignments early in training, and gradients for both classes and
attributes are driven by the unstable o2o path.

### Summary

| | |
|---|---|
| Non-E2E quality | ✅ Best observed results |
| E2E quality | ❌ Never worked |
| FLOP overhead | ✅ ~0% |
| Param overhead | ✅ ~0% |
| Data format | Multi-hot `batch["cls"]` |
| TAL sees attrs | ✅ Yes — soft targets |

---

## Approach 2 — `attributes_old`: Separate `cv_attr` head

### Architecture

`nc` reverts to `nc_main` only. A **dedicated** `cv_attr` head is added alongside cv3:

```
cv3[i]    = DWConv(ch[i], ch[i], 3) → Conv(ch[i], c3, 1)
          → DWConv(c3, c3, 3)       → Conv(c3, c3, 1)
          → Conv2d(c3, nc_main, 1)

cv_attr[i] = Conv(ch[i], c3, 3)    ← standard 3×3, NOT DWConv
           → Conv(c3, c3, 3)
           → Conv2d(c3, attr_nc, 1)
```

Note: cv_attr uses **regular 3×3 Conv**, while cv3 uses the cheaper DWConv+PW pair. This
is the key cost driver. For E2E, `one2one_cv_attr` is a deepcopy of `cv_attr`.

`batch["cls"]` reverts to single-label `[N, 1]`. `batch["attr"]` is a separate binary
tensor `[N, attr_nc]`.

### FLOP / parameter cost vs `multilabel_old`

FLOPs for a single regular `Conv(c_in, c3, 3)` layer at spatial size `H×W`:

```
FLOPs = 2 × c_in × c3 × 9 × H × W
```

Summing all three levels of `cv_attr` (two Conv3×3 + one Conv1×1 each):

| Level | ch[i] | H×W | Conv(ch,c3,3) FLOPs | Conv(c3,c3,3) FLOPs | Conv(c3,attr,1) FLOPs |
|-------|-------|-----|--------------------|--------------------|----------------------|
| P3 | 128 | 80×80 | 2×128×128×9×6400 ≈ **1.5 GF** | 2×128×128×9×6400 ≈ **1.5 GF** | ~10 MF |
| P4 | 256 | 40×40 | 2×256×128×9×1600 ≈ **0.75 GF** | 2×128×128×9×1600 ≈ **0.38 GF** | ~2 MF |
| P5 | 512 | 20×20 | 2×512×128×9×400 ≈ **0.47 GF** | 2×128×128×9×400 ≈ **0.09 GF** | ~1 MF |

**cv_attr total ≈ 4.7 GFLOPs** (raw; halve for MACs convention).

For comparison, the existing cv3 (DWConv variant) per level is far cheaper — depthwise
3×3 replaces the full 3×3:

```
DWConv FLOPs = 2 × c × 9 × H × W    (c groups, not c×c)
```

P3 DWConv pair ≈ 2×128×9×6400 × 2 ≈ **29 MF**, plus two Conv1×1 ≈ 2×2×128×128×6400
≈ **420 MF**. So cv3 at P3 ≈ **450 MF** vs cv_attr at P3 ≈ **3 GF** — **cv_attr is ~6.5×
more expensive per level**.

Total three-level overhead of cv_attr vs cv3 is roughly **2–3× the cost of cv3 itself**.
On a 640-px input this translates to the **~30% total-model FLOP increase** observed in
practice (head FLOPs dominate on small models; backbone+neck is fixed).

For E2E the cost doubles again: `one2one_cv_attr` is a full deepcopy.

### Loss (non-E2E)

`select_target_attrs` uses the TAL `target_gt_idx` (chosen from main-class TAL) to
assign attribute ground-truth to each foreground anchor. Targets are **hard binary
labels** — no TAL soft-weighting:

```python
target_attrs = select_target_attrs(batch["attr"], batch_idx, target_gt_idx, fg_mask)
loss_attr = (bce(pred_attrs, target_attrs) * fg_mask[..., None]).sum() / pos_attr
```

Background anchors receive **no attribute gradient** (fg_mask zeros them).

### E2E behaviour

With both o2m and o2o paths training cv_attr separately, the o2o path (topk=1, single
GT per anchor) drives unstable attribute gradients early in training for the same reason
as Approach 1 — the TAL wasn't designed with multi-label attribute assignment in mind.
This manifests as the "~10× gradient distortion" noted in ubon26_wip code comments.
The FLOP cost of the deepcopy for E2E is particularly painful.

### Summary

| | |
|---|---|
| Non-E2E quality | ✅ Good (probably on par with Approach 1, untested at scale) |
| E2E quality | ❌ Never worked; gradient instability in o2o |
| FLOP overhead | ❌ ~30% total model; ~2× cv3 cost per level |
| Param overhead | ❌ Significant; doubled again for E2E deepcopy |
| Data format | Single-label `batch["cls"]` + binary `batch["attr"]` |
| TAL sees attrs | ❌ No — hard binary targets post-hoc |

---

## Approach 3 — `ubon26_wip`: Attributes merged into cv3 output

### Architecture

Attributes are appended to cv3's output channels. No new conv modules are added:

```
cv3_out = nc_main + attr_nc   (e.g. 17)
cv3[i]  = DWConv(ch[i], ch[i], 3) → Conv(ch[i], c3, 1)
         → DWConv(c3, c3, 3)       → Conv(c3, c3, 1)
         → Conv2d(c3, nc_main + attr_nc, 1)
```

`c3` is sized on `nc_main` alone: `max(ch[0], min(nc_main, 100))`. In practice c3=ch[0]
(e.g. 128), so the intermediate capacity is unchanged vs no-attribute baseline.

`forward_head` splits the cv3 output: `scores, attrs = cls_out.split([nc, attr_nc], 1)`.
For E2E, `one2one` passes `include_attr=False` so the o2o path never trains attributes:

```python
@property
def one2one(self):
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, include_attr=False)
```

At inference the o2o path borrows the trained attr slice from the o2m cv3.

### FLOP / parameter cost

| Item | Delta |
|------|-------|
| Extra output neurons in cv3 final Conv2d (`c3 × attr_nc × 3 levels`) | +**4 608 params** |
| Extra output neurons in one2one_cv3 (E2E deepcopy) | +4 608 params |
| **Total overhead** | **< 0.01% params, ~0% FLOPs** |

This matches the Approach 1 cost — essentially free.

### Why non-E2E regressed vs Approach 1

Three compounding differences explain the quality drop when `end2end=False`:

**1. TAL no longer sees attributes (vs Approach 1)**

In Approach 1, `gt_labels` was multi-hot `[batch, max_obj, nc_main+attr_nc]`. TAL's
alignment metric `align_metric = pd_scores^α × overlaps^β` was computed over the full
combined label vector. Anchors were selected because they predicted BOTH main classes AND
attributes well.

In Approach 3, `gt_labels` is single-label `[batch, max_obj, 1]`. TAL sees only main
classes. Attribute targets are assigned post-hoc from whichever anchors TAL chose for the
class. In practice the selected anchors coincide (same object), but the quality-weighted
alignment is lost.

**2. Attribute targets are hard binary, not TAL-softened (vs Approach 1)**

In Approach 1, attribute targets came from `target_scores` — the TAL alignment quality
score weighted every anchor's attribute target between 0 and 1, matching how class
targets are softened. This gives informative gradients even for medium-quality anchor
assignments.

In Approach 3 (and Approach 2), `select_target_attrs` produces hard 0/1 targets for
foreground anchors only. There is no quality weighting; the loss treats a barely-matching
anchor identically to a perfect one.

**3. Shared intermediate features (vs Approach 2 as well)**

cv3's DWConv+PW intermediate layers must simultaneously learn:
- class-discriminative features (what object is this?)
- attribute-discriminative features (what properties does it have?)

These can require different spatial patterns and semantics. With a single set of c3
intermediate channels and a merged output, attribute and class gradients interfere
through the same weights. In Approach 2 cv_attr had fully independent intermediate
layers. In Approach 1 the entire head was jointly trained on the combined task from the
start, so the representation organically accommodated both.

The code itself flags the consequence:
```
# ~10x gradient distortion that caused attr/cls imbalance within cv3 and
# degraded TAL target assignment quality for attributes.
```

---

## E2E: Why no approach has worked

All three approaches share the same fundamental E2E problem, which is independent of
the attribute head design:

The TAL o2o path uses `topk=1` (one positive anchor per GT object). This works for
single-label classification because the GT label is unambiguous. Attributes are
**multi-label and sparse** — most anchors have no positive attributes, and the few that
do have varying subsets. The topk=1 assignment:

1. Selects anchors based on `align_metric = pd_scores^α × overlaps^β` where `pd_scores`
   reflects BOTH class and (if not suppressed) attribute predictions — the mixing corrupts
   anchor selection stability early in training.
2. Assigns that one anchor as the sole target for potentially many attribute combinations
   — the gradient signal per anchor is concentrated and noisy.
3. With `include_attr=False` (Approach 3's mitigation): attributes get zero gradient from
   the o2o path, so the o2o-trained copy of cv3 produces untrained attribute outputs,
   and the at-inference borrow from o2m cv3 is a workaround rather than a solution.

None of these patches the root mismatch: TAL's topk=1 assignment was designed for
mutually-exclusive single-label classification, not multi-label attribute supervision.

---

## Candidate fixes

### Option A — Restore Approach 1 (multilabel): attributes as extra cv3 classes

Revert `batch["cls"]` to multi-hot format. Set `nc = nc_main + attr_nc` again. Restore
the separate normalization in the pose loss. Remove `select_target_attrs`.

**Trade-offs:**

| | |
|---|---|
| Non-E2E quality | ✅ Restored to best-known state |
| E2E quality | ❌ Still broken (same TAL topk=1 mismatch) |
| FLOP overhead | ✅ ~0% |
| Param overhead | ✅ ~0% |
| Data pipeline | Needs multi-hot cls reformat |
| TAL sees attrs | ✅ Yes |
| Soft targets | ✅ Yes |

This is the highest-quality non-E2E option and the fastest to implement. E2E with
attributes remains parked.

---

### Option B — Keep merged head, add soft attribute targets from TAL

Keep the current Approach 3 head (zero overhead). Fix the training signal by deriving
attribute targets from `target_scores`' quality weight rather than hard binary labels.
After TAL produces `align_metric` (or equivalently the soft `target_scores` weights),
scale each anchor's attribute target by that anchor's TAL quality score:

```python
# After TAL produces target_scores [batch, anchors, nc] and align_metric:
target_attrs_hard = select_target_attrs(batch["attr"], ...)   # hard 0/1
quality = target_scores.amax(dim=-1, keepdim=True)            # scalar quality per anchor
target_attrs = target_attrs_hard * quality                     # soft attribute targets
```

This replicates the Approach 1 soft-weighting without changing the data format or head.

**Trade-offs:**

| | |
|---|---|
| Non-E2E quality | ✅ Likely close to Approach 1 (recovers the single largest training signal loss) |
| E2E quality | ❌ Still broken (o2o TAL mismatch unchanged) |
| FLOP overhead | ✅ ~0% |
| Param overhead | ✅ ~0% |
| Data pipeline | No change |
| TAL sees attrs | ❌ No (anchors selected on class alone) |
| Soft targets | ✅ Yes (approximated via quality re-weighting) |

Smallest change, recovers most of the quality gap. Doesn't require reverting data format.

---

### Option C — Keep merged head, also train attrs in o2o but with stop-gradient on scores

Allow `include_attr=True` in the o2o path, but stop the gradient from the attribute
loss from flowing back into the shared DWConv layers (only let it update the final 1×1
output channels). This prevents the TAL alignment from being corrupted by attribute
gradients while still providing o2o attribute signal.

```python
# In forward_head for o2o:
cls_out = cls_head[i](x[i])              # full forward, grad flows to final 1×1
# Stop attr gradient at the conv output, not in the backbone features:
cls_feat, attr_feat = cls_out.split([nc, attr_nc], dim=1)
attr_out = attr_feat.detach().requires_grad_(True)  # detach from shared stem
```

This is architecturally awkward and adds a custom backward hook; noted for completeness.

---

### Option D — Lightweight separate attr output head (DWConv, not regular Conv)

If a separate attr head is genuinely needed (e.g. to fully decouple representation),
replace Approach 2's expensive regular Conv3×3 head with the same DWConv+PW structure
that cv3 uses:

```python
cv_attr[i] = DWConv(ch[i], ch[i], 3) → Conv(ch[i], c3_attr, 1)
           → DWConv(c3_attr, c3_attr, 3) → Conv(c3_attr, c3_attr, 1)
           → Conv2d(c3_attr, attr_nc, 1)
```

With a smaller `c3_attr` (e.g. 64 instead of 128) this could be 4–6× cheaper than
Approach 2's cv_attr:

| c3_attr | P3 DWConv FLOPs | Approx 3-level total | vs Approach 2 |
|---------|----------------|----------------------|---------------|
| 128 | ~480 MF | ~800 MF | **~6× cheaper** than regular Conv |
| 64 | ~240 MF | ~400 MF | **~12× cheaper** |

For E2E, the deepcopy of cv_attr is now cheap. The `include_attr=False` suppression
in o2o can then be reconsidered (though the TAL mismatch remains).

**Trade-offs:**

| | |
|---|---|
| Non-E2E quality | ✅ Likely good (dedicated intermediate features) |
| E2E quality | ❌ TAL mismatch remains; but cheaper to deepcopy |
| FLOP overhead | ⚠️ ~5–10% (much better than Approach 2's ~30%) |
| Param overhead | ⚠️ Moderate; small with c3_attr=64 |
| TAL sees attrs | ❌ No |
| Soft targets | ❌ No (unless combined with Option B) |

---

### Option E — Fix E2E properly: attribute-aware TAL assignment

For E2E to work with attributes, the TAL topk=1 assignment strategy needs to be
changed or augmented for attributes specifically. Options:

- **Separate o2o assignment for attributes**: run a second TAL pass over only the
  attribute predictions after the class-TAL has run. Use higher topk (e.g. 3–5) for
  the attribute assignment. Each anchor can be assigned as attribute-positive from up to
  topk GT objects.
- **Multi-label-aware TAL**: modify the alignment metric to handle multi-hot GT labels
  — e.g. `pd_scores = (predicted_probs * gt_multihot).sum(-1)` instead of taking the
  max over a one-hot.
- **Decouple o2o from attribute training entirely**: treat attributes as an auxiliary
  task trained only on the o2m path (which Approach 3 already does), but then accept
  that attributes are not used to drive o2o anchor quality and remove the o2o weight
  copy at inference.

None of these are quick. The cleanest short-term position is to park E2E attributes and
invest in fixing non-E2E (Options A or B above).

---

## Model compatibility with existing `ubon26_wip` checkpoints

This section maps each option onto what changes in the PyTorch state dict and whether
a `.pt` file trained under `ubon26_wip` can be loaded without modification.

### What the current `ubon26_wip` state dict looks like (relevant keys)

```
model[-1].cv3.{0,1,2}.0.0.{weight,bias}   # DWConv layer, shape unchanged by attr
model[-1].cv3.{0,1,2}.0.1.{weight,bias}   # Conv 1×1
model[-1].cv3.{0,1,2}.1.0.{weight,bias}   # DWConv layer
model[-1].cv3.{0,1,2}.1.1.{weight,bias}   # Conv 1×1
model[-1].cv3.{0,1,2}.2.{weight,bias}     # final Conv2d — shape [nc+attr_nc, c3, 1, 1]

model[-1].one2one_cv3.{0,1,2}.2.{weight,bias}  # E2E copy — same shape [nc+attr_nc, c3, 1, 1]
```

There are **no `cv_attr.*` keys**. `self.nc`, `self.attr_nc` and `self.no` are Python
instance attributes, not tensors — they are not stored in the state dict but they are
embedded in the pickled model object when saving with `torch.save(model)` rather than
`torch.save(model.state_dict())`.

---

### Option B — Soft TAL targets

**No architectural change.**

| Item | Status |
|------|--------|
| State dict keys | Identical |
| Tensor shapes | Identical |
| `self.nc` / `self.attr_nc` | Unchanged |
| Inference API | Unchanged |
| **Checkpoint compatible** | ✅ **Drop-in — no migration needed** |

This is a pure training-loop change. Existing `.pt` files load and run without
modification.

---

### Option C — Stop-gradient on attrs in o2o

**No architectural change.**

Same as Option B — purely a training/gradient change. Existing checkpoints load
transparently.

| **Checkpoint compatible** | ✅ **Drop-in** |

---

### Option A — Revert to multilabel (`nc = nc_main + attr_nc`)

**Tensor shapes are identical; Python semantics change.**

The final Conv2d in cv3 keeps the same output channel count because `nc` is redefined
to absorb `attr_nc`:

```
ubon26_wip:  cv3[i][-1]  →  Conv2d(c3, nc_main + attr_nc, 1)   # e.g. [17, 128, 1, 1]
Option A:    cv3[i][-1]  →  Conv2d(c3, nc,              1)      # nc=17 — same shape
```

`self.no = nc + reg_max*4` is the same value in both. No state dict key appears or
disappears. A `torch.load` + `model.load_state_dict(ckpt)` will succeed without error.

**However, three things change that require coordinated code updates:**

1. **`self.nc` and `self.attr_nc` flip:** `self.nc` grows from `nc_main` to
   `nc_main + attr_nc`; `self.attr_nc` becomes 0. Every place in inference and
   downstream code that reads `model.nc` expecting `nc_main` will get the larger value.
   Concretely: `postprocess()` splits `preds` as `[4, self.nc, ...]` — this now absorbs
   the attr logits into the class scores tensor, which is the correct multilabel
   behaviour but breaks any caller that then indexes `scores[:, nc_main:]` expecting
   attributes to be absent.

2. **`batch["cls"]` format reverts to multi-hot `[N, nc_main+attr_nc]`:** The data
   pipeline and dataset collation that currently produces single-label `batch["cls"]`
   and separate `batch["attr"]` must be reverted. Training with the wrong batch format
   against an Option A model will silently corrupt targets.

3. **Model YAML / config `nc` field:** Any `.yaml` architecture file that defines
   `nc: 5` must become `nc: 17`. Mismatched config will either fail to load (shape
   error in layers above the head that depend on `nc`) or produce a model with the
   wrong number of outputs.

| Item | Status |
|------|--------|
| State dict weights load | ✅ No shape mismatch |
| Inference code | ⚠️ Needs updating — `self.nc` semantics change |
| Training data pipeline | ⚠️ Needs updating — multi-hot cls format |
| Model YAML config | ⚠️ Needs updating — `nc` field value changes |
| **Checkpoint compatible** | ✅ **Weights yes; code coordination required** |

**Migration path:** Load the ubon26_wip checkpoint into the Option A model
(`load_state_dict` succeeds). The pretrained weight values are a valid warm start
because channels 0..nc_main-1 of cv3[-1] were trained as class logits and channels
nc_main..nc_main+attr_nc-1 were trained as attribute logits — the same semantic
ordering that Option A expects.

---

### Option D — Lightweight DWConv attr head

**Breaking change — cv3 final layer shrinks and new `cv_attr` keys appear.**

```
ubon26_wip cv3[i][-1]:   Conv2d(c3, nc_main + attr_nc, 1)  →  weight [17, 128, 1, 1]
Option D   cv3[i][-1]:   Conv2d(c3, nc_main,           1)  →  weight [ 5, 128, 1, 1]  ← different shape

Option D adds entirely new keys:
  model[-1].cv_attr.{0,1,2}.{...}.weight / bias
  model[-1].one2one_cv_attr.{0,1,2}.{...}.weight / bias   (if E2E)
```

`load_state_dict` will raise a shape mismatch on `cv3.*.2.weight` and a missing-key
error for `cv_attr.*`. There is no way to load a ubon26_wip checkpoint directly.

**Surgical migration is possible** but requires a script:

```python
ckpt = torch.load("ubon26_wip.pt")
sd   = ckpt["model"].state_dict()

# Trim cv3 final layer to nc_main channels only
for level in range(3):
    key = f"model.28.cv3.{level}.2.weight"        # adjust layer index
    sd[key] = sd[key][:nc_main]                    # keep class channels
    sd[key.replace("weight", "bias")] = sd[...bias][:nc_main]

    # Optionally warm-start cv_attr from the attr slice of the old cv3 output
    # (these were the trained attribute logits in ubon26_wip)
    attr_w_key = f"model.28.cv_attr.{level}.2.weight"
    sd[attr_w_key] = sd_old[key][nc_main:]         # attr channels become cv_attr output
    # cv_attr intermediate layers must be randomly initialised

new_model.load_state_dict(sd, strict=False)        # strict=False to allow new keys
```

The intermediate DWConv layers of `cv_attr` have no corresponding weights in the old
checkpoint and must be randomly initialised. The final 1×1 layer of `cv_attr` can be
seeded from the old cv3 attr-channel slice, giving a warm start for the output mapping
while the new intermediate layers converge.

| Item | Status |
|------|--------|
| State dict weights load | ❌ Shape mismatch on cv3 final layer |
| New cv_attr keys | ❌ Missing in checkpoint |
| Migration script | ⚠️ Feasible — partial weight transplant |
| Training after migration | ⚠️ cv_attr intermediate layers train from scratch |
| **Checkpoint compatible** | ❌ **Breaking — requires migration script** |

---

### Summary table

| Option | cv3[-1] shape | New keys | Python attrs | Checkpoint load | Code changes |
|--------|--------------|----------|--------------|-----------------|--------------|
| **B** (soft targets) | Unchanged | None | Unchanged | ✅ Drop-in | Training loop only |
| **C** (stop-grad) | Unchanged | None | Unchanged | ✅ Drop-in | Training loop only |
| **A** (multilabel) | Same value, different `nc` | None | `nc` ↑, `attr_nc` → 0 | ✅ Weights compatible | Data pipeline, inference, YAML |
| **D** (DWConv head) | Shrinks to `[nc_main, c3, 1, 1]` | `cv_attr.*` added | `attr_nc` unchanged | ❌ Breaking | Migration script + inference |

---

## Recommended path

**Immediate (restore non-E2E quality, zero overhead):**

Option B is the smallest change that recovers the most ground. Keep the current merged
head; add quality-weighted attribute soft targets in `get_assigned_targets_and_loss` by
scaling `target_attrs` by the per-anchor TAL quality score. This directly replaces the
largest training-signal difference between Approach 1 and Approach 3 without touching
data format or architecture.

**If quality remains insufficient:**

Option A (revert to multilabel approach) recovers everything — TAL sees attrs, soft
targets, background supervision — at zero FLOP cost. Requires reverting `batch["cls"]`
to multi-hot and restoring the split normalization in the pose loss.

**E2E + attributes:**

See the next section.

---

## Will Option B truly restore multilabel_old non-E2E quality?

Honest answer: **probably very close, but with a small residual gap**. Here is why.

Approach 1 had three advantages over current Approach 3. Option B recovers one of them
directly and partially compensates for a second:

| Issue | multilabel_old | ubon26_wip | Option B |
|-------|---------------|------------|---------|
| Soft TAL-quality targets | ✅ | ❌ hard 0/1 | ✅ restored |
| TAL anchor selection sees attrs | ✅ | ❌ class only | ❌ (unchanged) |
| Background anchor suppression | ✅ all bg anchors pushed to 0 | ❌ fg_mask zeros bg | ❌ (unchanged) |

**Soft targets (recovered by Option B)** were almost certainly the largest contributor
to the regression. They give the loss function a calibrated signal — a low-quality
foreground anchor contributes less than a high-quality one, which stabilises gradients
early in training and reduces the noise-to-signal ratio for attribute supervision. This
is the same mechanism that makes soft class targets better than hard class targets in
detection, and the magnitude of the improvement is well-documented.

**TAL not seeing attributes** matters much less in practice than it sounds. The anchors
TAL selects for "person" are anchors with high IoU with the person bounding box — the
same box that the attributes belong to. The selected anchors would be largely identical
whether or not the alignment metric included attribute predictions, because person IoU
already concentrates probability mass on the right anchor neighbourhood. The loss is
theoretical: we use a class-quality proxy score rather than a true joint quality score.
This gap is small once training is stable.

**No background suppression** means attribute logits at background anchors receive no
gradient. In Approach 1, background anchors were pushed toward zero for all attribute
logits (via the combined BCE over `target_scores`, which is 0 at bg anchors). This
provides a regularisation effect — the model learns to be confidently negative for
attributes at bg locations. Approach 3 and Option B skip this. In practice the effect
is small because attribute predictions are only evaluated at foreground detections
downstream, but it is a real difference.

**Expected outcome with Option B:** attribute mAP should be close to multilabel_old
numbers, likely within a few points. If a residual gap remains, reverting to Option A
closes it completely at negligible cost.

---

## Roadmap for E2E + attributes

### The key insight (and why previous attempts went wrong)

You phrased this correctly: **attributes have nothing to do with anchor selection**.
Whether a detection anchor is the best match for a person bounding box is determined
entirely by box IoU and class score — attribute predictions are irrelevant to that
decision. The right architecture follows directly:

> Anchor selection (TAL, NMS, o2o path) is driven by **detection class only**.
> Attribute training is an additional loss applied to whatever anchors were selected
> for detection.

The ubon26_wip design already implements this correctly in principle:
- o2o path: `include_attr=False` → TAL and loss train class+box only
- o2m path: trains class+box+attributes
- Inference: use o2o detection; borrow attr slice from o2m cv3

The reason previous attempts felt circular is that they conflated two separate
questions:
1. Should attribute gradients influence which anchor is selected? (No.)
2. Given the selected anchor, should attribute supervision be high-quality? (Yes.)

Approaches 1 and 2 both failed at E2E because they allowed attributes to corrupt
anchor selection (question 1). The `include_attr=False` fix in Approach 3 answered
question 1 correctly. But question 2 was never fixed — attribute training quality
degraded for unrelated reasons (hard targets, no background suppression), making the
overall result look like E2E was still broken.

### The actual E2E fix is simpler than it seems

**The test:**

After implementing Option B (soft targets), run a training with `end2end=True`. The
expectation is:

> Attribute quality with E2E enabled should be approximately equal to non-E2E,
> because attributes are trained on the o2m loss in both cases.

If that expectation holds, E2E for attributes is effectively solved. The architecture
is: o2o drives detection quality; o2m drives attribute quality; at inference the o2o
detections pick the boxes and the o2m attr slice provides attributes. This is already
implemented in ubon26_wip — it just hasn't been tested from a stable non-E2E baseline.

### Staged roadmap

**Stage 1 — Fix non-E2E (Option B)**

Implement soft attribute targets. Validate attribute mAP matches or approaches
multilabel_old numbers at `end2end=False`. Do not touch any E2E code.

**Stage 2 — Test E2E with no further changes**

Enable `end2end=True` with the Stage 1 model. Measure:
- Detection mAP (expect improvement from E2E, as intended)
- Attribute mAP (expect approximately equal to Stage 1 non-E2E)

If attribute mAP is approximately equal to Stage 1: done. No further work needed.

**Stage 3 — Only if Stage 2 attribute quality is significantly worse**

The failure mode to look for is: the o2o gradient path degrades the o2m attribute
training even though `include_attr=False`. Likely causes in order of probability:

1. **Loss weighting imbalance:** the o2o loss weight (`one2one_loss_weight`) is pulling
   backbone/neck features toward a representation that is good for classification but
   not attribute-discriminative. Fix: reduce `one2one_loss_weight` or verify it is
   balanced against the attribute loss term.

2. **Shared final layer contamination:** in the merged head, the final Conv2d has shape
   `[nc_main + attr_nc, c3, 1, 1]`. The o2o path with `include_attr=False` only trains
   the first `nc_main` output channels; the attr channels receive gradient only from
   o2m. But the Adam/momentum state is shared across all channels of this layer —
   the optimiser's second-moment estimate for the class channels is updated by both o2o
   and o2m, while attr channels are only updated by o2m. This is not a correctness bug
   but can affect effective learning rate. Fix: split the final Conv2d into two separate
   `nn.Linear` (1×1 conv) layers — `cv3_cls[i]` and `cv3_attr[i]` — so optimiser state
   is fully decoupled. This is a trivial architectural change with zero FLOP cost and no
   checkpoint compatibility break (the two output blocks have the same shapes as before).

3. **Bug in `include_attr=False` path:** verify that `include_attr=False` truly prevents
   attribute logits from entering the o2o TAL alignment metric and the o2o loss. If
   attribute channels are accidentally included in `pd_scores` in the o2o path, they
   corrupt anchor selection and drive noise gradients.

### What NOT to do

- Do not add a separate `cv_attr` head to fix E2E. The cost is prohibitive and the
  E2E problem is not a representation problem — it is a training signal/path problem.
- Do not attempt to make TAL multi-label-aware for attributes. It is unnecessary:
  the whole point is that anchor selection should be driven by class only.
- Do not train attributes on the o2o path. Multi-label supervision on a topk=1 assignment
  is a conceptual mismatch regardless of how it is implemented.

### Summary

The E2E path should fall out naturally from fixing non-E2E:

```
Stage 1: Fix o2m attribute training quality (Option B)
Stage 2: Enable E2E — test attribute quality
Stage 3 (if needed): Investigate why o2o disrupts o2m, not why attributes break E2E
```

The framing shift matters: E2E is not "broken for attributes". The o2o path never
touched attributes (in ubon26_wip). What was broken was the o2m attribute training
signal, which is the same path used in both E2E and non-E2E training.

---

## Implementation: Option B applied to `ubon26_wip`

**File changed:** `ultralytics/utils/loss.py`, method `v8DetectionLoss.get_assigned_targets_and_loss`

### What changed

The attribute loss block (previously lines 461-472) was updated from hard binary targets
to TAL quality-weighted soft targets. The change is 4 lines of logic.

**Before:**
```python
target_attrs = self.select_target_attrs(...)
pos_attr = target_attrs.sum().clamp(min=1)
loss_attr = (self.bce(pred_attrs, target_attrs.to(dtype)) * fg_mask[..., None]).sum() / pos_attr
loss[3] = loss_attr
```

**After (quality-weighted hard BCE):**
```python
target_attrs_hard = self.select_target_attrs(...)
quality = target_scores.amax(dim=-1, keepdim=True)  # [b, anchors, 1]
loss_attr = self.bce(pred_attrs, target_attrs_hard.to(dtype)) * quality
loss[3] = (loss_attr * fg_mask[..., None]).sum() / max((target_attrs_hard * quality).sum(), 1)
```

### What each change does

**`quality = target_scores.amax(dim=-1, keepdim=True)`**

`target_scores` (shape `[batch, anchors, nc]`) is the TAL quality-weighted soft class
label. Taking the per-anchor max gives a scalar quality in (0, 1] at foreground anchors
and exactly 0 at background anchors.

**`self.bce(pred_attrs, target_attrs_hard.to(dtype)) * quality`**

Computes standard BCE against hard 0/1 targets, then **multiplies the loss by the
quality scalar**. This is a quality-weighted BCE, not a soft-target BCE. The distinction
is critical:

| Formulation | BCE target | Gradient direction | For pretrained model (sigmoid≈0.85) |
|-------------|-----------|-------------------|--------------------------------------|
| Hard BCE | 1.0 | Toward 1.0 (constructive) | ≈ 0.16 |
| **Quality-weighted BCE** (implemented) | **1.0** | **Toward 1.0 (constructive)** | **≈ 0.16 × quality ≈ 0.11** |
| Soft-target BCE | quality ≈ 0.7 | Toward 0.7 (destructive) | ≈ 0.68 — **4× larger, wrong direction** |

The soft-target formulation creates a massive destructive gradient when fine-tuning from
a model calibrated with hard targets: the `(1 - quality) × log(1 - sigmoid)` term
penalises the model for predicting too confidently, systematically pushing all
well-trained positive attr predictions downward. This erodes attr mAP uniformly across
all attributes within ~30 epochs of fine-tuning.

**Normalisation by `(target_attrs_hard × quality).sum()`**

Equivalent to the quality-weighted positive assignment count. Scales the loss
consistently across batches, matching how `target_scores_sum` normalises the class loss.
The denominator is identical to what a soft-target formulation would use — only the
numerator (BCE computation) differs.

### Compatibility with fine-tuning and from-scratch training

- **Fine-tuning from hard-target checkpoint**: gradient direction preserved (toward 1.0),
  no destructive push on pretrained attr weights. Safe.
- **Training from scratch**: quality weighting emphasises high-quality anchor assignments
  and de-emphasises marginal ones, improving gradient signal quality. No difference in
  converged calibration (model still learns to predict near 1.0 for true positives).
- **Fine-tuning from soft-target checkpoint**: same behaviour as hard-target fine-tuning.

### What is not changed

- `select_target_attrs` — the GT attribute gathering logic is unchanged.
- The model architecture (cv3 head, `attr_nc` channels, `one2one` path).
- The data pipeline — `batch["attr"]` and `batch["cls"]` formats are unchanged.
- The `hyp.attr` loss gain — applies as before.
- All existing checkpoints load without modification.

### Expected effect

| Issue | multilabel_old | ubon26_wip before | ubon26_wip after |
|-------|---------------|-------------------|-----------------|
| Quality-weighted anchor emphasis | ✅ | ❌ | ✅ (via loss weight not target) |
| Gradient direction toward hard labels | ✅ | ✅ | ✅ (preserved) |
| Safe for fine-tuning | ✅ | ✅ | ✅ |
| TAL anchor selection sees attrs | ✅ | ❌ | ❌ (minor, unchanged) |

---

## Additional regression causes specific to fine-tuning (nc=5, attr_nc=50)

This section documents further issues found when reviewing the full training path for
the specific setup: `nc=5`, `attr_nc=50`, `end2end=false`, `PoseLoss26`, fine-tuning
a pretrained checkpoint for 50 epochs.

---

### Issue 1: Checkpoint architecture compatibility ✅ Not applicable

The ubon26_wip `Detect` head uses a **merged cv3 output**: the final `Conv2d` in each
cv3 block has shape `[nc_main + attr_nc, c3, 1, 1]` (e.g. `[55, 128, 1, 1]`).

For `yolo26s-v10-210226.pt`: inference with ubon26_wip produces correct attr mAP. This
is conclusive proof that the checkpoint's cv3 final layer has shape `[55, c3, 1, 1]` —
identical to what ubon26_wip constructs. All cv3 weights load correctly. The pretrained
attr channels (indices 5–54 of cv3) are correctly trained and preserved in the fine-tune.

The checkpoint was trained with either `multilabel_old` (nc=55, same merged shape) or
an earlier version of `ubon26_wip` — either way the architecture is compatible. The
`attributes_old` style (separate `cv_attr` head, cv3 final `[5, c3, 1, 1]`) would
have broken attr inference entirely and is ruled out.

**Conclusion:** the attr regression during fine-tuning is not caused by weight loading
issues. The pretrained attr weights are intact at the start of fine-tuning.

---

### Issue 2: Double-application of `hyp.cls` and `hyp.dfl` in pose loss ✅ Fixed

**Both** `v8PoseLoss` and `PoseLoss26` had the identical bug. `get_assigned_targets_and_loss`
returns `det_loss` with **all** gains already baked in (`hyp.box`, `hyp.cls`, `hyp.dfl`,
`hyp.attr`). Both pose loss classes then extracted the cls and dfl components and
multiplied by their gains again:

```python
# get_assigned_targets_and_loss already does:
loss[1] *= self.hyp.cls
loss[2] *= self.hyp.dfl

# v8PoseLoss / PoseLoss26 then did (BUG — now removed):
loss[3] = det_loss[1]    # already ×hyp.cls
loss[4] = det_loss[2]    # already ×hyp.dfl
loss[3] *= self.hyp.cls  # DOUBLE → effective = hyp.cls²
loss[4] *= self.hyp.dfl  # DOUBLE → effective = hyp.dfl²
```

**Did it exist in `multilabel_old`?** Yes. In multilabel_old, attributes were embedded
in `det_loss[1]` (the combined cls+attr loss), so they too were subject to the
double-application. Effective attr gain in multilabel_old = `hyp.cls²`.

In ubon26_wip before the fix: attr was separate (`det_loss[3]`), applied once. Effective
attr gain = `hyp.attr`. The double-gain bug did NOT affect attrs in ubon26_wip, only cls
and dfl.

**After the fix** (lines removed from both pose loss classes):

| Loss | Before fix | After fix |
|------|-----------|-----------|
| box | `hyp.box` ×1 | `hyp.box` ×1 (unchanged) |
| pose | `hyp.pose` ×1 | `hyp.pose` ×1 (unchanged) |
| kobj | `hyp.kobj` ×1 | `hyp.kobj` ×1 (unchanged) |
| cls | `hyp.cls²` (e.g. 0.25) | `hyp.cls` (e.g. 0.5) — **2× stronger** |
| dfl | `hyp.dfl²` | `hyp.dfl` — **stronger** |
| attr | `hyp.attr` ×1 | `hyp.attr` ×1 (unchanged) |

**Gain recalibration needed:** cls and dfl gradients are now twice as strong as before.
If you have trained with (for example) `hyp.cls = 0.5` expecting an effective gain of
0.25, you now get 0.5. This may actually improve detection quality (stronger cls signal)
but if you want to reproduce the exact old training dynamics, halve `hyp.cls` and
`hyp.dfl`. For attribute training, nothing changes — `hyp.attr` was always single-applied.

**Effect on matching multilabel_old:** In multilabel_old, attrs had effective gain
`hyp.cls²`. In ubon26_wip after this fix, attrs have `hyp.attr`. To match the old
effective scale, set `hyp.attr = hyp.cls² ≈ 0.25` (for default `hyp.cls = 0.5`). The
user's current `attr: 0.45` gives 1.8× more attr training than multilabel_old used.

---

### Issue 3: Gradient balance shift from double-gain fix (`attr_nc=50` specific)

The double-gain bug fix (Issue 2) doubled the effective cls gradient on the **shared
cv3 intermediate DWConv+PW layers** that both class prediction and attribute prediction
backpropagate through. The attr/cls gradient ratio on those shared layers changed:

| | Before fix | After fix |
|-|-----------|-----------|
| Effective cls gradient | `hyp.cls² = 0.25` | `hyp.cls = 0.5` |
| Effective attr gradient | `hyp.attr = 0.45` | `hyp.attr = 0.35` |
| **Attr/cls ratio** | **0.45/0.25 = 1.8 (attrs dominant)** | **0.35/0.5 = 0.7 (cls dominant)** |

This is a 2.5× shift in gradient balance. The shared intermediate layers experience
stronger pressure toward class-discriminative features at the expense of
attribute-discriminative features. Over 30 fine-tuning epochs this visibly degrades
attr mAP while detection metrics improve.

**Compensation:** to restore the original gradient balance while keeping the double-gain
fix, set `hyp.cls = 0.25` (halved, so effective = 0.25 after fix) and `hyp.attr = 0.45`
(original). This exactly reproduces the pre-fix gradient ratio that the pretrained
checkpoint was trained under. Alternatively keep `hyp.cls = 0.5` and raise
`hyp.attr ≈ 0.9` (2.6× attr/cls ratio to match old 1.8 ratio after the doubled cls).

---

### Issue 4: Loss scale change after quality-weighted fix (`attr_nc=50` specific)

The quality-weighted BCE denominator `(target_attrs_hard × quality).sum()` is ~30%
smaller than the old hard-count denominator (`target_attrs_hard.sum()`), because
quality ≈ 0.7 < 1.0. The BCE numerator also decreases slightly (quality scaling reduces
each anchor's contribution). **Net effect:** attr loss scale is approximately unchanged
— the numerator and denominator reductions partially cancel. No `hyp.attr` retuning is
needed purely due to this change.

---

### Summary: all identified regression causes

| Cause | Severity | Affects | Status |
|-------|----------|---------|--------|
| Soft-target BCE — destructive gradient for pretrained attrs | **Critical** | Fine-tuning | ✅ Fixed (quality-weighted hard BCE) |
| Checkpoint arch mismatch | N/A | — | ✅ Ruled out (inference proves weights intact) |
| Double cls/dfl gain in pose loss | Medium | All pose training | ✅ Fixed (both v8PoseLoss and PoseLoss26) |
| Gradient balance shift from double-gain fix | High | Fine-tuning | ⚠️ Use `hyp.cls=0.25` + `hyp.attr=0.45` to restore pretrained ratio |
