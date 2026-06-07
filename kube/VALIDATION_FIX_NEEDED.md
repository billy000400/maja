# Validation Code Fix Needed

## Issue Discovered During Testing

The validation code in `spanet/network/resonance_regression/resonance_regression_validation.py` needs to be updated to handle 2D `gen_mass_logits` targets.

## Problem

When `gen_mass_logits` is stored as 2D logits (shape: [batch_size, num_classes]), the validation code fails with:
```
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

This happens at line 135 in `resonance_regression_validation.py`:
```python
pred_class_valid = pred_class[valid_mask]
```

The issue is that when `target_values` is 2D (from `gen_mass_logits`), `valid_mask = ~np.isnan(target_values)` is also 2D, but `pred_class` is 1D.

## Required Fix

In `resonance_regression_validation.py`, after the line:
```python
target_values = regression_targets_np[key]
```

Add:
```python
# Handle 2D targets (like gen_mass_logits) - convert to class indices
if hasattr(target_values, "ndim") and target_values.ndim > 1:
    target_values = target_values.argmax(axis=-1)
```

Also, ensure `pred_class` is properly converted from tensor to numpy and handle 2D case:
```python
pred_class = predictions[f"{key}_pred_class"]
# Convert to numpy and handle 2D case
import torch
if isinstance(pred_class, torch.Tensor):
    pred_class = pred_class.detach().cpu().numpy()
else:
    pred_class = np.asarray(pred_class)
# If 2D (logits), take argmax to get class indices
if pred_class.ndim > 1:
    pred_class = pred_class.argmax(axis=-1)
```

## Current Status

- ✅ `gen_mass_logits` added to h5 files (both training and test)
- ✅ `gen_mass_logits` added to event_info file as regression target
- ✅ Validation code updated to skip keys not in regression_targets
- ❌ Validation code needs fix for 2D target handling (file modification not persisting in pod)

## Next Steps

1. Apply the validation code fix in the SPANet repository
2. Re-test training
3. Once validation passes, launch full 500-epoch training
