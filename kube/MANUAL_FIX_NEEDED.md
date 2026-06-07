# Manual Fix Required for Validation Code

## Issue
File edits to `/maad-vol/SPANet/spanet/network/resonance_regression/resonance_regression_validation.py` are not persisting through kubectl exec commands.

## Required Fix

In the file `/maad-vol/SPANet/spanet/network/resonance_regression/resonance_regression_validation.py`, around line 125, in the classification loop section, add the following code:

**Location:** Right after the line:
```python
target_values = regression_targets_np[key]
```

**Add:**
```python
                # Handle 2D targets (like gen_mass_logits) - convert to 1D class indices
                if hasattr(target_values, "ndim") and target_values.ndim > 1:
                    target_values = target_values.argmax(axis=-1)
```

**OR** add this check earlier in the loop to skip `gen_mass_logits` entirely:

Right after:
```python
            for key in regression_targets_np:
```

Add:
```python
                # Skip logits targets (like gen_mass_logits) - they are 2D, not continuous values
                if key.endswith("_logits"):
                    continue
```

## How to Apply

You can apply this fix by:
1. Exec into the pod: `kubectl exec -it maja-pod-billy -- bash`
2. Edit the file: `vi /maad-vol/SPANet/spanet/network/resonance_regression/resonance_regression_validation.py`
3. Or use: `kubectl exec -it maja-pod-billy -- nano /maad-vol/SPANet/spanet/network/resonance_regression/resonance_regression_validation.py`

## Current Status

- ✅ `gen_mass_logits` added to h5 files
- ✅ `gen_mass_logits` added to event_info
- ✅ Validation code updated to skip missing keys
- ❌ Validation code fix for 2D targets not applied (file edits not persisting)
