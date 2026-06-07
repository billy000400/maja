# Adding gen_mass_logits to H5 Files

This directory contains scripts to add `gen_mass_logits` to h5 files, which is required for the resonance mass classification training.

## Problem

The training code expects `EVENT/gen_mass_logits` in the h5 files, but it's missing. This causes a `KeyError: 'EVENT/gen_mass_logits'` during validation.

## Solution

The script `add_gen_mass_logits.py` reads the mass information from the h5 file, calculates which mass category each sample belongs to (based on the mass classes: 200, 300, 400, 500, 600, 700, 800, 900, 1000), and adds the `gen_mass_logits` dataset to the h5 file.

## Usage

### Step 1: Copy the script to the pod

Once the pod is ready, copy the script:

```bash
kubectl cp add_gen_mass_logits.py <pod-name>:/maad-vol/SPANet/add_gen_mass_logits.py
```

Or if you're already in the pod:

```bash
# The script should be accessible from /maad-vol/SPANet if copied there
```

### Step 2: Process the h5 files

Run the script on your h5 files. For example:

```bash
# Inside the pod
cd /maad-vol/SPANet
python3 add_gen_mass_logits.py /maad-vol/data/train_datasets/your_file.h5
```

The script will:
- Automatically find the mass information in the h5 file
- Calculate gen_mass_logits based on the mass classes
- Add `EVENT/gen_mass_logits` to the h5 file (modifies in place by default)

### Step 3: Test with one epoch

Use the test job configuration:

```bash
kubectl create -f jobs/global_classification_only/test_one_epoch.yaml
```

This will run training for 1 epoch to verify that the gen_mass_logits are correctly formatted.

### Step 4: Launch full training

If the test succeeds, launch the full training:

```bash
kubectl create -f jobs/global_classification_only/global_classification_only.yaml
```

## Script Options

```bash
python3 add_gen_mass_logits.py <h5_file> [options]

Options:
  --mass-classes MASS_CLASSES    Comma-separated list of mass classes
                                  (default: 200,300,400,500,600,700,800,900,1000)
  --output OUTPUT                 Output file path (if not specified, modifies in place)
  --no-inplace                    Do not modify file in place (requires --output)
```

## How it works

1. The script searches for mass information in the h5 file (tries common keys like `EVENT/gen_mass`, `EVENT/mass`, etc.)
2. For each sample, it finds the closest mass class from the list [200, 300, 400, 500, 600, 700, 800, 900, 1000]
3. Creates a logit vector where the correct class has a high value (10.0) and others are 0.0
4. Saves this as `EVENT/gen_mass_logits` in the h5 file

## Integration with combine_*.py scripts

If you have existing `combine_*.py` scripts that process h5 files, you can integrate this functionality by:

1. Importing the function:
```python
from add_gen_mass_logits import calculate_gen_mass_logits, find_mass_key
```

2. Adding the calculation after reading the mass:
```python
import h5py
import numpy as np

with h5py.File('your_file.h5', 'r+') as f:
    mass_key = find_mass_key(f)
    mass_data = f[mass_key][:]
    mass_classes = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    gen_mass_logits = calculate_gen_mass_logits(mass_data, mass_classes)
    
    if 'EVENT' not in f:
        f.create_group('EVENT')
    if 'EVENT/gen_mass_logits' in f:
        del f['EVENT/gen_mass_logits']
    f.create_dataset('EVENT/gen_mass_logits', data=gen_mass_logits, compression='gzip')
```
