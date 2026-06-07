# Step-by-Step Instructions

## Step 1: Process H5 Files (Add gen_mass_logits)

After you've authenticated with kubectl, run these commands:

```bash
# 1. Copy the script to the pod
kubectl cp add_gen_mass_logits.py maja-pod-billy:/maad-vol/SPANet/add_gen_mass_logits.py

# 2. Make it executable
kubectl exec maja-pod-billy -- chmod +x /maad-vol/SPANet/add_gen_mass_logits.py

# 3. Process all h5 files in training_sets directory
kubectl exec maja-pod-billy -- bash -c "
    cd /maad-vol/data/training_sets && 
    for file in *.h5; do
        echo \"Processing \$file...\"
        python3 /maad-vol/SPANet/add_gen_mass_logits.py \"\$file\" --mass-classes \"200,300,400,500,600,700,800,900,1000\"
        if [ \$? -eq 0 ]; then
            echo \"  ✓ Successfully processed \$file\"
        else
            echo \"  ✗ Failed to process \$file\"
        fi
    done
"

# 4. Verify one file was processed correctly
kubectl exec maja-pod-billy -- python3 -c "
import h5py
import glob
files = glob.glob('/maad-vol/data/training_sets/*.h5')
if files:
    f = h5py.File(files[0], 'r')
    print('Keys in EVENT group:', list(f['EVENT'].keys()))
    if 'gen_mass_logits' in f['EVENT']:
        print('✓ gen_mass_logits found!')
        print('  Shape:', f['EVENT/gen_mass_logits'].shape)
        print('  Dtype:', f['EVENT/gen_mass_logits'].dtype)
    else:
        print('✗ ERROR: gen_mass_logits not found!')
"
```

## Step 2: Test Training (1 Epoch)

Launch the test training job to verify everything works:

```bash
kubectl create -f jobs/global_classification_only/test_one_epoch.yaml
```

Monitor the job:

```bash
# Check job status
kubectl get jobs hhh-job-resonance-mass-classification-test

# Check pod status
kubectl get pods -l job-name=hhh-job-resonance-mass-classification-test

# View logs
kubectl logs -l job-name=hhh-job-resonance-mass-classification-test --tail=100 -f
```

**Expected result:** The training should complete 1 epoch without the `KeyError: 'EVENT/gen_mass_logits'` error.

## Step 3: Launch Full Training (If Test Succeeds)

If the test training completes successfully without errors, launch the full training:

```bash
kubectl create -f jobs/global_classification_only/global_classification_only.yaml
```

Monitor the full training:

```bash
# Check job status
kubectl get jobs hhh-job-resonance-mass-classification

# Check pod status (there will be 5 pods due to parallelism: 5)
kubectl get pods -l job-name=hhh-job-resonance-mass-classification

# View logs from one of the pods
kubectl logs <pod-name> --tail=100 -f
```

## Troubleshooting

If you encounter the `KeyError: 'EVENT/gen_mass_logits'` error:

1. Verify the h5 files were processed:
   ```bash
   kubectl exec maja-pod-billy -- python3 -c "
   import h5py
   import glob
   for fname in glob.glob('/maad-vol/data/training_sets/*.h5'):
       f = h5py.File(fname, 'r')
       if 'gen_mass_logits' not in f.get('EVENT', {}):
           print(f'Missing gen_mass_logits in: {fname}')
   "
   ```

2. Re-process any files that are missing gen_mass_logits

3. Check that the mass information exists in the h5 files:
   ```bash
   kubectl exec maja-pod-billy -- python3 -c "
   import h5py
   import glob
   f = h5py.File(glob.glob('/maad-vol/data/training_sets/*.h5')[0], 'r')
   print('Available keys in EVENT:', list(f.get('EVENT', {}).keys()))
   "
   ```
