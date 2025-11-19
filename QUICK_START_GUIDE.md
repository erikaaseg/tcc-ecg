# GradCAM Analysis - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
```bash
# Ensure you have Python 3.8+ installed
python --version

# Install required packages (if not already installed)
pip install torch torchvision pytorch-lightning
pip install grad-cam albumentations
pip install matplotlib numpy scipy pandas
```

### Running the Analysis

1. **Open the Notebook:**
   ```bash
   cd /home/lhsmello/projects/medical/ECG-tcc-erika/notebooks
   jupyter notebook gradcam.ipynb
   # or use VS Code notebook interface
   ```

2. **Execute Cells in Order:**
   - Run cells sequentially from top to bottom
   - First cell loads the trained model
   - Subsequent cells perform different analyses

3. **Expected Runtime:**
   - Full notebook: ~10-15 minutes (CPU) or ~3-5 minutes (GPU)
   - Individual visualizations: ~5-30 seconds each

---

## 📊 What Each Section Does

### Section 1-2: Setup
- Loads the trained ECG classifier model
- Prepares the dataset with transformations
- Defines utility functions

**Action:** Just run these cells to set up the environment

### Section 3-4: Basic Visualization
- Single sample GradCAM example
- Comparison of 8 different CAM methods

**Output:** Visual heatmaps showing model attention

### Section 5: Class-wise Analysis
- Shows 3 samples from each class (MI, PMI, HB, Normal)
- Displays predictions with confidence scores
- Color-coded: Green (correct), Red (incorrect)

**Use Case:** Understand class-specific attention patterns

### Section 6: Quantitative Analysis
- Extracts numerical metrics from activation maps
- Creates box plots comparing classes
- Shows spatial distribution of attention centers

**Use Case:** Statistical comparison between classes

### Section 7: Misclassification Analysis
- Identifies all incorrect predictions
- Creates confusion matrix for errors
- Visualizes failure cases with GradCAM

**Use Case:** Understand model limitations and failure modes

### Section 8: Confidence Analysis
- Plots confidence vs activation intensity
- Separates correct vs incorrect predictions
- Calculates correlation statistics

**Use Case:** Assess model calibration and uncertainty

### Section 9: Layer-wise Analysis
- Compares GradCAM across ResNet layers
- Shows progression from low to high-level features

**Use Case:** Understand hierarchical feature learning

### Section 10-13: Documentation
- Summary statistics and findings
- Key insights and interpretations
- Recommendations for next steps
- Comprehensive conclusions

**Use Case:** Reference for study outcomes and future work

---

## 🎯 Common Tasks

### Generate GradCAM for a Specific Sample

```python
# Choose sample index
sample_idx = 42  # Change this number

# Generate visualization
viz, true_label, pred_label, conf, all_probs = plot_sample_with_prediction(sample_idx)

# Display
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.imshow(viz)
plt.title(f"True: {classes[true_label]}, Pred: {classes[pred_label]} ({conf:.2%})")
plt.axis('off')
plt.show()
```

### Compare Multiple CAM Methods for One Image

```python
data_idx = 100  # Choose your sample

methods = [GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, method in enumerate(methods):
    input_tensor = ecg_dataset[data_idx][0].unsqueeze(0)
    rgb_img = inverse_transform(input_tensor[0].numpy().transpose(1, 2, 0))
    rgb_img = np.array(rgb_img) / 255.0
    
    with method(model=model, target_layers=[model.backbone.layer4[-1]]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    
    axes[idx].imshow(visualization)
    axes[idx].set_title(method.__name__)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

### Analyze All Samples of a Specific Class

```python
class_to_analyze = 0  # 0=MI, 1=PMI, 2=HB, 3=Normal

# Find samples of this class
class_samples = [i for i, label in enumerate(ecg_dataset.dataset.labels) 
                 if label == class_to_analyze]

print(f"Class {classes[class_to_analyze]}: {len(class_samples)} samples")

# Visualize first 6 samples
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, sample_idx in enumerate(class_samples[:6]):
    viz, true_label, pred_label, conf, _ = plot_sample_with_prediction(sample_idx)
    axes[idx].imshow(viz)
    axes[idx].set_title(f"Pred: {classes[pred_label]} ({conf:.2%})")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

### Extract Activation Statistics

```python
# For a single sample
sample_idx = 50
input_tensor = ecg_dataset[sample_idx][0].unsqueeze(0)

with GradCAM(model=model, target_layers=[model.backbone.layer4[-1]]) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    stats = analyze_activation_pattern(grayscale_cam[0])

print("Activation Statistics:")
for key, value in stats.items():
    if key != 'center_of_mass':
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: ({value[0]:.1f}, {value[1]:.1f})")
```

---

## 🔧 Customization Options

### Change the Model Checkpoint

```python
# In the first cell, modify:
model_path = "lightning_logs/ecg_clf/version_X/checkpoints/YOUR_CHECKPOINT.ckpt"
model = ECGClassifier.load_from_checkpoint(model_path)
```

### Use Different Target Layers

```python
# For earlier layers:
target_layers = [model.backbone.layer3[-1]]  # Instead of layer4

# For multiple layers (combined):
target_layers = [model.backbone.layer3[-1], model.backbone.layer4[-1]]
```

### Adjust Visualization Parameters

```python
# Change figure sizes
fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Larger figures

# Change colormap
plt.imshow(visualization, cmap='jet')  # Different colormap

# Save figures
plt.savefig('gradcam_output.png', dpi=300, bbox_inches='tight')
```

---

## 📈 Interpreting Results

### GradCAM Heatmap Colors

- 🟥 **Red/Warm colors:** High activation - model focuses here strongly
- 🟦 **Blue/Cool colors:** Low activation - less important for prediction
- 🟨 **Yellow/Orange:** Moderate activation - some relevance

### Confidence Scores

- **> 95%:** Very confident - likely correct
- **85-95%:** Confident - usually correct
- **70-85%:** Moderate confidence - review carefully
- **< 70%:** Low confidence - high uncertainty

### Activation Patterns

**Focused (Localized):**
- Sharp, concentrated regions
- Often associated with high confidence
- Clear diagnostic feature identified

**Diffuse (Distributed):**
- Broad, spread-out activation
- May indicate uncertainty
- Multiple features considered

**Noisy:**
- Many small scattered regions
- Potentially problematic
- May need model improvement

---

## ⚠️ Troubleshooting

### Issue: Out of Memory Error

**Solution:**
```python
# Reduce batch size or use CPU
import torch
device = 'cpu'  # Instead of 'cuda'
model = model.to(device)
```

### Issue: Slow GradCAM Generation

**Solutions:**
1. Use GPU if available: `model = model.to('cuda')`
2. Use faster methods: `GradCAM` instead of `ScoreCAM`
3. Reduce number of samples analyzed

### Issue: Import Errors

**Solution:**
```bash
# Install missing packages
pip install pytorch-grad-cam
pip install albumentations
pip install scipy
```

### Issue: Dataset Not Found

**Solution:**
```python
# Update DATA_ROOT_DIR to your dataset location
DATA_ROOT_DIR = '/path/to/your/dataset/'
```

---

## 📚 Additional Resources

### Documentation
- Full Study Documentation: `GRADCAM_STUDY_DOCUMENTATION.md`
- Model Code: `ecgclassifier_model.py`
- Dataset Code: `dataset.py`

### External References
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)
- [pytorch-grad-cam GitHub](https://github.com/jacobgil/pytorch-grad-cam)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

### Support
- GitHub Issues: [Repository Issues Page]
- Documentation: See `GRADCAM_STUDY_DOCUMENTATION.md`

---

## ✅ Quick Checklist

Before running the full analysis:
- [ ] Model checkpoint exists at specified path
- [ ] Dataset is accessible at `DATA_ROOT_DIR`
- [ ] All dependencies installed
- [ ] Sufficient RAM/GPU memory
- [ ] Results directory for saving figures (optional)

For questions or issues:
1. Check this guide first
2. Review full documentation
3. Check code comments in notebook
4. Open GitHub issue if needed

---

**Last Updated:** November 3, 2025  
**Notebook Version:** Enhanced v1.0  
**Compatibility:** Python 3.8+, PyTorch 2.0+
