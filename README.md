# Tomato Leaf Disease Classification using Self-Supervised and Semi-Supervised Learning

This project presents a **comparative study of Supervised, Self-Supervised, and Semi-Supervised learning methods** for **tomato leaf disease classification** under **limited labeled data** settings.

The primary motivation is to demonstrate how **label-efficient learning paradigms** can significantly outperform traditional supervised learning when labeled data is scarce — a common scenario in agricultural and real-world datasets.

---

## Project Highlights

- Tomato-only dataset (10 disease classes)
- Very low labeled data regime (10%)
- Comparison across:
  - Supervised Learning
  - Self-Supervised Learning (SimCLR, BYOL)
  - Semi-Supervised Learning (FixMatch, MixMatch)
- Clean experimental setup with fair comparisons
- GPU-aware design (no large-scale label percentage sweeps)

---

## Repository Structure
├── self_supervised_comparison.ipynb
├── semi_supervised_comparison.ipynb
├── README.md


### 📓 Notebooks
- **`self_supervised_comparison.ipynb`**
  - Supervised baseline
  - SimCLR pretraining + fine-tuning
  - BYOL pretraining + fine-tuning

- **`semi_supervised_comparison.ipynb`**
  - FixMatch (with supervised warm-up)
  - MixMatch (soft pseudo-labeling)
  - Both trained from scratch (no SSL weights)

---

## Dataset Details

- **Dataset:** PlantVillage (Tomato subset only)
- **Number of classes:** 10
- **Classes include:**
  - Tomato_Bacterial_spot
  - Tomato_Early_blight
  - Tomato_Late_blight
  - Tomato_Leaf_Mold
  - Tomato_Septoria_leaf_spot
  - Tomato_Spider_mites_Two_spotted_spider_mite
  - Tomato_Target_Spot
  - Tomato_Tomato_mosaic_virus
  - Tomato_Tomato_YellowLeaf_Curl_Virus
  - Tomato_healthy

- **Total images:** ~16,000+
- **Labeled data used:** 10%
- **Unlabeled data:** 80%
- **Validation data:** Remaining 10%

---

## Experimental Setup

- **Backbone:** ResNet-18
- **Image size:** 224 × 224
- **Optimizer:** Adam
- **Loss Functions:**
  - Cross-Entropy (supervised)
  - NT-Xent (SimCLR)
  - BYOL loss
  - Consistency losses (FixMatch, MixMatch)
- **Training environment:** Kaggle Notebooks (GPU constrained)

---

## Results Summary (10% Labeled Data)

| Method | Learning Paradigm | Validation Accuracy |
|------|------------------|---------------------|
| **Supervised (Baseline)** | Supervised | **49.31%** |
| **SimCLR + Fine-tuning** | Self-Supervised | **≈ 76–77%** |
| **BYOL + Fine-tuning** | Self-Supervised | **≈ 74–75%** |
| **FixMatch** | Semi-Supervised | **≈ 73.4%** |
| **MixMatch** | Semi-Supervised | **≈ 79.15%** |

---

## Key Observations

- Supervised learning **performs poorly** with limited labeled data.
- Self-Supervised learning (SimCLR, BYOL) **learns strong representations** without labels.
- Semi-Supervised learning further improves performance by leveraging unlabeled data.
- **MixMatch outperformed FixMatch** in this setup, likely due to:
  - Soft pseudo-labeling
  - Better handling of uncertainty in fine-grained tomato disease classes

---

## Limitations

- Label percentage experiments (1%–50%) were **not conducted** due to GPU constraints.
- FixMatch and MixMatch were trained **without SSL initialization** in the semi-supervised notebook.
- No hyperparameter sweeping was performed.

These limitations are intentional and documented.

---

## Conclusion

This project clearly demonstrates that:

> **Self-Supervised and Semi-Supervised learning methods dramatically outperform traditional supervised learning in low-label regimes for tomato disease classification.**

Even with only **10% labeled data**, modern label-efficient learning methods achieved **up to ~30% absolute improvement** over supervised baselines.

---

## Future Work

- Label percentage experiments (1%–50%)
- SSL-initialized FixMatch / MixMatch
- Grad-CAM based explainability
- Lightweight models for mobile deployment

---

## 📜 License

This project is for academic and educational purposes.
