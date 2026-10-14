# Fine-Tuning the TerraMind Foundation Model for LUCAS Scene Classification

**Author:** Maria Cristina Zambon  
**Project Context:** Passion in Action 2025  

## 📖 Overview
This repository documents a land cover scene classification task centered on the Czech Republic. The primary goal is to define the methodological pipeline—from data acquisition to model evaluation—and identify key limitations when adapting large-scale pre-trained models to small, specific datasets.

The project leverages the **TerraMind** geospatial foundation model (accessed via the **TerraTorch** framework) to classify georeferenced ground truth points from the Land Use and Coverage Area frame Survey (LUCAS).

### 🔗 Datasets
* [LUCAS Dataset](https://drive.google.com/file/d/1DJpG_0gZPNOw0S8FpcrHR21phFj9wGO7/view?usp=share_link)
* [Test Dataset](https://drive.google.com/file/d/1GfsKvCwwjgHfZS8CgUFVVBHcquBDU_UF/view?usp=share_link)

---

## 📊 Dataset Creation & Preparation

The dataset is built using **LUCAS**, a harmonized in-situ land cover and land use data collection program by Eurostat. The dataset encompasses **10 land cover classes**, including Arable land, Permanent crops, Grass, Wooded areas, Shrubs, Bare surface, Artificial constructions, Inland water, Transitional/Coastal Water, and uninterpretable areas.

### Data Processing Pipeline (Google Earth Engine)
1. **Filtering:** LUCAS points were filtered for the Czech Republic.
2. **Sampling:** Proportional sampling was used to maintain representative class distributions (yielding a total of 96 sample points). 
3. **Imagery Extraction:** Used NASA Harmonized Landsat Sentinel-2 (HLSS30 v002) filtering for 2018–2019 acquisitions with <10% cloud coverage.
4. **Bands & Compositing:** Selected 6 spectral bands (B4, B3, B2, B8, B11, B12). A median composite was generated to reduce atmospheric noise.
5. **Image Chips:** Extracted as 224×224 pixel GeoTIFFs, centered on LUCAS points.

### Dataset Splits & Normalization
* **Split:** Stratified 80/20 split (Training/Validation). Minimum representation checks were performed due to the scarcity of chips in rare classes (e.g., Shrubs, Bare structure).
* **Normalization:** Calculated per-band mean and standard deviation across the training dataset to stabilize training against surface reflectance variance.

---

## ⚙️ Model Configuration

The **`terramind_v1_tiny`** variant was chosen to balance computational efficiency with generalized geospatial features. Data was processed using TerraTorch's `GenericNonGeoClassificationDataModule`.

### Final Model Architecture
After iterative debugging, the model was refined to perform scene classification rather than semantic segmentation. 

```python
# Key arguments refined in the ClassificationTask
model_args = {
    "backbone_modalities": ["S2L2A"], # Configured with band mapping for HLS data
    "necks": [],                      # Removed incompatible segmentation necks
    "decoder": "IdentityDecoder",     # Passes backbone features directly to the head
    "num_classes": 10                 # Matches LUCAS dataset classes
}
```

### Debugging & Troubleshooting
* **Mismatched Architectures:** Initial training crashed due to `conv_transpose2d` layers attempting to upsample feature maps for segmentation. Fixed by removing segmentation-specific necks and flattening the features for image-level classification.
* **Loss Function Error:** PyTorch threw a `TypeError` because `ignore_index` was receiving a `None` value. Fixed by setting `ignore_index = -1` in the `ClassificationTask` constructor.
* **Inference Desynchronization:** PyTorch Lightning's standard `trainer.test` caused fatal `IndexError` crashes by re-initializing the DataModule and desynchronizing the file list. Solved by writing a **custom manual inference loop** to maintain a persistent state.

---

## 📈 Experiments & Results

Two primary fine-tuning strategies were tested to evaluate parameter adaptation vs. fixed feature extraction:

1. **Experiment 1: Frozen Backbone** (`freeze_backbone=True`)
   * *Result:* ~11.3% validation accuracy.
   * *Conclusion:* Generic pre-trained features were insufficient to differentiate nuanced LUCAS land cover classes without adaptation. Barely better than random guessing.
2. **Experiment 2: Full Fine-Tuning** (`freeze_backbone=False`)
   * *Settings:* Learning rate lowered to `1e-5`.
   * *Result:* Peak validation accuracy of ~19-21% (reaching a final baseline of ~32% on evaluated splits). 
   * *Conclusion:* `EarlyStopping` halted training after 8 epochs. The 5.6M parameter model successfully learned to map features, but the small dataset size heavily limited generalization.

### Evaluation Interpretations
* **Strengths:** The model showed a reasonable capacity to identify *Wooded areas* and *Inland water*.
* **Weaknesses:** Struggled significantly with *Artificial surfaces* and *Grass*.
* **Confusion Matrix Patterns:** 
  * *Artificial surfaces* misclassified as *Wooded areas* (likely due to spectral similarities between low-density urban vegetation and forests).
  * *Grass* consistently misclassified as *Wooded areas*.

---

## 🚀 Conclusions & Future Work

This project successfully established a working pipeline but highlighted the primary limitation of applying large foundation models to small datasets: **data starvation**. The model quickly exhausted the learning potential of the ~100 training samples. 

**Next Steps for Improvement:**
1. **Scale the Dataset:** Directly address data starvation by scaling the dataset from ~100 to 500+ image chips.
2. **Class Consolidation:** Merge semantically similar classes (e.g., Crops and Arable land) to simplify decision boundaries and improve robustness.
3. **Hyperparameter Tuning:** With a larger dataset, perform systematic tuning of learning rates and batch sizes to unlock further performance gains.
