# References

Sources that influenced specific parts of this codebase, grouped by the approach they relate to.

## Competition Pipeline

### Mahajan (2026) — CB+XGB Residual RF

- **Author:** Divye Mahajan (dmahajanbe23)
- **Title:** CB&XGB|Residual RF|Feature Growth
- **URL:** https://www.kaggle.com/code/dmahajanbe23/cb-xgb-residual-random-forest-feature-growth
- **Used in:** `src/training_stacked.py`, `src/features/top_pipe.py`
- **What we adapted:** The rank-blend + RF residual correction architecture and the correlation-based feature interaction strategy.

### Omid Baghchehsaraei (2026) — RealMLP

- **Author:** Omid Baghchehsaraei
- **Title:** The Best Solo Model So Far: RealMLP (LB 0.95397)
- **URL:** https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397
- **Used in:** `src/models/tabular_dl.py`
- **What we adapted:** RealMLP model configuration and categorical embedding strategy for the heart disease dataset.

### Caruana et al. (2004) — Ensemble Selection

- **Paper:** Ensemble selection from libraries of models (ICML 2004)
- **Used in:** `src/ensemble/blend.py` (hill-climbing forward selection)

### Hollmann et al. (2023) — TabPFN

- **Paper:** TabPFN: A transformer that solves small tabular classification problems in a second (ICLR 2023)
- **Used in:** `src/models/tabular_dl.py`, `src/models/registry.py`

### Holzmüller et al. (2024) — RealMLP

- **Paper:** Better by default: Strong pre-tuned MLPs and boosted trees on tabular data (NeurIPS 2024)
- **Used in:** `src/models/tabular_dl.py`

## Self-Supervised Pretraining

### Bahri et al. (2022) — SCARF

- **Paper:** SCARF: Self-supervised contrastive learning using random feature corruption (ICLR 2022)
- **Used in:** `experiments/ssl/models.py` (SCARF architecture), `experiments/ssl/pretrain.py`

### Vincent et al. (2008) — Denoising Autoencoders / He et al. (2022) — Masked Autoencoders

- **Papers:** Extracting and composing robust features with denoising autoencoders (ICML 2008); Masked autoencoders are scalable vision learners (CVPR 2022)
- **Used in:** `experiments/ssl/models.py`, `experiments/ssl/pretrain.py`
- **Influence:** Our tabular MAE adapts the mask-and-reconstruct principle from Vincent et al. (2008) to tabular features. He et al. (2022) further popularized masked reconstruction as a pretraining objective. SemiMAE extends this by including the target label as a 14th input dimension.

## Tabular-to-Image

### Sharma et al. (2019) — DeepInsight

- **Paper:** DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture (Scientific Reports, 2019)
- **Influence:** Motivated the tabular-to-image projection approach in `experiments/tabular_to_image_raw13/`.

### Chen et al. (2020) — SimCLR

- **Paper:** A simple framework for contrastive learning of visual representations (ICML 2020)
- **Used in:** `experiments/tabular_to_image_raw13/training/contrastive.py`

### He et al. (2020) — MoCo

- **Paper:** Momentum contrast for unsupervised visual representation learning (CVPR 2020)
- **Used in:** `experiments/tabular_to_image_raw13/training/contrastive.py`

### Kingma & Welling (2014) — VAE

- **Paper:** Auto-encoding variational Bayes (ICLR 2014)
- **Used in:** `experiments/tabular_to_image_raw13/models/vae_model.py`, `experiments/tabular_to_image_raw13/training/generative.py`

### Odena et al. (2017) — AC-GAN

- **Paper:** Conditional image synthesis with auxiliary classifier GANs (ICML 2017)
- **Used in:** `experiments/tabular_to_image_raw13/models/discriminator.py`, `experiments/tabular_to_image_raw13/training/generative.py`

### Dosovitskiy et al. (2021) — ViT

- **Paper:** An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021)
- **Used in:** `experiments/tabular_to_image_raw13/models/vit.py`
