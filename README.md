# Learning Latent Space Representations - GANs

Implementation of various GAN architectures for improving image generation quality and diversity.

> **📖 Read [report.pdf](report.pdf) for complete details and results.**

## Methods

1. **Static GM-GAN** - Gaussian Mixture prior for better diversity
2. **WGAN-GP** - Wasserstein GAN with Gradient Penalty for stable training  
3. **Conditional WGAN-GP** - Class-conditional generation with learnable latent distributions

## Results

| Method | FID | Precision | Recall |
|--------|-----|-----------|--------|
| Vanilla GAN | 108.1 | 0.56 | 0.09 |
| Static GM-GAN | 35.82 | 0.28 | 0.52 |
| WGAN-GP | 26.90 | 0.42 | 0.59 |
| Conditional WGAN-GP | 15.15 | 0.52 | 0.67 |

See report.pdf for methodology and analysis.

## References

**GM-GAN:**
```
Ben-Yosef & Weinshall. "Gaussian mixture generative adversarial networks 
for diverse datasets." arXiv:1808.10356, 2018.
```

**WGAN-GP:**
```
Gulrajani et al. "Improved training of Wasserstein GANs." NeurIPS 2017.
```

**Metrics:**
```
Heusel et al. "GANs trained by a two time-scale update rule converge 
to a local Nash equilibrium." NeurIPS 2017. (FID)

Kynkäänniemi et al. "Improved precision and recall metric for assessing 
generative models." NeurIPS 2019.
```

## Authors

Paul Martinetti, Mathéo Quatreboeufs, Hannah Gloaguen  
Team: Nano Kiwi  
M2 IASD, Université Paris Dauphine - PSL

## Dataset

MNIST (28×28 grayscale handwritten digits)
