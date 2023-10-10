# DeepFocus
Deep learning based focus and stigmation correction for electron microscopes.

## Installation
To install the `deepfocus` package, run the following (requires `conda`):
```
git clone https://github.com/StructuralNeurobiologyLab/DeepFocus
cd DeepFocus
conda env create -f environment.yml -n deepfocus
conda activate deepfocus
pip install -e .
```

## Inference example
After installing the dependencies and the package you can run `utils/infer.py` to apply the pretrained model to the
example data:
```
python scripts/infer_examples.py
```
which results in:
```
Predicted `working distance` correction 13.03 +- 0.37 (mean +- s.d.) for a known aberration of -13.324 um.
Predicted `stigmator x` correction 0.32 +- 0.09 (mean +- s.d.) for a known aberration of 0.466 (a.u.).
Predicted `stigmator y` correction -0.15 +- 0.04 (mean +- s.d.) for a known aberration of 0.219 (a.u.).
```
The pretrained model is the baseline model used in the publication with two 512x512 input patches (symmetric perturbation of
5 µm). The example data is part of the training/validation data.

## Training example
For trainings, we provide example scripts for the baseline model and the EfficientNet (`{_efficientnet}.py` suffix).
To start a training run the following command:
```
python scripts/train_deepfocus.py
```
The default root directory for the training data is `data_root='~/DeepFocus/GT/'` and for the training results
`save_root='~/DeepFocus/trainings/'`. Adjust the paths in the script(s) accordingly.

## Team
The DeepFocus project was developed at the Max Planck Institute for Biological Intelligence in Martinsried by Philipp
Schubert under the supervision of Joergen Kornfeld. We would like to thank Rangoli Saxena for supporting us with MAPFoSt.

## How to cite DeepFocus

```
 @ARTICLE{DeepFocus,
   title     = "DeepFocus: Fast focus and astigmatism correction for electron microscopy",
   author    = "Philipp Johannes Schubert, Rangoli Saxena, Joergen Kornfeld",
   abstract  = "High-throughput 2D and 3D scanning electron microscopy, which relies on automation and dependable control algorithms, requires high image quality with minimal human intervention. Classical focus and astigmatism correction algorithms attempt to explicitly model image formation and subsequently aberration correction. Such models often require parameter adjustments by experts when deployed to new microscopes, challenging samples, or imaging conditions to prevent unstable convergence, making them hard to use in practice or unreliable. Here, we introduce DeepFocus, a purely data-driven method for aberration correction in scanning electron microscopy. DeepFocus works under very low signal-to-noise ratio conditions, reduces processing times by more than an order of magnitude compared to the state-of-the-art method, rapidly converges within a large aberration range, and is easily recalibrated to different microscopes or challenging samples.",
   journal   = "arXiv",
   publisher = "",
   year      = 2023,
   month     = May,
   day       = 8,
   url       =  https://doi.org/10.48550/arXiv.2305.04977
 }

