# DeepFocus
Deep learning based focus and stigmation correction for electron microscopes (https://doi.org/10.1038/s41467-024-45042-3).

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
@article{Schubert2024,
  title = {DeepFocus: fast focus and astigmatism correction for electron microscopy},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-024-45042-3},
  DOI = {10.1038/s41467-024-45042-3},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Schubert,  P. J. and Saxena,  R. and Kornfeld,  J.},
  year = {2024},
  month = jan 
}
```
