# DeepFocus
Deep learning based focus and stigmation correction for electron microscopes.

## Installation
To install the `deepfocus` package, run the following in an existing python environment:
```
pip install -r requirements.txt
pip install .
```

## Example
After installing the dependencies and the package you can run `utils/infer.py` to apply the pretrained model file to the
example data:
```
python utils/infer.py
```
which results in:
```
Predicted `working distance` correction 13.03 +- 0.37 (mean +- s.d.) for a known aberration of -13.324 um.
Predicted `stigmator x` correction 0.32 +- 0.09 (mean +- s.d.) for a known aberration of 0.466 (a.u.).
Predicted `stigmator y` correction -0.15 +- 0.04 (mean +- s.d.) for a known aberration of 0.219 (a.u.).
```
The pretrained model is the base model used in the publication with two 512x512 input patches (symmetric perturbation of
5 um). The example data is part of the training/validation data.