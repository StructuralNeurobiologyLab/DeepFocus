import os

import imageio
import numpy as np
import torch

from deepfocus import DeepFocus

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    print(f'Using torch device `{_DEVICE}`.')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    model_path = os.path.join(data_dir, 'StackedConv2D__21-07-27_16-21-40_normal_512.pts')
    model = DeepFocus(model_path, perturbation_list=[5e-6], normalization=(128., 128.), use_fix_locations=True,
                      crop_edge_length=512, n_consensus=10, device=_DEVICE)
    # Known aberration: Working distance in microns (um), stigx in arbitrary units (a.u.), stigy in a.u.
    known_aberration = ['-13.324 um', '0.466 (a.u.)', '0.219 (a.u.)']
    beam_parameter = ['working distance', 'stigmator x', 'stigmator y']
    # Retrieve test data.
    images = np.array([
        imageio.imread(os.path.join(data_dir, 'series_0_def-13324_astx-466_asty219_pert-5000.png')),
        imageio.imread(os.path.join(data_dir, 'series_0_def-13324_astx-466_asty219_pert5000.png'))],
        dtype='f4'
    )
    # Apply DeepFocus.
    predicted_aberration, prediction_sds = model.apply(images)
    result_summary = zip(beam_parameter, predicted_aberration, prediction_sds, known_aberration)
    for param_type, pred_mean, pred_sd, aberration in result_summary:
        if param_type == 'working distance':  # Convert meters into microns.
            pred_mean *= 1e6
            pred_sd *= 1e6
        print(f'Predicted `{param_type}` correction {pred_mean:.2f} +- {pred_sd:.2f} (mean +- s.d.) for a known '
              f'aberration of {aberration}.')
