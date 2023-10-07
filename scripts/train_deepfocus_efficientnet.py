import argparse
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision.models import efficientnet_b0

from deepfocus.af_utils import RandomCrop
from deepfocus.dataloader import DeepFocusData


class EfficientNetAdapterLateFuse(nn.Module):
    def __init__(self, in_channels, out_channels, num_patches, act='relu'):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        else:
            raise NotImplemented()
        if 64 % (in_channels * num_patches) != 0:
            raise ValueError('Number of input channels times the number of patches must be a multiple of 64.')
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.efficientnet_backbone = efficientnet_b0(
            weights='EfficientNet_B0_Weights.DEFAULT'
        )

        self.conv_final = nn.Sequential(
            nn.Conv1d(num_patches * 1280, 320, kernel_size=(1,)),
            act,
            nn.Conv1d(320, out_channels, kernel_size=(1,)),
        )

    def forward(self, x_stacked: torch.Tensor):  # shape B C D H W.
        # Stack input images on channel axis.
        processed_images = []
        for image_index in range(x_stacked.size()[2]):  # Images are stacked along Depth (D) axis.
            x = x_stacked[:, :, image_index]  # shape B C H W.
            x = self.input_adapter(x)
            x = self.efficientnet_backbone.features(x)
            x = self.efficientnet_backbone.avgpool(x)
            x = torch.flatten(x, 1).unsqueeze(2)  # B C W, add spatial axis
            processed_images.append(x)
        x = self.conv_final(torch.cat(processed_images, 1))
        return x.squeeze(-1)  # B C


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
    parser.add_argument(
        '-s', '--epoch-size', type=int, default=800,
        help='How many training samples to process between '
             'validation/preview/extended-stat calculation phases.'
    )
    parser.add_argument(
        '-m', '--max-steps', type=int, default=1000000,
        help='Maximum number of training steps to perform.'
    )
    parser.add_argument(
        '-t', '--max-runtime', type=int, default=3600 * 24 * 16,  # 4 days
        help='Maximum training time (in seconds).'
    )
    parser.add_argument(
        '-r', '--resume', metavar='PATH',
        help='Path to pretrained model state dict or a compiled and saved '
             'ScriptModule from which to resume training.'
    )
    parser.add_argument(
        '-j', '--jit', metavar='MODE', default='onsave',
        choices=['disabled', 'train', 'onsave'],
        help="""Options:
    "disabled": Completely disable JIT (TorchScript) compilation;
    "onsave": Use regular Python model for training, but JIT-compile it on-demand for saving training state;
    "train": Use JIT-compiled model for training and serialize it on disk."""
    )
    parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Run in fully deterministic mode (at the cost of execution speed).'
    )
    parser.add_argument('-i', '--ipython', action='store_true',
                        help='Drop into IPython shell on errors or keyboard interrupts.'
                        )
    args = parser.parse_args()

    # Set up all RNG seeds, set level of determinism
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    deterministic = args.deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

    # Don't move this stuff, it needs to be run this early to work
    import elektronn3

    elektronn3.select_mpl_backend('Agg')
    logger = logging.getLogger('elektronn3log')
    # Write the flags passed to python via argument passer to logfile
    # They will appear as "Namespace(arg1=val1, arg2=val2, ...)" at the top of the logfile
    logger.debug("Arguments given to python via flags: {}".format(args))

    from elektronn3.data import transforms
    from elektronn3.training import Trainer
    from elektronn3.training import Backup

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    logger.info(f'Running on device: {device}')

    # You can store selected hyperparams in a dict for logging to tensorboard, e.g.
    hparams = {}
    perturbation_list = [5000, ]
    only_defoc = False
    magnitude_only_target = False
    in_channels = 1
    out_channels = 1 if only_defoc else 3
    patch_resolution = 512
    model = EfficientNetAdapterLateFuse(in_channels=in_channels, out_channels=out_channels,
                                        num_patches=len(perturbation_list) * 2)
    model = model.to(device)
    # Example for a model-compatible input.
    example_input = torch.ones(4, in_channels, len(perturbation_list) * 2, patch_resolution, patch_resolution).to(
        device)
    model(example_input)
    save_jit = None if args.jit == 'disabled' else 'script'
    if args.jit == 'onsave':
        # Make sure that compilation works at all
        jitmodel = torch.jit.script(model)
    elif args.jit == 'train':
        jitmodel = torch.jit.script(model)
        model = jitmodel
    save_root = os.path.expanduser(os.path.expanduser('~/DeepFocus/trainings/'))
    data_root = os.path.expanduser(os.path.expanduser('~/DeepFocus/GT/*'))
    os.makedirs(save_root, exist_ok=True)
    max_steps = args.max_steps
    max_runtime = args.max_runtime

    optimizer_state_dict = None
    lr_sched_state_dict = None
    if args.resume is not None:  # Load pretrained network
        pretrained = os.path.expanduser(args.resume)
        logger.info(f'Loading model from {pretrained}')
        if pretrained.endswith('.pt'):  # nn.Module
            model = torch.load(pretrained, map_location=device)
        elif pretrained.endswith('.pts'):  # ScriptModule
            model = torch.jit.load(pretrained, map_location=device)
        elif pretrained.endswith('.pth'):
            state = torch.load(pretrained)
            model.load_state_dict(state['model_state_dict'], strict=False)
            optimizer_state_dict = state.get('optimizer_state_dict')
            lr_sched_state_dict = state.get('lr_sched_state_dict')
            if optimizer_state_dict is None:
                logger.warning('optimizer_state_dict not found.')
            if lr_sched_state_dict is None:
                logger.warning('lr_sched_state_dict not found.')
        else:
            raise ValueError(f'{pretrained} has an unknown file extension. Supported are: .pt, .pts and .pth')

    # Transformations to be applied to samples before feeding them to the network
    common_transforms = [
        RandomCrop((len(perturbation_list) * 2, patch_resolution, patch_resolution), independent_xy=False,
                   deterministic=False),  # Crop before
        # augmentations to save compute
        transforms.Normalize(mean=128, std=128),
    ]
    valid_transforms = transforms.Compose(common_transforms)
    train_transforms = transforms.Compose(common_transforms + [
        transforms.AdditiveGaussianNoise(prob=0.75, sigma=0.2),
        transforms.RandomGammaCorrection(prob=0.75, gamma_std=0.25),
        transforms.RandomBrightnessContrast(prob=0.75, brightness_std=0.25, contrast_std=0.25)
    ])

    train_dataset = DeepFocusData(data_root, focus_perturbations=perturbation_list, train=True, only_defoc=only_defoc,
                                  transform=train_transforms, magnitude_only_target=magnitude_only_target,
                                  n_entries_per_sample=12)

    # use same data but only common augmentation
    valid_dataset = DeepFocusData(data_root, focus_perturbations=perturbation_list, train=False, only_defoc=only_defoc,
                                  transform=valid_transforms, magnitude_only_target=magnitude_only_target,
                                  n_entries_per_sample=12)
    logger.info(f'Using {train_dataset.n_samples} training samples and {valid_dataset.n_samples} validation samples.')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Learning rate is set by the lr_sched below
        weight_decay=0.5e-4,
    )
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, 2000, 0.99)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if lr_sched_state_dict is not None:
        lr_sched.load_state_dict(lr_sched_state_dict)

    # Validation metrics
    valid_metrics = {}
    for evaluator in []:
        valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
        for c in range(out_channels):
            valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)

    criterion = nn.L1Loss()

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=4,
        num_workers=10,
        save_root=save_root,
        exp_name=args.exp_name,
        example_input=example_input,
        save_jit=save_jit,
        schedulers={'lr': lr_sched},
        valid_metrics=valid_metrics,
        hparams=hparams,
        out_channels=out_channels,
        ipython_shell=args.ipython,
        use_custom_collate=False,
    )

    if args.deterministic:
        assert trainer.num_workers <= 1, 'num_workers > 1 introduces indeterministic behavior'

    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps=max_steps, max_runtime=max_runtime)
