import argparse
import logging
import os
import random

import numpy as np
import torch
from elektronn3.models.simple import Conv3DLayer
from torch import nn
from torch import optim

from deepfocus.af_utils import RandomCrop
from deepfocus.dataloader import DeepFocusData


class StackedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_z, dropout_rate=0.1, act='relu'):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        else:
            raise NotImplemented()
        self.seq = nn.Sequential(
            Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(50, 60, (1, 3, 3), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(60, 70, (1, 3, 3), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
        )
        self.conv_final = nn.Sequential(
            nn.Conv1d(140, 25, kernel_size=(1,)),
            act,
            nn.Conv1d(25, out_channels, kernel_size=(1,)),
        )

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size()[0], -1, 1)  # shape B C D H W -> B C D
        x = self.conv_final(x)  # B C
        return x.squeeze(-1)


class EnsembleDeepFocus(nn.Module):
    """
    Combine `n_consensus` consecutive samples in the batch using weighted average.
    """

    def __init__(self, backbone, n_consensus: int):
        super().__init__()
        self.n_consensus = n_consensus
        self.backbone = backbone

    def forward(self, x):
        # x shape: (B, C, Z, Y, X), e.g. (8, 1, 2, 1024, 1024)
        assert len(x) % self.n_consensus == 0
        # if prediction is wd, stigx, stigy AND additional score (total: 3+1=4)
        orig_out = self.backbone(x)
        # x shape: (B // n_consensus, n_consensus, 4)
        x = orig_out.view(orig_out.size()[0] // self.n_consensus, self.n_consensus, 4)
        # create new score variable to prevent in-place error during backprop
        scores = torch.softmax(x[..., -1:], dim=1)
        # x shape: (B // n_consensus, n_consensus, 3)
        x = torch.sum(x[..., :3] * scores, dim=1) / torch.sum(scores, dim=1)  # weighted mean using last output
        return x, orig_out


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
        '-t', '--max-runtime', type=int, default=3600 * 24 * 4,  # 4 days
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
    prepare_aux_score = 5  # None

    # Don't move this stuff, it needs to be run this early to work
    import elektronn3

    elektronn3.select_mpl_backend('Agg')
    logger = logging.getLogger('elektronn3log')
    # Write the flags passed to python via argument passer to logfile
    # They will appear as "Namespace(arg1=val1, arg2=val2, ...)" at the top of the logfile
    logger.debug("Arguments given to python via flags: {}".format(args))

    from elektronn3.data import transforms

    if prepare_aux_score is not None:
        from elektronn3.training.trainer_deepfocusscore import Trainer
    else:
        from elektronn3.training import Trainer
    from elektronn3.training import Backup

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    logger.info(f'Running on device: {device}')

    # You can store selected hyperparams in a dict for logging to tensorboard, e.g.
    # hparams = {'n_blocks': 4, 'start_filts': 32, 'planar_blocks': (0,)}
    hparams = {}
    perturbation_list = [5000, ]
    only_defoc = False
    magnitude_only_target = False
    in_channels = 1
    out_channels = 1 if only_defoc else 3
    if prepare_aux_score is not None:
        out_channels += 1
    model = StackedConv2D(in_channels=in_channels, out_channels=out_channels, n_z=len(perturbation_list) * 2)
    if prepare_aux_score is not None:
        model = EnsembleDeepFocus(model, n_consensus=prepare_aux_score)
    model = model.to(device)
    # model = StackedConv2DSmall(in_channels=in_channels, out_channels=out_channels).to(device)
    # Example for a model-compatible input.
    example_input = torch.ones(prepare_aux_score * 2 if prepare_aux_score is not None else 2, in_channels,
                               len(perturbation_list) * 2, 384, 384).to(device)
    model(example_input)
    save_jit = None if args.jit == 'disabled' else 'script'
    if args.jit == 'onsave':
        # Make sure that compilation works at all
        jitmodel = torch.jit.script(model)
    elif args.jit == 'train':
        jitmodel = torch.jit.script(model)
        model = jitmodel
    save_root = os.path.expanduser(os.path.expanduser('~/Documents/DeepFocus/trainings_TESTEST/'))
    data_root = os.path.expanduser(os.path.expanduser('~/Documents/DeepFocus/GT/*'))
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
            raise ValueError(f'{pretrained} has an unkown file extension. Supported are: .pt, .pts and .pth')

    # Transformations to be applied to samples before feeding them to the network
    common_transforms = [
        RandomCrop((len(perturbation_list) * 2, 384, 384), independent_xy=False, deterministic=False),  # Crop before
        # augmentations to save compute
        transforms.Normalize(mean=128, std=128),
    ]
    valid_transforms = transforms.Compose(common_transforms)
    train_transforms = transforms.Compose(common_transforms + [
        # RandomMult(-1, p=0.25),
        transforms.AdditiveGaussianNoise(prob=0.75, sigma=0.2),
        transforms.RandomGammaCorrection(prob=0.75, gamma_std=0.25),
        transforms.RandomBrightnessContrast(prob=0.75, brightness_std=0.25, contrast_std=0.25)
    ])

    train_dataset = DeepFocusData(data_root, focus_perturbations=perturbation_list, train=True, only_defoc=only_defoc,
                                  transform=train_transforms, magnitude_only_target=magnitude_only_target,
                                  prepare_aux_score=prepare_aux_score, n_entries_per_sample=12)

    # use same data but only common augmentation
    valid_dataset = DeepFocusData(data_root, focus_perturbations=perturbation_list, train=False, only_defoc=only_defoc,
                                  transform=valid_transforms, magnitude_only_target=magnitude_only_target,
                                  prepare_aux_score=prepare_aux_score, n_entries_per_sample=12)
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
        batch_size=4 if prepare_aux_score is not None else 8,
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
        use_custom_collate=prepare_aux_score is not None,
    )

    if args.deterministic:
        assert trainer.num_workers <= 1, 'num_workers > 1 introduces indeterministic behavior'

    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps=max_steps, max_runtime=max_runtime)
