# python3.7
"""Collects all available models together."""

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator

import os
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator',
    'build_generator', 'build_discriminator', 'build_model', 'parse_gan_type'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2']
_MODULES_ALLOWED = ['generator', 'discriminator', 'segmentator', 'segmentation_discriminator']


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')

def build_segmentator(resolution, config_path, **kwargs):

    cfg.merge_from_file(config_path)

    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.num_epoch))
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.num_epoch))
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), \
            "cannot find segmentation models: checkpoint does not exist!"

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, None, fixed=True)
    return segmentation_module

def build_segmentation_discriminator(gan_type, resolution, **kwargs):
    
    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')

def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    if module == 'segmentator':
        return build_segmentator(resolution, **kwargs)
    if module == 'segmentation_discriminator':
        return build_segmentation_discriminator(gan_type, resolution, **kwargs)

    raise NotImplementedError(f'Unsupported module `{module}`!')


def parse_gan_type(module):
    """Parses GAN type of a given module.

    Args:
        module: The module to parse GAN type from.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    if isinstance(module, (PGGANGenerator, PGGANDiscriminator)):
        return 'pggan'
    if isinstance(module, (StyleGANGenerator, StyleGANDiscriminator)):
        return 'stylegan'
    if isinstance(module, (StyleGAN2Generator, StyleGAN2Discriminator)):
        return 'stylegan2'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')
