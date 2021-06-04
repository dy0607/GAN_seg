# python3.7
"""Collects all runners."""

from .stylegan_runner import StyleGANRunner
from .base_gan_runner import BaseGANRunner
from .seggan_runner import SegGANRunner

__all__ = ['StyleGANRunner', 'BaseGANRunner', 'SegGANRunner']
