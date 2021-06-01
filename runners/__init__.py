# python3.7
"""Collects all runners."""

from .stylegan_runner import StyleGANRunner
from .base_gan_runner import BaseGANRunner

__all__ = ['StyleGANRunner', 'BaseGANRunner']
