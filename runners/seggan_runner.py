# python3.7
"""Contains the runner for StyleGAN."""

from copy import deepcopy
from .stylegan_runner import StyleGANRunner

__all__ = ['SegGANRunner']

class SegGANRunner(StyleGANRunner):
    """Defines the runner for SegGAN."""

    def __init__(self, config, logger):
        super().__init__(config, logger)

    def build_models(self):
        super().build_models()
        self.models['segmentator'].requires_grad_(False)