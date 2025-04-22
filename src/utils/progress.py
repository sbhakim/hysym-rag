# src/utils/progress.py

import sys
import logging
from tqdm.auto import tqdm as _tqdm


class ProgressManager:
    enabled = False

    @classmethod
    def disable_progress(cls):
        # Disable tqdm
        cls.enabled = False
        # Also disable progress bars from sentence-transformers
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

    @classmethod
    def enable_progress(cls):
        cls.enabled = True
        logging.getLogger('sentence_transformers').setLevel(logging.INFO)


def tqdm(*args, **kwargs):
    """Global wrapper for tqdm that respects global settings"""
    if not ProgressManager.enabled:
        kwargs['disable'] = True
    return _tqdm(*args, **kwargs)