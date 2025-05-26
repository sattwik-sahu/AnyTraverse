from abc import ABC, abstractmethod
from typing import NamedTuple
from PIL.Image import Image
import torch
from anytraverse.utils.pipelines.base import (
    Pipeline2 as Pipeline,
    create_pipeline,
)


__all__ = ["Pipeline", "create_pipeline"]



