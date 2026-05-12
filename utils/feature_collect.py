import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional

class FeatureCollector:
    def __init__(self):
        self.features: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __call__(self, module: nn.Module, input, output: torch.Tensor):
        self.features.append(output.detach().cpu().numpy())

    def register_hook(self, module: nn.Module):
        self.remove_hook()
        self.hook_handle = module.register_forward_hook(self)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def collect_label(self, label_tensor: torch.Tensor):
        self.labels.append(label_tensor.detach().cpu().numpy())

    def clear(self):
        self.features.clear()
        self.labels.clear()

    def get_features(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        return np.concatenate(self.features, axis=0)

    def get_labels(self) -> Optional[np.ndarray]:
        if not self.labels:
            return None
        return np.concatenate(self.labels, axis=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()
        return False