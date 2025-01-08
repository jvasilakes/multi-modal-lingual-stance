from typing import Optional

import torch
from qwen_vl_utils.vision_process import fetch_image
from captum.attr import InterpretableInput, TextTokenInput


class VisionInput(InterpretableInput):

    def __init__(self, inputs):
        self.inputs = inputs

    def to_tensor(self) -> torch.Tensor:
        return torch.ones_like(self.inputs["pixel_values"])

    def to_model_input(self, perturbed_tensor: Optional[torch.Tensor] = None) -> str:
        return self.inputs
