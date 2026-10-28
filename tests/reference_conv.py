#!/usr/bin/env python3
"""Independent PyTorch reference for valid 2-D cross-correlation."""

from __future__ import annotations

from typing import Sequence


def convolution_reference(
    input_values: Sequence[Sequence[float]],
    kernel_values: Sequence[Sequence[float]],
) -> list[list[float]]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for the external-reference correctness tests"
        ) from error

    input_tensor = torch.tensor(input_values, dtype=torch.float64)[None, None]
    kernel_tensor = torch.tensor(kernel_values, dtype=torch.float64)[None, None]
    output = functional.conv2d(input_tensor, kernel_tensor)
    return output[0, 0].tolist()
