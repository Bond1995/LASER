from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
import numpy as np
from utils import pack, unpack

from powersgd import Aggregator


@dataclass
class Sketch:
    seed: int
    original_shape: torch.Size
    sketch: torch.Tensor


class CountSketchAggregator(Aggregator):
    def __init__(
        self,
        compression_factor: float,
        mode: Literal["all together", "one by one"],
    ):
        assert compression_factor > 0
        assert compression_factor <= 1
        self._compression_factor = compression_factor
        self._generator = torch.Generator().manual_seed(0)
        assert mode in ["all together", "one by one"]
        self._mode = mode

    def aggregate(self, gradients, power):
        if len(gradients) == 0:
            return []
        
        device = gradients[0][0].device
        workers = len(gradients)
        out = [torch.zeros_like(g) for g in gradients[0]]
        mean_sketch = []
        for g in gradients[0]:
            compressed_size = int(self._compression_factor * g.numel())
            if compressed_size == 0:
                compressed_size = 1
            mean_sketch.append(torch.zeros(compressed_size).to(device))  
        shared = [int(torch.randint(1000000000, (1,), generator=self._generator)) for g in gradients[0]]
        power_per_param = torch.zeros(len(gradients[0])).to(device)
        noise_coeffs = torch.zeros(len(gradients[0])).to(device)

        for i in range(workers):
            worker_power = torch.zeros(len(gradients[0])).to(device)
            for j, g in enumerate(gradients[i]):
                compressed_size = int(self._compression_factor * g.numel())
                if compressed_size == 0:
                    compressed_size = 1
                message = sketch(g, compressed_size, shared[j])
                worker_power[j] = torch.linalg.norm(message.sketch)
                # Average gradient
                mean_sketch[j].add_(message.sketch, alpha=1/workers)
                # Error feedback (removed, does not converge)
                # g.add_(unsketch_mean(message), alpha=-1)
            noise_coeffs.copy_(torch.maximum(noise_coeffs, worker_power))
            power_per_param.add_(F.normalize(worker_power, p = 1, dim = 0), alpha=power/workers)

        # Add noise to sketched grads
        for g, p, c in zip(mean_sketch, power_per_param, noise_coeffs):
            g.add_(torch.randn_like(g), alpha=c/(workers*torch.sqrt(p)))
        
        # Reconstruct grads
        for i, g in enumerate(mean_sketch):
            out[i].add_(unsketch_mean(Sketch(seed=shared[i], original_shape=out[i].shape, sketch=g)))

        return out


def sketch(tensor: torch.Tensor, compressed_size: int, seed: int) -> Sketch:
    """Create a Count Sketch with one column."""
    flat_tensor = tensor.view(-1)
    indices, signs = _indices_and_signs(seed, flat_tensor.size(0), compressed_size, flat_tensor.device)
    sketch = torch.zeros(
        compressed_size, dtype=flat_tensor.dtype, device=flat_tensor.device
    )
    sketch.scatter_add_(0, indices, flat_tensor * signs)
    return Sketch(seed=seed, original_shape=tensor.shape, sketch=sketch)


def unsketch_mean(sketch: Sketch) -> torch.Tensor:
    """Count Sketch reconstruction from a single-column sketch."""
    indices, signs = _indices_and_signs(
        sketch.seed, sketch.original_shape.numel(), sketch.sketch.numel(), sketch.sketch.device
    )
    return (sketch.sketch[indices] * signs).view(*sketch.original_shape)


def _indices_and_signs(seed: int, tensor_size: int, compressed_size: int, device):
    """Generate random indices and signs for Count sketching.

    The indices are uniformly distributed in [0, compressed_size), and
    the signs are either +1 or -1."""
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(0, compressed_size, (tensor_size,), generator=generator, device=device)
    signs = torch.randint(0, 2, (tensor_size,), generator=generator, device=device) * 2 - 1
    return indices, signs
