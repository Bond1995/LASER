from abc import ABC, abstractmethod
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, NamedTuple, Union
from itertools import compress

import torch
import torch.nn.functional as F
import numpy as np

class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass

class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    power: float = 0 # power constraint
    exponent: float = 0 # exponent for power allocation (e = 0 corresponds to uniform allocation)


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: list[torch.Tensor], config: Config):
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]
        self.compressed_params, self.uncompressed_params = self._split(params)
        self.params_per_shape = self._matrices_per_shape(self.compressed_params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

    def aggregate(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:

        compressed_grads, uncompressed_grads = self._split(gradients)
        compressed_out, uncompressed_out = self._compress(compressed_grads, uncompressed_grads)

        # Increment the step counter
        self.step_counter += 1

        return self._merge(compressed_out, uncompressed_out)

    def _compress(self, compressed_grads: list[torch.Tensor], uncompressed_grads: list[torch.Tensor]):
        # Allocate memory for the return value of this function
        compressed_out = [torch.empty_like(g) for g in compressed_grads]

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(compressed_grads)
        outputs_per_shape = self._matrices_per_shape(compressed_out)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])),
                    in_batch,
                    out=out_batch,
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, batch_transpose(out_batch), alpha=-1
                )

            # Power allocation
            compressed_norm = sum([torch.linalg.norm(out_batch, dim=(1,2)).sum() for out_batch in out_batches])
            uncompressed_norm = sum([torch.linalg.norm(grad) for grad in uncompressed_grads])
            total_norm = compressed_norm + uncompressed_norm

            # Noise
            if self.config.power > 0:
                for out_batch in out_batches:
                    for example in out_batch: # set of self.config.rank vectors
                        grad_norm = torch.linalg.norm(example)
                        grad_power = self.config.power * grad_norm / total_norm
                        K = torch.nn.functional.normalize(torch.pow(torch.linalg.norm(example, dim=0), self.config.exponent), p=1, dim=0)
                        P = grad_power * K
                        example.add_(torch.randn_like(example) * (torch.linalg.norm(example, dim=0) / torch.sqrt(P)))

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, batch_transpose(out_batch), alpha=1
                )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Add noise to uncompressed gradients
        if len(uncompressed_grads) == 0:
            uncompressed_out = []
        buffer, shapes = pack(uncompressed_grads)
        uncompressed_out = unpack(buffer, shapes)
        for g in uncompressed_grads:
            g.zero_()
        for o in uncompressed_out:
            p = self.config.power * torch.linalg.norm(o) / total_norm 
            o.add_(
                torch.randn_like(o),
                alpha=torch.linalg.norm(o) / torch.sqrt(p)
            )
        
        return compressed_out, uncompressed_out

    def update(self, compressed_grads, uncompressed_grads):
        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(compressed_grads)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                grad_batch=torch.stack(matrices),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])),
                    in_batch,
                    out=out_batch,
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, batch_transpose(out_batch), alpha=-1
                )
        
        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for m, mb in zip(
                group["grads"],
                group["grad_batch"],
            ):
                m.copy_(mb)
        
        # Power allocation
        compressed_norm = sum([torch.linalg.norm(out_batch, dim=(1,2)).sum() for out_batch in out_batches])
        uncompressed_norm = sum([torch.linalg.norm(grad) for grad in uncompressed_grads])
        total_norm = compressed_norm + uncompressed_norm
        compressed_power = []
        for batch in out_batches:
            compressed_power.append(torch.linalg.norm(batch, dim=(1,2)).to(self.device) / total_norm)
        uncompressed_power = torch.Tensor([torch.linalg.norm(grad) for grad in uncompressed_grads]).to(self.device) / total_norm
        
        return self._ps, self._qs, compressed_power, uncompressed_power

    def set_grads(self, params, ps, qs, uncompressed_grads):
        self.set_pq(ps, qs)
        compressed_params, uncompressed_params = self._split(params)
        compressed_grads = [p.grad.data for p in compressed_params]
        compressed_out = [torch.empty_like(g) for g in compressed_grads]
        gradients_per_shape = self._matrices_per_shape(compressed_grads)
        outputs_per_shape = self._matrices_per_shape(compressed_out)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        iter_is_even = self.step_counter * num_iters_per_step % 2 == 0
        if iter_is_even:
            maybe_transpose = lambda g: g
            out_batches, in_batches = self._qs, self._ps
            out_buffer = self._qs_buffer
        else:
            maybe_transpose = batch_transpose
            out_batches, in_batches = self._ps, self._qs
            out_buffer = self._ps_buffer

        # Construct low-rank reconstruction and update the approximation and error buffer
        for group, in_batch, out_batch in zip(
            shape_groups, in_batches, out_batches
        ):
            maybe_transpose(group["approximation"]).baddbmm_(
                in_batch, batch_transpose(out_batch), alpha=1
            )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, approx in zip(
                group["outputs"],
                group["approximation"]
            ):
                o.copy_(approx)
        
        for (p, g) in zip(compressed_params, compressed_out):
            p.grad = g
        for (p, g) in zip(uncompressed_params, uncompressed_grads):
            p.grad = g

        # Increment the step counter
        self.step_counter += 1

    def _split(self, params: list[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: list[torch.Tensor], uncompressed: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )

    def _init_p_batch(
        self, shape: torch.Size, params: list[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: list[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    def get_uncompressed_grads(self, grads):
        compressed, uncompressed = self._split(grads)
        return uncompressed
    
    def get_pq(self):
        return self._ps, self._qs
    
    def set_pq(self, ps, qs):
        for p1, p2, q1, q2 in zip(self._ps, ps, self._qs, qs):
            p1.copy_(p2)
            q1.copy_(q2)
    
    def get_counter(self):
        return self.step_counter

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: list[torch.Tensor],
    ) -> dict[torch.Size, list[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats

def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])

def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)

def avg_compressed_size(shape: torch.Size, config: Config) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)

def orthogonalize(matrix: torch.Tensor, eps=torch.tensor(1e-16)):
    if matrix.shape[-1] == 1:
        matrix.div_(torch.maximum(matrix.norm(), eps))
    else:
        matrix.copy_(torch.linalg.qr(matrix).Q)

def pack(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Size]]:
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes

def unpack(buffer: torch.Tensor, shapes: list[torch.Size]) -> list[torch.Tensor]:
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries

def params_in_optimizer(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params

def flatten(tensors: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out

def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = params_in_optimizer(optimizer)
    grads = [p.grad.data for p in params]  # type: ignore
    avg_grads = aggregator.aggregate(grads)  # subtracts the approximation from grads

    # Temporarily set parameter's gradients to the aggregated values
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, grads):
        p.grad = g