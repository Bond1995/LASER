import torch


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