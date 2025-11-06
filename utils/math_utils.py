import torch
import torch.nn.functional as F
import sys

from triton.language import zeros_like

sys.path.append(sys.path[0] + '/..')

def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def make_arange(shape: torch.Size, dim: int = 1):
    arange_shape = [1] * len(shape)
    arange_shape[dim] = shape[dim]
    arange = torch.arange(shape[dim]).reshape(arange_shape)
    arange = arange.expand(shape)
    return arange

class OneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, length: int):
        assert input.le(length).all()
        one_hot = F.one_hot(input.to(torch.long), length)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(one_hot)
        return one_hot.float()

    @staticmethod
    def backward(ctx, grad_output):
        one_hot = ctx.saved_tensors[0].to(ctx._input_dtype)
        masked = one_hot @ grad_output.transpose(-1, -2)
        grad_input = masked.diagonal(dim1 = -2, dim2 = -1)
        return grad_input, None

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim: int = 1, keepdim = False):
        idx = torch.argmax(input, dim = dim, keepdim = keepdim)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx._dim = dim
        ctx._keepdim = keepdim
        ctx.save_for_backward(idx)
        return idx.float()

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        grad_input = torch.zeros(ctx._input_shape, device = ctx._input_device, dtype = ctx._input_dtype)

        if not ctx._keepdim:
            idx = idx.unsqueeze(ctx._dim)
            grad_output = grad_output.unsqueeze(ctx._dim)
        grad_input.scatter_(ctx._dim, idx, grad_output)
        return grad_input, None, None