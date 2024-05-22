import torch.nn as nn
import torch.autograd as autograd


class ScaleGradient(autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class ScaleGradientModule(nn.Module):
    def __init__(self, scale):
        super(ScaleGradientModule, self).__init__()
        self.scale = scale

    def forward(self, input):
        return ScaleGradient.apply(input, self.scale)