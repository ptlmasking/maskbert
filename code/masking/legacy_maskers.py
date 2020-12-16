import torch
import torch.nn as nn
import torch.nn.functional as F
import types


class _Binarizer(torch.autograd.Function):
    # NOTE the bug here when using .abs()
    @staticmethod
    def forward(ctx, inputs, thre):
        outputs = inputs.clone()
        inputs = inputs.abs()
        outputs[inputs.le(thre)] = 0.0
        outputs[inputs.gt(thre)] = 1.0
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None)


def _thred_linear(self, x):
    M_w = self.thre_fn(self.weight_mask, self.thre)
    if hasattr(self, "bias_mask"):
        M_b = self.thre_fn(self.bias_mask, self.thre)
        return F.linear(x, self.weight * M_w, self.bias * M_b)
    return F.linear(x, self.weight * M_w)


class OverrideMasker(object):
    def __init__(self, threshold, init_scale, log_fn, mask_biases):
        self.threshold = torch.tensor(threshold)
        self.init_scale = init_scale
        self.mask_biases = mask_biases
        self.log_fn = log_fn

    def patch_modules(self, model, names_tobe_masked, name_of_masker=None):
        for m_name, m in model.named_modules():
            if m_name in names_tobe_masked:
                self.patch_linear(m)
                self.log_fn("\t {} is MASKED".format(m_name))

    def patch_linear(self, module):
        assert isinstance(module, nn.Linear)
        for param in module.parameters():
            assert not param.requires_grad
        module.weight_mask = nn.Parameter(
            torch.empty_like(module.weight).uniform_(
                -1 * self.init_scale, self.init_scale
            )
        )
        if module.bias is not None and self.mask_biases:
            module.bias_mask = nn.Parameter(
                torch.empty_like(module.bias).uniform_(
                    -1 * self.init_scale, self.init_scale
                )
            )
        module.thre_fn = _Binarizer().apply
        module.thre = nn.Parameter(self.threshold.clone(), requires_grad=False)
        module.forward = types.MethodType(_thred_linear, module)
