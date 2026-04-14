from __future__ import annotations

import torch

from .backend_module import normalize_device_index
from .tree_utils import tree_map
from .tensor_wrapper import wrap_tensor, unwrap_tensor

_FGPU_LIB = torch.library.Library("_", "IMPL")


def _wrap_result(obj):
    if isinstance(obj, torch.Tensor):
        return wrap_tensor(obj)
    return obj


def _unwrap_arg(obj):
    return unwrap_tensor(obj)


def fgpu_fallback(keyset, op, *args, **kwargs):
    cpu_args = tree_map(_unwrap_arg, args)
    cpu_kwargs = tree_map(_unwrap_arg, kwargs)
    cpu_keyset = keyset.remove(torch._C.DispatchKey.PrivateUse1)
    result = op.redispatch(cpu_keyset, *cpu_args, **cpu_kwargs)
    return tree_map(_wrap_result, result)


_FGPU_LIB.fallback(fgpu_fallback, "PrivateUse1", with_keyset=True)


@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    device_index = normalize_device_index(device, optional=True)
    return wrap_tensor(torch.empty(size, dtype=dtype or torch.float32), device_index=device_index)


@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    device_index = normalize_device_index(device, optional=True)
    return wrap_tensor(
        torch.empty_strided(size, stride, dtype=dtype or torch.float32),
        device_index=device_index,
    )


@torch.library.impl("aten::_copy_from", "privateuseone")
def copy_from(src, dst, non_blocking=False):
    dst.raw_data = unwrap_tensor(src).clone()
    return dst


@torch.library.impl("aten::add.out", "privateuseone")
def add_out(a, b, *, alpha=1, out):
    out.raw_data = unwrap_tensor(a) + unwrap_tensor(b) * alpha
    return out


@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    return wrap_tensor(
        torch.ops.aten.convolution.default(
            unwrap_tensor(input),
            unwrap_tensor(weight),
            unwrap_tensor(bias),
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    )


@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    bias_sizes = [unwrap_tensor(weight).shape[0]] if output_mask[2] else None
    result = torch.ops.aten.convolution_backward.default(
        unwrap_tensor(grad_output),
        unwrap_tensor(input),
        unwrap_tensor(weight),
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    return tree_map(_wrap_result, result)


@torch.library.impl("aten::native_batch_norm", "privateuseone")
def native_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
):
    result = torch.ops.aten.native_batch_norm.default(
        unwrap_tensor(input),
        unwrap_tensor(weight),
        unwrap_tensor(bias),
        unwrap_tensor(running_mean),
        unwrap_tensor(running_var),
        training,
        momentum,
        eps,
    )
    return tree_map(_wrap_result, result)


@torch.library.impl("aten::native_batch_norm_backward", "privateuseone")
def native_batch_norm_backward(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    train,
    eps,
    output_mask,
):
    result = torch.ops.aten.native_batch_norm_backward.default(
        unwrap_tensor(grad_out),
        unwrap_tensor(input),
        unwrap_tensor(weight),
        unwrap_tensor(running_mean),
        unwrap_tensor(running_var),
        unwrap_tensor(save_mean),
        unwrap_tensor(save_invstd),
        train,
        eps,
        output_mask,
    )
    return tree_map(_wrap_result, result)


@torch.library.impl("aten::relu", "privateuseone")
def relu(input):
    return wrap_tensor(torch.ops.aten.relu.default(unwrap_tensor(input)))


@torch.library.impl("aten::threshold_backward", "privateuseone")
def threshold_backward(grad_output, self, threshold):
    return wrap_tensor(
        torch.ops.aten.threshold_backward.default(
            unwrap_tensor(grad_output),
            unwrap_tensor(self),
            threshold,
        )
    )


@torch.library.impl("aten::mean.dim", "privateuseone")
def mean_dim(self, dim, keepdim=False, *, dtype=None):
    return wrap_tensor(
        torch.ops.aten.mean.dim(
            unwrap_tensor(self),
            dim,
            keepdim,
            dtype=dtype,
        )
    )


@torch.library.impl("aten::sum.dim_IntList", "privateuseone")
def sum_dim_intlist(self, dim, keepdim=False, *, dtype=None):
    return wrap_tensor(
        torch.ops.aten.sum.dim_IntList(
            unwrap_tensor(self),
            dim,
            keepdim,
            dtype=dtype,
        )
    )


@torch.library.impl("aten::addmm", "privateuseone")
def addmm(self, mat1, mat2, *, beta=1, alpha=1):
    return wrap_tensor(
        torch.ops.aten.addmm.default(
            unwrap_tensor(self),
            unwrap_tensor(mat1),
            unwrap_tensor(mat2),
            beta=beta,
            alpha=alpha,
        )
    )


@torch.library.impl("aten::mm", "privateuseone")
def mm(self, mat2):
    return wrap_tensor(torch.ops.aten.mm.default(unwrap_tensor(self), unwrap_tensor(mat2)))


@torch.library.impl("aten::expand", "privateuseone")
def expand(self, size, *, implicit=False):
    return wrap_tensor(torch.ops.aten.expand.default(unwrap_tensor(self), size, implicit=implicit))


@torch.library.impl("aten::t", "privateuseone")
def t(self):
    return wrap_tensor(torch.ops.aten.t.default(unwrap_tensor(self)))


@torch.library.impl("aten::_log_softmax", "privateuseone")
def log_softmax(self, dim, half_to_float):
    return wrap_tensor(torch.ops.aten._log_softmax.default(unwrap_tensor(self), dim, half_to_float))


@torch.library.impl("aten::_log_softmax_backward_data", "privateuseone")
def log_softmax_backward_data(grad_output, output, dim, input_dtype):
    return wrap_tensor(
        torch.ops.aten._log_softmax_backward_data.default(
            unwrap_tensor(grad_output),
            unwrap_tensor(output),
            dim,
            input_dtype,
        )
    )


@torch.library.impl("aten::nll_loss_forward", "privateuseone")
def nll_loss_forward(self, target, weight, reduction, ignore_index):
    result = torch.ops.aten.nll_loss_forward.default(
        unwrap_tensor(self),
        unwrap_tensor(target),
        unwrap_tensor(weight),
        reduction,
        ignore_index,
    )
    return tree_map(_wrap_result, result)


@torch.library.impl("aten::nll_loss_backward", "privateuseone")
def nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight):
    return wrap_tensor(
        torch.ops.aten.nll_loss_backward.default(
            unwrap_tensor(grad_output),
            unwrap_tensor(self),
            unwrap_tensor(target),
            unwrap_tensor(weight),
            reduction,
            ignore_index,
            unwrap_tensor(total_weight),
        )
    )


@torch.library.impl("aten::zeros_like", "privateuseone")
def zeros_like(self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return wrap_tensor(torch.ops.aten.zeros_like.default(unwrap_tensor(self), dtype=dtype, layout=layout, device="cpu", pin_memory=False, memory_format=memory_format))


@torch.library.impl("aten::ones_like", "privateuseone")
def ones_like(self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return wrap_tensor(torch.ops.aten.ones_like.default(unwrap_tensor(self), dtype=dtype, layout=layout, device="cpu", pin_memory=False, memory_format=memory_format))


@torch.library.impl("aten::sqrt", "privateuseone")
def sqrt(self):
    return wrap_tensor(torch.ops.aten.sqrt.default(unwrap_tensor(self)))


@torch.library.impl("aten::div.Tensor", "privateuseone")
def div_tensor(self, other):
    return wrap_tensor(torch.ops.aten.div.Tensor(unwrap_tensor(self), unwrap_tensor(other)))


@torch.library.impl("aten::div.Scalar", "privateuseone")
def div_scalar(self, other):
    return wrap_tensor(torch.ops.aten.div.Scalar(unwrap_tensor(self), other))


@torch.library.impl("aten::add_.Tensor", "privateuseone")
def add__tensor(self, other, *, alpha=1):
    self.raw_data = torch.ops.aten.add_.Tensor(unwrap_tensor(self), unwrap_tensor(other), alpha=alpha)
    return self


@torch.library.impl("aten::mul_.Tensor", "privateuseone")
def mul__tensor(self, other):
    self.raw_data = torch.ops.aten.mul_.Tensor(unwrap_tensor(self), unwrap_tensor(other))
    return self


@torch.library.impl("aten::addcmul_", "privateuseone")
def addcmul_(self, tensor1, tensor2, *, value=1):
    self.raw_data = torch.ops.aten.addcmul_.default(
        unwrap_tensor(self),
        unwrap_tensor(tensor1),
        unwrap_tensor(tensor2),
        value=value,
    )
    return self


@torch.library.impl("aten::addcdiv_", "privateuseone")
def addcdiv_(self, tensor1, tensor2, *, value=1):
    self.raw_data = torch.ops.aten.addcdiv_.default(
        unwrap_tensor(self),
        unwrap_tensor(tensor1),
        unwrap_tensor(tensor2),
        value=value,
    )
    return self


@torch.library.impl("aten::lerp_.Scalar", "privateuseone")
def lerp__scalar(self, end, weight):
    self.raw_data = torch.ops.aten.lerp_.Scalar(unwrap_tensor(self), unwrap_tensor(end), weight)
    return self


@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def local_scalar_dense(self):
    return torch.ops.aten._local_scalar_dense.default(unwrap_tensor(self))
