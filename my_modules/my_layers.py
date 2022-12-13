from torch.autograd import Function
import torch.nn as nn
import torch
import mmcv


class GradientReverseLayer(Function):
    """
    Implementation of Gradient Reverse Layer in paper
    `Unsupervised Domain Adaptation by Backpropagation <http://arxiv.org/abs/1409.7495>`_

    The input of this class must be torch.Tensor, can't be tuple[torch.Tensor].
    """
    @staticmethod
    def forward(ctx, input, alpha):
        # 正向传播过程中将参数保存到上下文ctx，其中input_是需要传递的张量，alpha_是GRL的超参数
        ctx.save_for_backward(input, alpha)
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        # 取出正向传播时在上下文ctx保存的超参数alpha_
        _, alpha = ctx.saved_tensors
        # 反向传播过程中，如果输入张量input_需要计算梯度，则反转其梯度(*-alpha_)；如果不需要计算梯度，默认其梯度为None
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        # GRL的超参数的超参数alpha_不需要梯度信息，默认其梯度为None
        return grad_input, None


class RevGrad(nn.Module):
    """A complete gradient reversal layer.

    Implementation of Gradient Reverse Layer in paper
    `Unsupervised Domain Adaptation by Backpropagation <http://arxiv.org/abs/1409.7495>`_
    """
    def __init__(self, alpha=1., *args, **kwargs):
        """
        This layer has only one hyperparameter (default to 1), and simply reverses
        the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, dtype=torch.double, requires_grad=False)

    def forward(self, input):
        """
        Initiate the class GradientReverseLayer as grl,
        if the input is a tuple[tensor], get every tensor pass the grl.

        :param input: Tensor or tuple[Tensor], input of GRL
        :return: Tensor or tuple[Tensor], output of GRL
        """
        # use .apply instead of .forward in the forward pass
        grl = GradientReverseLayer.apply

        if isinstance(input, torch.Tensor):
            output = grl(input, self._alpha)
        elif isinstance(input, tuple):
            assert mmcv.is_tuple_of(input, torch.Tensor)
            output = list()
            for level in input:
                output.append(grl(level, self._alpha))
            output = tuple(output)
        else:
            raise ValueError('input must be either torch.Tensor or tuple of torch.Tensor')
        return output


if __name__ == '__main__':
    from torch.autograd import gradcheck

    x = torch.randn([4, 3, 5, 5], dtype=torch.double, requires_grad=True)
    a = torch.tensor([1])
    grl = GradientReverseLayer.apply
    test = gradcheck(grl, (x, a), eps=1e-6, atol=1e-4)
    print(test)
    """
    直接运行本脚本出现以下报错是正常的：
    torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0, ...

    因为按照forward将input直接作为output输出，这应该是一个恒等映射，
    ”按理说“backward应该将grad_output直接作为grad_input输出，
    也就是说GRL是一个特殊的层不能用torch.autograd.gradcheck来检查。
    将第28行改为grad_input = grad_output，再次运行可输出True。
    """


