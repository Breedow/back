import numpy as np

# 从 module 包中导入一些自定义类和函数
from module import hello
from module import Zeros, XavierUniform, Activation, module


class Linear(module):
    """一个模仿 PyTorch 线性层 (nn.Linear) 功能的简单实现。
    
    参数：
        d_in:  输入维度 (int)
        d_out: 输出维度 (int)
        w_init: 权重初始化方式（默认为 XavierUniform)
        b_init: 偏置初始化方式（默认为 Zeros)
    """
    def __init__(self, d_in, d_out, w_init=XavierUniform(), b_init=Zeros()):
        # 调用父类(module)的构造函数
        super().__init__()

        # 记录初始化函数，这些用来生成权重和偏置的初始值
        self.initializers = {
            "weight": w_init,
            "bias": b_init,
        }

        # 存放该层最近一次前向传播时的输入，用于反向传播计算梯度
        self.input = None

        # 记录输入/输出维度
        self.d_in = d_in
        self.d_out = d_out

        # 如果传进来的 d_in 不为 0，则直接根据 d_in, d_out 初始化参数
        if d_in:
            self.shapes = {
                "weight": [d_in, d_out],
                "bias": [d_out]
            }
            # 初始化权重和偏置（即 self.params["weight"], self.params["bias"]）
            self._init_params()

    def _forward(self, inputs):
        """前向传播：输入 x -> 输出 y = xW + b
        
        如果还未初始化 (self.is_init = False)，则根据传入的 inputs 动态推断 d_in,
        并完成参数初始化。然后进行矩阵乘法与加偏置操作。
        """
        # 如果还没有初始化（比如 d_in 未知），根据 inputs 的最后一维形状进行初始化
        if not self.is_init:
            d_in = inputs.shape[-1]
            self.shapes = {
                "weight": [d_in, self.d_out],
                "bias": [self.d_out]
            }
            self.d_in = d_in
            self._init_params()

        # 使用 @ 运算符做矩阵乘法，相当于 np.dot
        # out 的形状为 (N, d_out) 或 (d_out,)
        out = inputs @ self.params['weight'] + self.params['bias']

        # 保存这一次 forward 的输入，供 backward 计算时使用
        self.input = inputs

        return out

    def _backward(self, grad):
        """反向传播：根据梯度 grad (dL/dY) 计算并存储对权重、偏置的梯度，
        同时返回对输入的梯度 dL/dX。
        
        grad:
            来自下一层或损失函数对该层输出的梯度 (d_out,) 或 (N, d_out)
        返回:
            dInput = dL/dX, 形状 (d_in,) 或 (N, d_in)
        """
        # is_unbatched 用于标记是否只有一条输入数据(非批量)
        # 如果 grad 一维，就扩充维度到 (1, d_out)，统一按批量形式处理
        is_unbatched = False
        if grad.ndim == 1:
            grad = grad[None, ...]
            is_unbatched = True

        # 同理，如果 self.input 一维，扩成 (1, d_in)
        if self.input.ndim == 1:
            inp = self.input[None, ...]
        else:
            inp = self.input

        # 计算对权重的梯度 dW = X^T * dY
        self.grads["weight"] = inp.T @ grad

        # 计算对偏置的梯度 dB = sum(dY) （在 batch 维度上求和）
        self.grads["bias"] = np.sum(grad, axis=0)

        # 计算对输入的梯度 dX = dY * W^T
        dInput = grad @ self.params["weight"].T

        # 如果原来只有一条数据，就把批量维度去掉，还原成 (d_in,)
        if is_unbatched:
            dInput = dInput[0]

        return dInput

    @property
    def param_names(self):
        """返回可训练参数的名称，供外部调用"""
        return ('weight', 'bias')

    @property
    def weight(self):
        """便捷访问 self.params['weight']"""
        return self.params['weight']

    @property
    def bias(self):
        """便捷访问 self.params['bias']"""
        return self.params['bias']


# 下面是用于测试该 Linear 层的代码
# ------------------------------------------

# 测试用的单条输入 x，形状 (5,)
x = np.array(
    [0.41259363, -0.40173373, -0.9616683, 0.32021663, 0.30066854],
    dtype=np.float32
)

# 测试用的权重 w，形状 (5, 3)
w = np.array(
    [[-0.29742905, -0.4652604,  0.03716598],
     [ 0.63429886,  0.46831214,  0.22899507],
     [ 0.7614463,   0.45421863, -0.7652458 ],
     [ 0.6237591,   0.71807355,  0.81113386],
     [-0.34458044,  0.094055,    0.70938754]],
    dtype=np.float32
)

# 测试用的偏置 b，形状 (3,)
b = np.array([0., 0., 0.], dtype=np.float32)

# 创建一个 Linear 实例，输入维度 5，输出维度 3
l = Linear(5, 3)
# 手动覆盖默认的随机初始化，设置为测试所需的 w、b
l.params['weight'] = w
l.params['bias'] = b
# 标记已经初始化完毕
l.is_init = True

# 做一次前向传播
y = l._forward(x)

# 用前向输出 y 作为“梯度”进行一次反向传播(仅用于测试)
l._backward(y)

# 下面是我们希望得到的梯度，对偏置 grad_b 与 对权重 grad_w
grad_b = np.array([-1.0136619, -0.5586895,  1.1322811], dtype=np.float32)
grad_w = np.array([
    [-0.41823044, -0.23051172,  0.46717197],
    [ 0.40722215,  0.2244444,  -0.4548755 ],
    [ 0.9748065,   0.53727394, -1.0888789 ],
    [-0.32459137, -0.17890166,  0.36257523],
    [-0.30477622, -0.16798034,  0.3404413 ]
], dtype=np.float32)

# 对比计算得到的梯度与期望梯度是否近似相等
assert (np.abs(l.grads['bias'] - grad_b) < 1e-5).all(), "bias 梯度不符合预期"
assert (np.abs(l.grads['weight'] - grad_w) < 1e-5).all(), "weight 梯度不符合预期"

print('success!')
