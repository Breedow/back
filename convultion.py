import numpy as np

# 从 module 包中导入一些自定义类和函数
from module import hello
from module import Zeros, XavierUniform, Activation, module

class Conv2D(module):
    """
    实现 2D 卷积层
    :param kernel: 一个长度为 4 的列表/元组，形状为 (in_channels, k_h, k_w, out_channels)
    :param stride: 一个长度为 2 的列表/元组，形状为 (stride_h, stride_w)
    :param padding: 字符串，支持 ["SAME", "VALID"]
    :param w_init: 权重初始化函数
    :param b_init: 偏置初始化函数
    """

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        # 调用父类 module 的初始化
        super().__init__()

        # 卷积核参数：形状 (in_channels, k_h, k_w, out_channels)
        self.kernel_shape = kernel
        # 步幅：形状 (stride_h, stride_w)
        self.stride = stride
        # 初始化器记录
        self.initializers = {"weight": w_init, "bias": b_init}
        # 记录权重和偏置各自的形状
        self.shapes = {
            "weight": self.kernel_shape,         # 与 kernel_shape 相同
            "bias": self.kernel_shape[-1]        # bias 数量等于 out_channels
        }

        self.padding_mode = padding
        # 只允许 padding 为 'SAME' 或 'VALID'
        assert padding in ['SAME', 'VALID'], "padding 只能是 'SAME' 或 'VALID'"
        # 若选择 'SAME'，则暂不支持 stride != (1, 1)
        if padding == 'SAME' and stride != (1, 1):
            raise RuntimeError("padding='SAME' 不支持 stride != (1, 1) 的情况。")

        # 这里先将 self.padding 置为 None，在后续根据输入张量再动态计算
        self.padding = None

        # 初始化参数（分配内存并使用初始化器赋值）
        self._init_params()

    def _forward(self, inputs):
        """
        前向传播：
        :param inputs: 输入张量，形状为 (batch_size, in_c, in_h, in_w)
        :return: 输出张量，形状为 (batch_size, out_c, out_h, out_w)
        """
        # 基本的形状合法性检查
        assert len(inputs.shape) == 4, "输入形状必须是 (batch_size, in_c, in_h, in_w)"

        # 解析卷积核形状
        in_c, k_h, k_w, out_c = self.kernel_shape
        # 步幅
        s_h, s_w = self.stride
        # 对输入进行 padding（若需要）
        X = self._inputs_preprocess(inputs)
        # 现在 X 的形状是 (batch_size, in_c, padded_h, padded_w)
        bsz, _, h, w = X.shape

        # 计算输出的高和宽
        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1

        # 用零初始化输出
        Y = np.zeros([bsz, out_c, out_h, out_w])

        # 卷积操作：遍历所有输入通道、输出通道和输出空间位置
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                # 取出对应 (in_c_i -> out_c_i) 的卷积核
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                # 扫描输出特征图每一个坐标 (r, c)
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        # 对应在 X 上取出的 patch，并与卷积核 element-wise 相乘
                        patch = X[:, in_c_i, r_start : r_start + k_h,
                                  c_start : c_start + k_w] * kernel
                        # 将这个 patch 的所有元素累加到输出
                        Y[:, out_c_i, r, c] += patch.reshape(bsz, -1).sum(axis=-1)

        # 将原始输入保存到 self.input（方便后续反向传播使用）
        self.input = inputs
        # 加上 bias（形状广播：bias.reshape(1, -1, 1, 1)）
        return Y + self.params['bias'].reshape(1, -1, 1, 1)

    def _backward(self, grad):
        """
        反向传播：
        计算本层参数的梯度(dW, dB)以及对输入的梯度 d_in
        :param grad: 来自上一层的梯度，形状 (batch_size, out_c, out_h, out_w)
        :return: 传递给下一层的梯度 d_in,形状 (batch_size, in_c, in_h, in_w)
        """
        # 检查梯度张量形状
        assert len(grad.shape) == 4, "上游梯度形状必须是 (batch_size, out_c, out_h, out_w)"

        # 解析各个维度
        bsz, out_c, out_h, out_w = grad.shape
        in_c, k_h, k_w, _ = self.kernel_shape
        s_h, s_w = self.stride

        # 在前向传播时对输入做了 padding，所以这里也需要获取并使用补过的输入
        X = self._inputs_preprocess(self.input)  # (bsz, in_c, padded_h, padded_w)

        # 初始化存放梯度的数组
        dW = np.zeros_like(self.params["weight"])  # (in_c, k_h, k_w, out_c)
        dB = np.zeros_like(self.params["bias"])    # (out_c,)
        dX = np.zeros_like(X)                      # (bsz, in_c, padded_h, padded_w)

        # (1) 对 bias 求梯度：
        # 因为每个输出通道对应一个 bias，故将 grad 在 batch 和输出空间两个维度上求和
        dB = grad.sum(axis=(0, 2, 3))  # 最后得到形状为 (out_c,)

        # (2) 对 weight 求梯度 & 计算对输入的梯度
        for oc in range(out_c):
            for ic in range(in_c):
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        # 取出在 X 对应位置的 patch
                        patch = X[:, ic, r_start : r_start + k_h, c_start : c_start + k_w]
                        # 取出该输出通道对应位置上的梯度：(batch_size,)
                        grad_val = grad[:, oc, r, c].reshape(-1, 1, 1)

                        # ----- (2a) weight 梯度累加 -----
                        # 将 patch 与 grad_val 相乘后，在 batch 维度上累加
                        dW[ic, :, :, oc] += np.sum(patch * grad_val, axis=0)

                        # ----- (2b) 输入梯度累加 -----
                        # 每个位置的梯度分配回输入 dX 对应的 patch
                        dX[:, ic, r_start : r_start + k_h, c_start : c_start + k_w] += (
                            self.params["weight"][ic, :, :, oc] * grad_val
                        )

        # 将计算好的梯度存到 self.grads 字典
        self.grads["weight"] = dW
        self.grads["bias"]   = dB

        # (3) 去除 padding，得到与原始输入同大小的梯度
        pad = self.padding  # 形如 ((0,0),(0,0),(pad_top, pad_bottom),(pad_left, pad_right))
        d_in = dX[:, :,
                  pad[2][0] : dX.shape[2] - pad[2][1],
                  pad[3][0] : dX.shape[3] - pad[3][1]
                 ]

        return d_in

    def _inputs_preprocess(self, inputs):
        """
        根据 padding 设置，对输入进行零填充
        """
        _, _, in_h, in_w = inputs.shape
        _, k_h, k_w, _ = self.kernel_shape

        # 如果尚未计算过 self.padding，就先进行计算
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode
            )
        # 利用 numpy.pad 进行零填充
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def get_padding_2d(self, in_shape, k_shape, stride, mode):
        """
        计算 2D 卷积时，高度和宽度各自需要的 padding 大小
        in_shape: (in_h, in_w)
        k_shape:  (k_h, k_w)
        stride:   (s_h, s_w)
        mode:     "SAME" or "VALID"
        """

        def get_padding_1d(w, k, s):
            if mode == "SAME":
                # SAME 模式下的公式：pads = s * (w - 1) + k - w
                pads = s * (w - 1) + k - w
                half = pads // 2
                # 如果是奇数，需要多填充 1
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                # VALID 不做填充
                padding = (0, 0)
            return padding

        # 分别计算 height 和 width 的填充
        h_pad = get_padding_1d(in_shape[0], k_shape[0], stride[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1], stride[1])
        # numpy.pad 的格式是 (batch_pad, channel_pad, height_pad, width_pad)
        return (0, 0), (0, 0), h_pad, w_pad

    @property
    def param_names(self):
        # 返回本层含有可学习参数的名称
        return "weight", "bias"

    @property
    def weight(self):
        # 便于外部访问 self.params["weight"]
        return self.params['weight']

    @property
    def bias(self):
        # 便于外部访问 self.params["bias"]
        return self.params['bias']


# ---------------- 测试代码 ----------------

# 准备测试用的输入张量 x (batch_size=2, in_c=2, in_h=3, in_w=3)
x = np.array([[[[-1.75957111,  0.0085911 ,  0.30235818],
                [-1.05931037,  0.75555462, -2.03922536],
                [ 0.86653209, -0.56438439, -1.68797524]],

               [[-0.74832044,  0.21611616,  0.571611  ],
                [-1.61335018, -0.37620906,  1.0353189 ],
                [-0.26074537,  1.98065489, -1.30691981]]],


              [[[ 0.32680334,  0.29817393,  2.25433969],
                [-0.16831957, -0.98864486,  0.36653   ],
                [ 1.52712821,  1.19630751, -0.02024759]],

               [[ 0.48080474, -1.15229596, -0.95228854],
                [-1.68168285, -2.86668484, -0.34833734],
                [ 0.73179971,  1.69618114,  1.33524773]]]], dtype=np.float32)

# 卷积核参数 w 和偏置 b
w = np.array([[[[-0.322831  ,  0.38674766,  0.32847992,  0.3846352 ],
                [-0.21158722, -0.53467643, -0.28443742, -0.20367976]],

               [[ 0.4973593 , -0.30178958, -0.02311361, -0.53795236],
                [-0.1229187 , -0.12866518, -0.40432686,  0.5104686 ]]],


              [[[ 0.19288206, -0.49516755, -0.26484585, -0.35625377],
                [ 0.5058061 , -0.17490079, -0.40337119,  0.10058666]],

               [[-0.24815331,  0.34114942, -0.06982624,  0.4017606 ],
                [ 0.16874631, -0.42147416,  0.43324274,  0.16369782]]]], dtype=np.float32)
b = np.array([0., 0., 0., 0.], dtype=np.float32)

# 创建一个 Conv2D 层示例
l = Conv2D(kernel=(2, 2, 2, 4), padding='SAME', stride=(1, 1))
# 手动设置其 weight 和 bias
l.params['weight'] = w
l.params['bias'] = b
l.is_init = True

# 前向传播
y = l._forward(x)
# 反向传播（这里为了测试，直接用 y 作为上游梯度）
l._backward(y)

# 下面是参考的正确梯度值，用于断言测试
grad_b = np.array([-0.49104962,  1.4335476 ,  2.70048173, -0.0098734 ],
                  dtype=np.float32)
grad_w = np.array(
    [[
        [[ -3.0586028,   7.7819834,   1.3951588,   5.9249396],
         [ -1.5760803, -10.541515 ,  -2.694372 ,  -3.9848034]],
        [[  2.9096646,   0.6696263,   8.230143 ,  -0.3434674],
         [ -2.9487448,  -3.264796 ,  -1.1822633,   4.1672387]]
    ],
     [
        [[  3.7202294,  -5.4176836, -10.34358  ,  -6.4479938],
         [  7.0336857,  -0.41946477, -8.181945 ,   3.0968976]],
        [[  0.25020388, 13.39637   ,  5.8576417,  12.522377 ],
         [  3.360495  , -6.597466  ,  8.375789  ,   3.8179488]]
    ]],
    dtype=np.float32
)

# 通过断言比较计算得到的梯度和期望值是否一致
assert (np.abs(l.grads['bias'] - grad_b) < 1e-5).all(), "bias 梯度不匹配!"
assert (np.abs(l.grads['weight'] - grad_w) < 1e-5).all(), "weight 梯度不匹配!"

print('success!')
