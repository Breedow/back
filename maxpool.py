import numpy as np

# 从 module 包中导入一些自定义类和函数
from module import hello
from module import Zeros, XavierUniform, Activation, module
# 假设 module 是你的项目里已有的一个基类
class module:
    def __init__(self):
        pass

class MaxPool2D(module):
    """
    2D 最大池化层
    :param pool_size: 形如 (pool_height, pool_width)
    :param stride: 形如 (stride_height, stride_width)
    :param padding: "SAME" 或 "VALID"
    """
    def __init__(self, pool_size, stride, padding="VALID"):
        super().__init__()
        # 池化窗口的大小 (k_h, k_w)
        self.kernel_shape = pool_size
        # 步幅 (s_h, s_w)
        self.stride = stride

        # padding 的方式：可取 "SAME" 或 "VALID"
        self.padding_mode = padding
        # 具体的 padding 值在 forward 时计算，这里先设为 None
        self.padding = None

    def _forward(self, inputs):
        """
        前向传播:
        :param inputs: 形状 (batch_size, in_c, in_h, in_w)
        :return: shape (batch_size, in_c, out_h, out_w)
        """
        # 从 stride 和 kernel_shape 中解析出所需的数值
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_c, in_h, in_w = inputs.shape

        # 若 self.padding 还未计算，则根据 padding_mode 自动计算需要的 padding 大小
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode
            )

        # 对输入做零填充 (若 padding 为 0，则不影响输入)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[2], X.shape[3]

        # 计算输出的高宽
        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # 创建存放结果的数组: max_pool 保存最大值, argmax 保存最大值位置(索引)
        max_pool = np.empty(shape=(batch_sz, in_c, out_h, out_w))
        argmax   = np.empty(shape=(batch_sz, in_c, out_h, out_w), dtype=int)

        # 遍历输出特征图的每个像素 (r, c)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                # 取出对应输入 patch, 大小为 (k_h, k_w)
                # 形状: (batch_sz, in_c, k_h, k_w)
                pool = X[:, :, r_start : r_start + k_h, c_start : c_start + k_w]
                # 将 (k_h, k_w) reshape 到同一维度, 得到 (batch_sz, in_c, k_h*k_w)
                pool = pool.reshape((batch_sz, in_c, -1))

                # 对每个 (batch, in_c) 找到最大值所在位置
                _argmax = np.argmax(pool, axis=2)[:, :, np.newaxis]   # (batch_sz, in_c, 1)
                argmax[:, :, r, c] = _argmax.squeeze(axis=2)          # (batch_sz, in_c)

                # 取最大值本身
                _max_pool = np.take_along_axis(pool, _argmax, axis=2).squeeze(axis=2)
                max_pool[:, :, r, c] = _max_pool

        # 记录一些必要的信息给反向传播使用
        self.X_shape = X.shape                 # (batch_sz, in_c, padded_h, padded_w)
        self.out_shape = (out_h, out_w)        # 输出的高宽
        self.argmax = argmax                   # 每个输出位置的最大值索引
        return max_pool

    def _backward(self, grad):
        """
        反向传播: 将上游梯度 grad 通过 MaxPool 的 argmax 索引传回输入
        :param grad: 形状 (batch_size, in_c, out_h, out_w)
        :return: d_in, 形状 (batch_size, in_c, in_h, in_w)
        """
        # 从前向传播中得到的信息
        batch_sz, in_c, padded_h, padded_w = self.X_shape  # 补过零填充后的形状
        out_h, out_w = self.out_shape                      # 输出特征图的高、宽
        k_h, k_w = self.kernel_shape
        s_h, s_w = self.stride

        # dX 用于存储对“补过零后的输入”所产生的梯度
        dX = np.zeros((batch_sz, in_c, padded_h, padded_w), dtype=grad.dtype)

        # 遍历输出特征图, 根据 argmax 将梯度“路由”到输入的正确位置
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                # argmax[:, :, r, c] 里保存了每个 (batch, in_c) patch 内最大值的索引 (一维)
                patch_argmax = self.argmax[:, :, r, c]   # (batch_sz, in_c)
                grad_val = grad[:, :, r, c]              # 对应的上游梯度 (batch_sz, in_c)

                # 将一维索引拆分为 (row_offset, col_offset)
                row_offset = patch_argmax // k_w
                col_offset = patch_argmax % k_w

                # 遍历 batch 和通道, 将梯度加回到输入梯度 dX 相应位置上
                for b in range(batch_sz):
                    for ic_ in range(in_c):
                        ro = row_offset[b, ic_]
                        co = col_offset[b, ic_]
                        dX[b, ic_, r_start + ro, c_start + co] += grad_val[b, ic_]

        # 从 dX 中去除在前向时添加的 padding，得到与原始输入形状相同的 d_in
        pad = self.padding  # 形如((0,0),(0,0),(pad_top, pad_bottom),(pad_left, pad_right))
        d_in = dX[:,
                  :,
                  pad[2][0] : padded_h - pad[2][1],
                  pad[3][0] : padded_w - pad[3][1]
                 ]
        return d_in

    def get_padding_2d(self, in_shape, k_shape, stride, mode):
        """
        计算 2D 池化/卷积时需要的 padding 大小
        in_shape = (in_h, in_w)
        k_shape  = (k_h, k_w)
        stride   = (s_h, s_w)
        mode     = "SAME" 或 "VALID"
        """

        def get_padding_1d(w, k, s):
            if mode == "SAME":
                # 参考公式：pads = s*(w-1) + k - w
                pads = s * (w - 1) + k - w
                half = pads // 2
                # 若 pads 为奇数，则需要再加 1
                padding = (half, half) if (pads % 2 == 0) else (half, half + 1)
            else:
                # VALID 不填充
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0], stride[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1], stride[1])
        # numpy.pad 的四元组格式: ((batch前, batch后), (channel前, channel后),
        #                         (height前, height后), (width前, width后))
        return (0, 0), (0, 0), h_pad, w_pad


# ---------------- 测试代码 ----------------
# 构造一个简单输入 x: (batch_size=2, in_c=1, in_h=4, in_w=4)
x = np.array([
    [[
        [-2.23317337,  0.9750834,  -1.30762567, -0.71442179],
        [ 0.24624013, -1.77593893, -0.43530428,  1.03446008],
        [ 1.58317228, -0.66459249,  0.54894879, -1.19706709],
        [ 0.06013156,  1.05886458,  0.26634763,  1.03497421]
    ]],
    [[
        [ 2.20214308, -0.53358514,  0.96765812, -1.74976553],
        [-0.07049627,  0.88147726,  2.15051543, -0.78627764],
        [ 1.19180886,  0.00468398, -1.74774108,  0.18564536],
        [ 1.39397303, -1.0462731 ,  0.4786774 , -0.51543751]
    ]]
], dtype=np.float32)

# 创建一个 2D 最大池化，窗口=2x2, 步幅=2x2
l = MaxPool2D(pool_size=(2, 2), stride=(2, 2))
l.is_init = True

# 前向计算
y = l._forward(x)
# 用前向输出当作上游梯度传给 _backward，仅做测试用
grad_ = l._backward(y)

# 期望的梯度结果
grad_expected = np.array([
    [[[ 0.       ,  0.9750834,  0.       ,  0.       ],
      [ 0.       ,  0.       ,  0.       ,  1.0344601],
      [ 1.5831723,  0.       ,  0.       ,  0.       ],
      [ 0.       ,  0.       ,  0.       ,  1.0349742]]],

    [[[ 2.2021432,  0.       ,  0.       ,  0.       ],
      [ 0.       ,  0.       ,  2.1505153,  0.       ],
      [ 0.       ,  0.       ,  0.       ,  0.       ],
      [ 1.393973 ,  0.       ,  0.4786774,  0.       ]]]], dtype=np.float32)

# 验证是否与期望值一致
assert (np.abs(grad_ - grad_expected) < 1e-5).all(), "反向传播结果不匹配!"
print("success!")
