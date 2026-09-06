# 论文公式索引

公式均来自论文第5页。这里使用LaTeX重排，并保留论文中的公式编号；不把重排版本伪装成论文截图。

## RGB转YUV - 公式(13)

$$
\begin{aligned}
Y &= 0.299R + 0.587G + 0.114B \\
U &= -0.1687R - 0.3313G + 0.500B + 128 \\
V &= 0.500R - 0.4187G - 0.0813B + 128
\end{aligned}
\tag{13}
$$

对应我们的`src/yuv_rae/yuv_conversion.py::rgb_to_yuv`。

## YUV转RGB - 公式(14)

$$
\begin{aligned}
R &= Y + 1.402(V-128) \\
G &= Y - 0.34414(U-128) - 0.71414(V-128) \\
B &= Y + 1.772(U-128)
\end{aligned}
\tag{14}
$$

对应我们的`src/yuv_rae/yuv_conversion.py::yuv_to_rgb`。

## 关于4:4:4

4:4:4表示Y、U、V三个通道具有相同的空间分辨率，本项目没有对U/V进行色度下采样。它描述的是数据表示方式，不是颜色转换函数名称必须包含的后缀，也不代表整数取整后必然逐像素可逆。
