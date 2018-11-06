import numpy as np

def im2col(input, kh, kw, stride=1):
    '''
    input: N * C * H * W with pad
    out: C*kh*kw * N*hout*wout
    '''
    N, C, H, W = input.shape
    hout = (H - kh) / stride + 1
    wout = (W - kw) / stride + 1
    shapes = (C, kh, kw, N, hout, wout)
    strides = input.itemsize * np.array([H * W, W, 1, C * H * W, W * stride, stride])
    output = np.lib.stride_tricks.as_strided(input, shape=shapes, strides=strides)
    return output.reshape(C * kh * kw, N * hout * wout)

def im2col_conv(input, w, b=None): # with pad
    N, C, H, W = input.shape
    cout, _, kh, kw = w.shape
    col_input = im2col(input, kh, kw)
    col_out = w.reshape(cout, -1).dot(col_input)
    if b is not None:
        col_out += b.reshape(-1, 1)
    hout = H - kh + 1
    wout = W - kw + 1
    return col_out.reshape(cout, N, hout, wout).transpose((1, 0, 2, 3))

def conv2d_forward(input, w, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    return im2col_conv(input, w, b)

def conv2d_backward(input, grad_output, w, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    grad_b = grad_output.sum(axis=(0, 2, 3))
    grad_w = im2col_conv(input.transpose((1, 0, 2, 3)), grad_output.transpose((1, 0, 2, 3))).transpose((1, 0, 2, 3))
    grad_out = np.pad(grad_output, ((0, 0), (0, 0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)), 'constant')
    grad_w_loc = np.rot90(w.transpose((1, 0, 2, 3)), 2, axes=(2, 3))
    grad_input = im2col_conv(grad_out, grad_w_loc)
    if pad != 0:
        grad_input = grad_input[:, :, pad:-pad, pad:-pad]
    return grad_input, grad_w, grad_b

def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    if kernel_size == 2 and pad == 0:
        out = (input[..., ::2, ::2] + input[..., ::2, 1::2] + input[..., 1::2, ::2] + input[..., 1::2, 1::2]) * .25
        return out
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    N, C, H, W = input.shape
    hout = H / kernel_size
    wout = W / kernel_size
    input = input.reshape([N * C, 1, H, W])
    col_input = im2col(input, kernel_size, kernel_size, stride=kernel_size).reshape(kernel_size*kernel_size, -1)
    output = col_input.mean(axis=0).reshape(N, C, hout, wout)
    return output

def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad = np.kron(grad_output, np.ones((kernel_size, kernel_size))) / (kernel_size * kernel_size)
    if pad == 0:
        return grad
    else:
        return grad[:, :, pad:-pad, pad:-pad]