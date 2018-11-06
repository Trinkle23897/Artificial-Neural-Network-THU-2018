import numpy as np
from functions import conv2d_forward, conv2d_backward, avgpool2d_forward, avgpool2d_backward


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)
        input[input <= 0] = 0
        return input

    def backward(self, grad_output):
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(1. / (1. + np.exp(-input)))
        return self._saved_tensor

    def backward(self, grad_output):
        '''Your codes here'''
        return self._saved_tensor * (1. - self._saved_tensor) * grad_output


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        self.grad_W = self._saved_tensor.T.dot(grad_output) # 784x10 = 784x100 * 100x10
        self.grad_b = grad_output.sum(axis=0)
        return grad_output.dot(self.W.T)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


class Reshape(Layer):
    def __init__(self, name, new_shape):
        super(Reshape, self).__init__(name)
        self.new_shape = new_shape

    def forward(self, input):
        self._saved_for_backward(input)
        return input.reshape(*self.new_shape)

    def backward(self, grad_output):
        input = self._saved_tensor
        return grad_output.reshape(*input.shape)


class Conv2D(Layer):
    def __init__(self, name, in_channel, out_channel, kernel_size, pad, init_std):
        super(Conv2D, self).__init__(name, trainable=True)
        self.kernel_size = kernel_size
        self.pad = pad
        self.W = np.random.randn(out_channel, in_channel, kernel_size, kernel_size) * init_std
        self.b = np.zeros(out_channel)

        self.diff_W = np.zeros(self.W.shape)
        self.diff_b = np.zeros(out_channel)

    def forward(self, input):
        self._saved_for_backward(input)
        output = conv2d_forward(input, self.W, self.b, self.kernel_size, self.pad)
        return output

    def backward(self, grad_output):
        input = self._saved_tensor
        grad_input, self.grad_W, self.grad_b = conv2d_backward(input, grad_output, self.W, self.b, self.kernel_size, self.pad)
        return grad_input

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


class AvgPool2D(Layer):
    def __init__(self, name, kernel_size, pad):
        super(AvgPool2D, self).__init__(name)
        self.kernel_size = kernel_size
        self.pad = pad

    def forward(self, input):
        self._saved_for_backward(input)
        output = avgpool2d_forward(input, self.kernel_size, self.pad)
        return output

    def backward(self, grad_output):
        input = self._saved_tensor
        grad_input = avgpool2d_backward(input, grad_output, self.kernel_size, self.pad)
        return grad_input
