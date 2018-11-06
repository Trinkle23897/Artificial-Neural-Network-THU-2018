from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
# model = Network()
# model.add(Linear('fc1', 784, 10, 0.001))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

# config = {
#     'learning_rate': 0.000001,
#     'weight_decay': 0.005,
#     'momentum': 0.9,
#     'batch_size': 100,
#     'max_epoch': 100,
#     'disp_freq': 50,
#     'test_epoch': 5
# }

def one_layer_net():
    model = Network()
    model.add(Linear('fc1', 784, 10, 0.001))
    config = {
        'learning_rate': 0.00001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 50,
        'max_epoch': 10,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config

def two_layer_sigmoid():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 256, 10, 0.001))
    model.add(Sigmoid('sg2'))
    config = {
        'learning_rate': 0.01,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 20,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config

def two_layer_relu():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 10, 0.001))
    model.add(Relu('rl2'))
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 200,
        'max_epoch': 40,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config

def three_layer_sigmoid():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 256, 128, 0.001))
    model.add(Sigmoid('sg2'))
    model.add(Linear('fc3', 128, 10, 0.001))
    model.add(Sigmoid('sg3'))
    config = {
        'learning_rate': 0.01,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 300,
        'max_epoch': 60,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config

def three_layer_relu():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 128, 0.001))
    model.add(Relu('rl2'))
    model.add(Linear('fc3', 128, 10, 0.001))
    model.add(Relu('rl3'))
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 300,
        'max_epoch': 60,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config

model, config = one_layer_net()

loss_, acc_ = [], []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    a, b = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    loss_ += a
    acc_ += b
    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
test_net(model, loss, test_data, test_label, config['batch_size'])

import numpy as np
np.save('loss', loss_)
np.save('acc', acc_)