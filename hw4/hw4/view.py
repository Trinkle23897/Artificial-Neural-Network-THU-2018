import os
import numpy as np
from pylab import *
from matplotlib.legend_handler import HandlerTuple

length = np.load('gru1_dev_loss.npy').shape[0]
models = ['rnn1', 'rnn2', 'lstm1', 'lstm2', 'gru1', 'gru2']
iter_ = np.arange(length)
# train_loss + dev_loss
train_loss = []
dev_loss = []
for model in models:
	train_loss.append(np.load(model + '_train_loss.npy'))
	dev_loss.append(np.load(model + '_dev_loss.npy'))

figure()
p = subplot(111)
for i, model in enumerate(models):
	l1 = p.plot(iter_, train_loss[i], '-', label='%s train loss' % model.upper())
for i, model in enumerate(models):
	l2 = p.plot(iter_, dev_loss[i], '--', label='%s dev loss' % model.upper())
p.set_ylim((0, .5))
p.set_xlabel(r'# of Epochs')
p.set_ylabel(r'Loss')
p.legend(loc="upper left", ncol=2)
tight_layout()
savefig("loss.pdf")

# train_acc + dev_acc

train_acc = []
dev_acc = []
for model in models:
	train_acc.append(np.load(model + '_train_acc.npy'))
	dev_acc.append(np.load(model + '_dev_acc.npy'))

figure()
p = subplot(111)
for i, model in enumerate(models):
	p.plot(iter_, train_acc[i], '-', label='%s train acc' % model.upper())
for i, model in enumerate(models):
	p.plot(iter_, dev_acc[i], '--', label='%s dev acc' % model.upper())
p.set_ylim((-.1, 1.1))
p.set_xlabel(r'# of Epochs')
p.set_ylabel(r'Accuracy')
p.legend(loc="lower right", ncol=2)
tight_layout()
savefig("acc.pdf")

# test_acc

test_acc = []
for model in models:
	test_acc.append(np.load(model + '_test_acc.npy'))

figure()
p = subplot(111)
for i, model in enumerate(models):
	p.plot(iter_, test_acc[i], '-', label='%s test acc' % model.upper())
p.set_ylim((.0, 1))
p.set_xlabel(r'# of Epochs')
p.set_ylabel(r'Accuracy')
p.legend(loc='lower right')
tight_layout()
savefig("test.pdf")
