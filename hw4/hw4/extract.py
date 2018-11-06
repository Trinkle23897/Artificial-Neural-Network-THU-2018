import numpy as np
name = ['rnn', 'lstm', 'gru']
for i in name:
    for j in [1,2]:
        info = open('%s%d.log'%(i, j)).read().split('Created model with fresh parameters.\n')[-1].split('\n')[:-1]
        train_loss = []
        train_acc = []
        dev_loss = []
        dev_acc = []
        test_acc = []
        for train in info[::3]:
            train_loss.append(float(train.split('loss ')[-1].split('acc')[0]))
            train_acc.append(float(train.split('[')[-1].replace(']','')))
        train_loss = np.array(train_loss)
        train_acc = np.array(train_acc)
        print('%s%d & %f & %f '%(i.upper(),j, train_loss.min(), train_acc.max()))
        np.save('%s%d_train_loss'%(i,j), train_loss)
        np.save('%s%d_train_acc'%(i,j), train_acc)
        for dev in info[1::3]:
            dev_loss.append(float(dev.split('loss ')[-1].split(', ')[0]))
            dev_acc.append(float(dev.split('[')[-1].replace(']','')))
        dev_loss = np.array(dev_loss)
        dev_acc = np.array(dev_acc)
        print('& %f & %f \\\\' % (dev_loss.min(), dev_acc.max()))
        np.save('%s%d_dev_loss'%(i,j), dev_loss)
        np.save('%s%d_dev_acc'%(i,j), dev_acc)
        for test in info[2::3]:
            test_acc.append(float(test.split('test ')[-1].split(' ')[0]))
        test_acc = np.array(test_acc)
        #print('%s%d test: '%(i,j), test_acc.max())
        np.save('%s%d_test_acc' %(i,j), test_acc)
