import numpy as np
from pylab import *

D = 10
acc1 = np.load('res/small/acc.npy').reshape(D, -1).mean(axis=0)
loss1 = np.load('res/small/loss.npy').reshape(D, -1).mean(axis=0)
acc2 = np.load('res/large/acc.npy').reshape(D, -1).mean(axis=0)
loss2 = np.load('res/large/loss.npy').reshape(D, -1).mean(axis=0)
cut = int(acc1.shape[0] / 10 * 4)
print(' 1: %.2f %.6f'%(100*acc1[:cut].max(), loss1[:cut].min()))
print(' 2: %.2f %.6f'%(100*acc2[:cut].max(), loss2[:cut].min()))
iter_ = np.arange(acc1.shape[0]) * D
print(acc1.shape, iter_.shape[0])
figure()
p = subplot(111)
p.plot(iter_[:cut], loss1[:cut], '-', label='Original CNN')
p.plot(iter_[:cut], loss2[:cut], '-', label='Designed CNN')
p.set_ylim((0, .4))
p.set_xlabel(r'# of Iterations')
p.set_ylabel(r'Loss')
p.legend(loc='upper right')
tight_layout()
savefig("loss.pdf")
figure()
p = subplot(111)
p.plot(iter_[:cut], acc1[:cut], '-', label='Original CNN')
p.plot(iter_[:cut], acc2[:cut], '-', label='Designed CNN')
p.set_ylim((.9, 1))
p.set_xlabel(r'# of Iterations')
p.set_ylabel(r'Accuracy')
p.legend(loc='lower right')
tight_layout()
savefig("acc.pdf")

#  1: 23:24:44.414     Testing, total mean loss 0.019417, total acc 0.863300 - 23:24:33.131
# 2s: 20:20:39.807     Testing, total mean loss 0.003224, total acc 0.967700 - 20:18:21.597
# 2r: 20:48:01.448     Testing, total mean loss 0.002306, total acc 0.981300 - 20:45:16.709
#-2r: 20:38:47.940     Testing, total mean loss 0.002271, total acc 0.981500 - 20:35:59.910
# 3s: 00:38:10.865     Testing, total mean loss 0.001759, total acc 0.980098 - 00:33:01.622
# 3r: 21:24:04.253     Testing, total mean loss 0.001675, total acc 0.980588 - 21:19:28.262