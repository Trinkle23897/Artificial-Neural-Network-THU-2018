`plot.py`是我用于画loss曲线的脚本。

关于SGD+Momentum，我觉得如下代码有问题，不太符合原始SGD规范（比如lr乘的位置都不对）：

```python
self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
self.W = self.W - lr * self.diff_W

self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
self.b = self.b - lr * self.diff_b
```

我把代码修改为以下代码：

```python
self.diff_W = mm * self.diff_W - lr * (self.grad_W + wd * self.W)
self.W = self.W + self.diff_W

self.diff_b = mm * self.diff_b - lr * (self.grad_b + wd * self.b)
self.b = self.b + self.diff_b
```

并进行对比测试，几组对比实验下来，我修改过的SGD代码跑出的结果都要比没修改过的要好。