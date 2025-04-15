import math

import d2l
import numpy as np
import torch

# from timerEg import *
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec')
# timer.start()
# d = a + b
# print(f'{timer.stop():.5f} sec')

def normal(x,mu,sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma **2 * (x - mu) ** 2)
x = np.arange(-7, 7, 0.01)
print(x)

params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])