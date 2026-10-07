import torch

# 本部分的测试都是很小的张量运算；单线程避免线程调度开销，在繁忙的机器上也能稳定在数秒内。
torch.set_num_threads(1)
