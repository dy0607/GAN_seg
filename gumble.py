import torch
import torch.nn.functional as F

a = torch.tensor([[[[2, 1, 3]]]]) * 1.0
a = a.transpose(1,3)
print (a.shape)
a.requires_grad = True
b = torch.tensor([0, 1, 2]) * 1.0
b.requires_grad = False

for i in range(100):
    C = torch.matmul(F.gumbel_softmax(a, tau=1, hard=1, dim=1).transpose(1, 3), b).unsqueeze(-1).transpose(1,3)
    C.backward()
    print (a.shape, a.grad)
    quit()