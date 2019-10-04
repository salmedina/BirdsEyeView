import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib.pyplot import imshow

num_classes = 6
model = models.inception_v3(pretrained=True)
num_ftrs = model.AuxLogits.fc.in_features
model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
# Handle the primary net
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

for param in model.parameters():
    param.requires_grad = False
model.training = False
print(model)

modules = list(model.children())

print(f'Total modules: {len(modules)}')
print(modules[0])
print(modules[-1])


fox_image = Image.open('../res/fox.jpg')
small_fox_image = fox_image.resize((299,299)).convert('RGB')
imshow(small_fox_image)

fox_tensor = torch.tensor(np.transpose(np.array(small_fox_image, dtype=np.float32), (2, 0, 1))).float()
fox_tensor = fox_tensor.unsqueeze(0)
x = model(fox_tensor)[0]

#  x1   y1   x2   y2   x3   y3
#   0    1    2    3    4    5
A = torch.tensor([[x[0]*x[2]+x[1]*x[3], x[0]+x[2], x[1]+x[3], 1],
                  [x[0]*x[4]+x[1]*x[5], x[0]+x[4], x[1]+x[5], 1],
                  [x[2]*x[4]+x[3]*x[5], x[2]+x[4], x[3]+x[5], 1]],
                 dtype=torch.float, requires_grad=True)

w = torch.randn(4, 1, dtype=torch.float, requires_grad=False)
loss = w.t() @ A.t() @ A @ w
loss.backward()

