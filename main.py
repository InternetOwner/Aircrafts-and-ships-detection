import torch
from PIL import Image
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train = dataset.ImageFolder(root='./train/', transform=transform)
# test = dataset.ImageFolder(root='./test/', transform=transform)
#
# train_loader = DataLoader(train, batch_size=32, shuffle=True)
# test_loader = DataLoader(test, batch_size=32, shuffle=False)


resnet152 = models.resnet152(pretrained=True)

transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# откроем пример изображения, которого необходимо классифицировать
image = Image.open("./test/aircraft/171.jpg")
image = transform(image)
image = image.unsqueeze(0)

# переводим модель в режим теста и формируем предсказания
resnet152.eval()
preds = resnet152(image)
pred = preds.argmax()
print(pred)


