import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
          transforms.ToTensor()
])

train = dataset.ImageFolder(root='./train/', transform=transform)
test = dataset.ImageFolder(root='./test/', transform=transform)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # стек сверточных слоев
        self.conv_layers = nn.Sequential(
            # здесь определяются сверточные слои
            # можно явно вычислить размер выходной карты признаков каждого
            # сверточного слоя по следующей формуле:
            # [(shape + 2*padding - kernel_size) / stride] + 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),  # (N, 1, 28, 28)
            nn.ReLU(),
            # после первого сверточного слоя размер выходной карты признаков равен:
            # [(28 + 2*1 - 3)/1] + 1 = 28.
            nn.MaxPool2d(kernel_size=2),
            # при прохождении слоя MaxPooling с размером окна 2
            # карты признаков сжимаются вдвое
            # 28 / 2 = 14
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # после второго сверточного слоя размер выходной карты признаков равен:
            # [(14 + 2*1 - 3)/1] + 1 = 14
            nn.MaxPool2d(kernel_size=2),
            # после второго слоя MaxPooling2D выходнае карты признаков имеют размерность
            # 14 / 2 = 7
        )
        # стек полносвязных слоев
        self.linear_layers = nn.Sequential(
            # после второго сверточного слоя имеем количество выходных карт признаков
            # равное 24 размером 7х7
            # эти данные и будут входными признаками в первом линейном слое
            nn.Linear(in_features=24 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),  # обнуляем 20% входного тензора для предотвращения переобучения
            nn.Linear(in_features=64, out_features=10)  # количество выходных признаков равно количеству классов

        )

    # определение метода для прчмого распространения сигналов по сети
    def forward(self, x):
        x = self.conv_layers(x)
        # перед отправкой в блок полносвязных слоев признаки необходимо сделать одномерными
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# вывод структуры модели
cnn = CNN()
print(cnn)