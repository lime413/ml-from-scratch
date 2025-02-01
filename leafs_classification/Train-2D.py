import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch
from tqdm import tqdm
import numpy as np

from dataset import LeafsDataset

class ConvNet(nn.Module): 
    def __init__(self, num_classes): 
        super(ConvNet, self).__init__() 
        self.layer_conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer_conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer_dropout = nn.Dropout()
        self.layer_fc1 = nn.Linear(262144, 1000)
        self.layer_fc2 = nn.Linear(1000, num_classes)

    def forward(self, x): 
       out = self.layer_conv1(x) 
       out = self.layer_conv2(out)
       out = out.reshape(out.size(0), -1)
       out = self.layer_dropout(out)
       out = self.layer_fc1(out)
       out = self.layer_fc2(out)
       return out
    

def main():
    np.random.seed(2025)

    num_epochs = 5
    num_classes = 4
    batch_size = 24
    learning_rate = 0.001
    datapath = 'leafs_classification/data/images_transformed/Apple'
    ckpt_path = 'leafs_classification/ckpts/'

    trans = transforms.Compose([transforms.ToTensor(), 
                                # transforms.Normalize((0.1307,), (0.3081,))
                                ])

    train_dataset = LeafsDataset(datapath, attribute='__original', train=True, transform=trans)
    test_dataset = LeafsDataset(datapath, attribute='__original', train=False, transform=trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True) 
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for (images, labels) in tqdm(train_loader, desc='Epoch: ' + str(epoch+1)):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
        
        print('Loss: {:.4f}, Accuracy: {:.2f}%'.format(loss.item(), (correct / float(total)) * 100))
                
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Сохраняем модель и строим график
    torch.save(model.state_dict(), ckpt_path + 'leafs_2D.ckpt')


if __name__ == '__main__':
    main()