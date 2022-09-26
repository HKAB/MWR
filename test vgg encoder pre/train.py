import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from ResMaskNet import res_attention
from DataLoader import ImageLoader_New
from torch.utils.data import DataLoader
from VGG_encoder import VGG_cls_pre

def apparent_age(m, predict, age_range):
    predict = m(predict)
    return torch.sum(predict * age_range, dim=1)



def train(net, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    mea_loss = nn.L1Loss()
    age_range = torch.arange(21, 61).float().cuda()
    m = torch.nn.Softmax(dim=1).cuda()

    best_epoch = 0
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_mae = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels2 = labels-21

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                running_mae += mea_loss(apparent_age(m, outputs, age_range), labels).item()

            if i % 100 == 99:
                print('[%d, %5d] train cls loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                print('[%d, %5d] train mae loss: %.3f' % (epoch + 1, i + 1, running_mae / 100))
                running_loss = 0.0
                running_mae = 0.0

        if epoch % 3 == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_mae = 0.0
                for i, data in enumerate(test_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels2 = labels-21

                    outputs = net(inputs)
                    loss = criterion(outputs, labels2)

                    val_loss += loss.item()
                    val_mae += mea_loss(apparent_age(m, outputs, age_range), labels).item()
            if val_loss/len(test_loader) < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f"vgg_epoch_{best_epoch}.pth")
            print('save parameters to file: %s' % "vgg.pth")
            print('test cls loss: %.3f' % (running_loss / len(test_loader)))
            print('test mae loss: %.3f' % (running_mae / len(test_loader)))

    print('Finished Training')
if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG_cls_pre().to(device)
    # unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    train_path = 'UTK_mtcnn/csv_files/UTK_train_coral.csv'
    test_path = 'UTK_mtcnn/csv_files/UTK_test_coral.csv'
    train_dataset = ImageLoader_New(train_path, img_size=224, aug=True)
    test_dataset = ImageLoader_New(test_path, img_size=224, aug=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    train(model, criterion, optimizer, train_loader, test_loader, num_epochs=50)

