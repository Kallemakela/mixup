import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def mixup(X, y, alpha=0.2, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    n_samples = X.shape[0]
    lam = rng.beta(alpha, alpha)
    perm_ix = torch.randperm(n_samples)
    Xp = X[perm_ix]
    yp = y[perm_ix]
    X_mix = lam * X + (1 - lam) * Xp
    return X_mix, y, yp, lam

def mixup_loss(y_pred, y, y_p, lam):
    return lam * F.cross_entropy(y_pred, y) + (1 - lam) * F.cross_entropy(y_pred, y_p)

def train(model, trainloader, optimizer, epoch, use_mixup=False):
    model.train()
    running_loss, e_loss = 0.0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x, y1, y2, lam = mixup(x, y)
            y_pred = model(x)
            loss = mixup_loss(y_pred, y1, y2, lam)
        else:
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        e_loss += loss.item()
        if i % 500 == 499:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss/500:.3f}')
            running_loss = 0.0
        t1 = time.time()
    return e_loss, t1 - t0

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct, total

use_mixup = True
model_name = f'cifar_net{"_mixup" if use_mixup else ""}'
model_path = f'models/{model_name}.pth'

model = resnet18().to(device)
model.load_state_dict(torch.load(model_path))

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
e_losses = []
for epoch in range(n_epochs):
    e_loss, t = train(model, train_loader, optimizer, epoch, use_mixup)
    e_losses.append(e_loss)
    print(f'[{epoch+1}] loss: {e_loss:.3f} time: {t:.3f}')
    torch.save(model.state_dict(), model_path)
    correct, total = test(model, test_loader)
    print(f'Accuracy: {100 * correct / total:.2f}%')

print(e_losses)