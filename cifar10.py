import argparse
import time
import numpy as np
import torch
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--use-mixup', '-m', action='store_true', default=False, help='use mixup')
parser.add_argument('--augment', '-a', action='store_true', default=False, help='use data augmentation')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

if args.augment:
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = args.batch_size

train_loader = torch.utils.data.DataLoader(
    CIFAR10('./data', train=True, download=True, transform=transform_train),
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    CIFAR10('./data', train=False, download=True, transform=transform_test),
    batch_size=batch_size,
    shuffle=True
)

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

def train(model, trainloader, optimizer, lr_scheduler, epoch, use_mixup=False):
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
    lr_scheduler.step()
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

model_name = f'cifar_net{"_mixup" if args.use_mixup else ""}'
model_path = f'models/{model_name}.pth'

model = resnet18().to(device)
if args.resume:
    try:
        model.load_state_dict(torch.load(model_path))
        print(f'Model {model_name} loaded.')
    except:
        print(f'Model {model_name} not found.')
        exit()
else:
    print(f'Training from {model_path} scratch.')

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

n_epochs = args.epochs
e_losses = []
for epoch in range(n_epochs):
    e_loss, t = train(model, train_loader, optimizer, lr_scheduler, epoch, use_mixup=args.use_mixup)
    e_losses.append(e_loss)
    print(f'[{epoch+1}] loss: {e_loss:.3f} time: {t:.3f}')
    torch.save(model.state_dict(), model_path)
    correct, total = test(model, test_loader)
    print(f'Accuracy: {100 * correct / total:.2f}%')

print(e_losses)