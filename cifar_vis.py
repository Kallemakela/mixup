#%%
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
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
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
# %%
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#%%
#%%
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
#%%
PATH = 'models/cifar_net_mixup.pth'
#%%
model = resnet18()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    t0 = time.time()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels1, labels2, lam = mixup(inputs, labels)
        outputs = model(inputs)
        loss = mixup_loss(outputs, labels1, labels2, lam)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if i % 500 == 499:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss/500:.3f}')
            running_loss = 0.0
    
    t1 = time.time()
    torch.save(model.state_dict(), PATH)
    print(f'Epoch {epoch+1} completed in {t1-t0:.2f} seconds. Model saved to {PATH}.')
#%%
model = resnet18()
model.load_state_dict(torch.load(PATH))
model.eval()
''
#%%
dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('Ground truth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))
#%%
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
#%%