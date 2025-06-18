# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from unet_model import UNet  # You can use a standard U-Net

# Dataset and loader setup
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
train_dataset = datasets.ImageFolder('data/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model, loss, optimizer
model = UNet(n_channels=3, n_classes=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    for images, masks in train_loader:
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), 'rooftop_unet.pth')
