
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_model import SimpleCNN

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder('dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCNN(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load('modelo_treinado.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Acurácia no conjunto de teste: {100 * correct / total:.2f}%")
