import torch
from torchvision import transforms
from PIL import Image
from cnn_model import SimpleCNN


with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]


image_path = "./avaliation/test1.jpg"  


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("modelo_treinado.pth"))
model.eval()


image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  


with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]
    print(f"Classe prevista: {predicted_class}")
