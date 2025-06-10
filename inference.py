import torch
from torchvision import transforms
from PIL import Image
from cnn_model import SimpleCNN

# Carrega as classes do arquivo classes.txt
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Caminho da imagem a ser testada
image_path = "./avaliation/test1.jpg"  # Substitua pelo caminho real da imagem

# Transformações da imagem (iguais às usadas no treino)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Inicializa e carrega o modelo
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("modelo_treinado.pth"))
model.eval()

# Abre e transforma a imagem
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Adiciona dimensão de batch

# Faz a predição
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]
    print(f"Classe prevista: {predicted_class}")
