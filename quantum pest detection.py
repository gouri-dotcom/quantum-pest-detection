import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pennylane as qml
from pennylane import qnn
from PIL import Image

# Define paths to the IP102 dataset
base_dir = 'D:/VIT- STUDY MATERIAL/QCNN IN PEST DETECTION- PROJECT/pest analysis code/Classification/ip102_v1.1'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
classes_file = 'D:/VIT- STUDY MATERIAL/QCNN IN PEST DETECTION- PROJECT/pest analysis code/Classification/classes.txt'

# Load class names from classes file
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f]

# Step 2: Define QCNN Model with PennyLane and PyTorch
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def qcnn_circuit(inputs, weights):
    for i in range(num_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (3, num_qubits, 3)}
qlayer = qnn.TorchLayer(qcnn_circuit, weight_shapes)

class HybridQCNN(nn.Module):
    def _init_(self):
        super(HybridQCNN, self)._init_()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_qubits)
        self.q_layer = qlayer
        self.fc2 = nn.Linear(num_qubits, len(class_names))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        quantum_outputs = [self.q_layer(x[i]) for i in range(x.size(0))]
        x = torch.stack(quantum_outputs)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Step 3: Prepare Dataloaders with Grayscale and Normalization
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder(root=train_dir, transform=transform)
val_ds = datasets.ImageFolder(root=val_dir, transform=transform)
test_ds = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Step 4: Train the QCNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridQCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Step 5: Evaluate the QCNN Model with Accuracy Metrics
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate and display metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall: {recall * 100:.2f}%")
print(f"Test F1 Score: {f1 * 100:.2f}%")

# Step 6: Predict a Single Image
def predict_image(image_path, model, class_names, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    print(f"Predicted class: {predicted_class}")

# Example usage of predict_image function
image_path = "D:/VIT- STUDY MATERIAL/QCNN IN PEST DETECTION- PROJECT/pest analysis code/Classification/ip102_v1.1/images/01049.jpg"
predict_image(image_path, model, class_names, device)