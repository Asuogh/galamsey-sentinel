import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms # <-- Added vision tools
import rasterio
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CUSTOM SATELLITE DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class GalamseyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []
        
        # This will trim every image to exactly 128x128 pixels
        self.cropper = transforms.CenterCrop(128)
        
        self.class_map = {
            "class_0_forest": 0,
            "class_1_galamsey": 1,
            "class_2_water": 2
        }

        for class_name, label_idx in self.class_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith('.tif') or file.endswith('.tiff'):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        with rasterio.open(img_path) as src:
            img_array = src.read()

        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        
        # Take the scissors and perfectly crop the center to 128x128
        img_tensor = self.cropper(img_tensor)
        
        return img_tensor, label

# ═══════════════════════════════════════════════════════════════════════════════
# 2. THE NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
class SentinelCNN(nn.Module):
    def __init__(self):
        super(SentinelCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ═══════════════════════════════════════════════════════════════════════════════
# 3. THE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_model():
    print("="*60)
    print("INITIALIZING GALAMSEY SENTINEL AI")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), "extraction", "patches")
    dataset = GalamseyDataset(data_path)
    
    if len(dataset) == 0:
        print("ERROR: No satellite patches found.")
        return
        
    print(f"Successfully loaded {len(dataset)} multi-spectral satellite patches.")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = SentinelCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15 
    
    print("\nStarting Training Phase...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()             
            outputs = model(images)           
            loss = criterion(outputs, labels) 
            loss.backward()                   
            optimizer.step()                  
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad(): 
            for test_images, test_labels in test_loader:
                test_outputs = model(test_images)
                _, predicted = torch.max(test_outputs.data, 1)
                total_predictions += test_labels.size(0)
                correct_predictions += (predicted == test_labels).sum().item()
                
        epoch_acc = (correct_predictions / total_predictions) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Validation Accuracy: {epoch_acc:.2f}%")
        
    model_save_path = os.path.join(os.path.dirname(__file__), "sentinel_model.pt")
    torch.save(model.state_dict(), model_save_path)
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE. AI Model saved to: {model_save_path}")
    print("="*60)

if __name__ == "__main__":
    train_model()