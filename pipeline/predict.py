import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import rasterio
import numpy as np
import json

# 1. WE MUST DEFINE THE BRAIN EXACTLY AS IT WAS TRAINED
class SentinelCNN(nn.Module):
    def __init__(self):
        super(SentinelCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(128, 3) 
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 2. THE PREDICTION FUNCTION
def analyze_image(image_path):
    # Setup the class names
    classes = {0: "FOREST", 1: "GALAMSEY", 2: "WATER"}
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"File not found: {image_path}"}))
        sys.exit(1)

    try:
        # Load the saved brain
        model_path = os.path.join(os.path.dirname(__file__), "sentinel_model.pt")
        model = SentinelCNN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval() # Put AI in "Testing" mode (no learning)

        # Load and prep the image exactly like we did in training
        cropper = transforms.CenterCrop(128)
        
        with rasterio.open(image_path) as src:
            img_array = src.read()

        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = cropper(img_tensor).unsqueeze(0) # Add a batch dimension

        # Ask the AI!
        with torch.no_grad():
            output = model(img_tensor)
            
            # Calculate percentages (confidence)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            result = classes[predicted_idx.item()]
            conf_score = round(confidence.item() * 100, 2)

        # Output a clean JSON response for the backend team
        response = {
            "status": "success",
            "prediction": result,
            "confidence": f"{conf_score}%"
        }
        print(json.dumps(response))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    # This allows the Node.js backend to pass the image path via the terminal
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please provide an image path. Usage: python predict.py <path_to_tif>"}))
        sys.exit(1)
        
    target_image = sys.argv[1]
    analyze_image(target_image)