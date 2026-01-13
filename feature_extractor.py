import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- SETTINGS ---
DATASET_PATH = "./fruit_dataset/fruits-360_100x100/fruits-360/deneme"
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS = "labels.npy"

# Mapping: 0 = Apple, 1 = Banana, 2 = Other (Cherry, Peach, etc. - treated as 'Obstacles/Bombs')
def get_label(folder_name):
    name = folder_name.lower()
    if 'apple' in name: return 0
    if 'cucumber' in name: return 1
    return 2 # Treat everything else as a 3rd class

# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
resnet.fc = nn.Identity() # Remove classifier
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features_list = []
labels_list = []

# --- EXTRACTION LOOP ---
print(f"Starting extraction on {device}...")
folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

for folder in tqdm(folders):
    label = get_label(folder)
    folder_path = os.path.join(DATASET_PATH, folder)
    
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = resnet(input_tensor).cpu().numpy().flatten()
            
            features_list.append(feat)
            labels_list.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Save to disk
np.save(OUTPUT_FEATURES, np.array(features_list))
np.save(OUTPUT_LABELS, np.array(labels_list))
print(f"\nSaved {len(features_list)} feature vectors to {OUTPUT_FEATURES}")