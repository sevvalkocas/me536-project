import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import numpy as np
import cv2

class VisionAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ResNet50 kullanarak özellik çıkarımı yapıyoruz
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.fc = torch.nn.Identity() # Son katmanı kaldırarak 2048 boyutlu vektör alıyoruz
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame, x, y):
        # Meyvenin olduğu bölgeyi kırp (ROI - Region of Interest)
        try:
            roi = frame[max(0, y-30):min(frame.shape[0], y+30), 
                        max(0, x-30):min(frame.shape[1], x+30)]
            if roi.size == 0: return None
            
            input_tensor = self.preprocess(roi).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(input_tensor)
            return features.cpu().numpy().flatten()
        except:
            return None
