import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import numpy as np
import cv2

# 1. Model Mimarisini Buraya da Ekle (Aynı eğittiğin gibi olmalı)
class FruitBrain(nn.Module):
    def __init__(self):
        super(FruitBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 0: Apple, 1: Cucumber, 2: Bomb/Other
        )

    def forward(self, x):
        return self.network(x)

class VisionAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet Feature Extractor
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

        # --- BURASI KRİTİK: self.brain BURADA TANIMLANMALI ---
        self.brain = FruitBrain().to(self.device)
        try:
            self.brain.load_state_dict(torch.load("fruit_classifier.pth", map_location=self.device))
            print("Model başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
        self.brain.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame, x, y):
        try:
            roi = frame[max(0, y-30):min(frame.shape[0], y+30), 
                        max(0, x-30):min(frame.shape[1], x+30)]
            if roi.size == 0: return None
            input_tensor = self.preprocess(roi).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.resnet(input_tensor)
            return features.cpu().numpy().flatten()
        except:
            return None

    def classify_fruit(self, features):
        if features is None: return "unknown"
        
        feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.brain(feat_tensor) # Artık self.brain tanımlı!
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            
        THRESHOLD = 0.85
        if conf.item() < THRESHOLD: return "unknown"
        
        mapping = {0: "apple", 1: "unknown", 2: "unknown"}
        return mapping.get(idx.item(), "unknown")
    
    def find_and_classify(self, frame):
        # 1. Ekrandaki nesneleri bul (Basit bir kontur analizi ile)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Beyaz arka planda olmayan her şeyi bul (thresholding)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.find_centers = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 20: # Çok küçük gürültüleri ele
                cx, cy = x + w//2, y + h//2
                
                # 2. ResNet + MLP ile sınıflandır
                features = self.extract_features(frame, cx, cy)
                label = self.classify_fruit(features)
                
                # Konum ve sınıf bilgisini kaydet
                detections.append({'pos': (cx, cy), 'label': label})
                
        return detections