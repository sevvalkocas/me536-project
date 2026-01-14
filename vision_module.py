import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import numpy as np
import cv2

class FruitBrain(nn.Module):
    def __init__(self):
        super(FruitBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5) 
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

        self.missed_fruits_memory = [] # Puan kaybettiğimiz meyvelerin feature vektörleri
        self.similarity_threshold = 0.92 # Benzerlik eşiği (Cosine Similarity)
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

    def extract_features(self, frame, cx, cy):
        # 1. Meyvenin olduğu bölgeyi kırp (ROI)
        size = 50 # 100 çok büyük olabilir, 50-60 genellikle yeterlidir
        y1, y2 = max(0, cy-size), min(frame.shape[0], cy+size)
        x1, x2 = max(0, cx-size), min(frame.shape[1], cx+size)
        patch = frame[y1:y2, x1:x2].copy()
        
        if patch.size == 0: return None

        # 2. HIZLI MASKELEME (Rembg Yerine)
        # Arka plan beyaz (255) olduğu için beyaza yakın her şeyi şeffaf/beyaz kabul edeceğiz
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Arka plan beyazsa (255), 240'tan büyük değerleri maskele (arka planı seç)
        # Eğer arka planın siyahsa cv2.THRESH_BINARY kullanmalısın
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Maskelenen (arka plan) kısımları tam beyaza boya (Modelin için temizlik)
        patch[mask == 255] = [255, 255, 255]
        
        # 3. ResNet Ön İşleme ve Özellik Çıkarma
        # OpenCV (BGR) -> RGB çevrimi yapmayı unutma
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(patch_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(input_tensor)
        
        return features.cpu().numpy().flatten()
        
    def classify_fruit(self, features):
        if features is None: return "avoid"
        feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.brain(feat_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
        
        if conf.item() < 0.75: return "avoid" # Güven düşükse dokunma

        # Eğitimdeki get_label fonksiyonunla aynı sırayı takip etmeli:
        # 0: Apple, 1: Banana, 2: Cucumber, 3: Eggplant, 4: Orange, 5: Other
        mapping = {
            0: "apple", 
            1: "banana", # Bu bir engel (avoid)
            2: "cucumber", 
            3: "eggplant", 
            4: "orange", 
            5: "avoid"
        }
        return mapping.get(idx.item(), "avoid")
    
    def find_and_classify(self, frame):
        # 1. Ekrandaki nesneleri bul (Basit bir kontur analizi ile)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Beyaz arka planda olmayan her şeyi bul (thresholding)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
    
    def is_in_memory(self, current_features):
        """Şu anki objenin hafızadaki kaçırılmış meyvelere benzeyip benzemediğini kontrol eder."""
        if not self.missed_fruits_memory:
            return False
        
        current_feat_tensor = torch.tensor(current_features).to(self.device)
        
        for remembered_feat in self.missed_fruits_memory:
            remembered_tensor = torch.tensor(remembered_feat).to(self.device)
            # Cosine Similarity (Vektörler arası benzerlik) hesapla
            similarity = F.cosine_similarity(current_feat_tensor.unsqueeze(0), 
                                             remembered_tensor.unsqueeze(0))
            if similarity.item() > self.similarity_threshold:
                return True
        return False