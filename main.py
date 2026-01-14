import cv2
import numpy as np
import sys
from vision_module import VisionAI
from search_module import PathFinder, SearchPlanner
from helper import spawn_and_physics, draw_blade_and_trail, draw_detection_boxes, find_safe_spot

# --- 1. AYARLAR ---
WIDTH, HEIGHT = 800, 600
WAIT_THRESHOLD = 200   # AI'nın nesneyi algılayıp tepki vermesi için gereken dikey sınır
BLADE_SPEED = 5      # AI bıçağının hızı
TRAIL_LENGTH = 10
fruit_values = {
    'apple': 10, 'orange': 15, 'cucumber': 20, 
    'eggplant': 25, 'banana': 30, 'new_fruit': 50
}

# --- 2. SİSTEMLERİ BAŞLAT ---
# VisionAI artık hem 'find' (bulma) hem 'classify' (tanıma) işini yapacak
vision = VisionAI() 
planner = PathFinder(WIDTH, HEIGHT, 25)
search_planner = SearchPlanner(WIDTH, HEIGHT, 25)

try:
    apple_img = cv2.resize(cv2.imread("apple.jpeg"), (60, 60))
    banana_img = cv2.resize(cv2.imread("banana.jpg"), (60, 60))
    cucumber_img = cv2.resize(cv2.imread("cucumber.jpeg"), (60, 60))
    eggplant_img = cv2.resize(cv2.imread("eggplant.jpeg"), (60, 60))
    orange_img = cv2.resize(cv2.imread("orange.jpeg"), (60, 60))
    # watermelon_img = cv2.resize(cv2.imread("cherry.jpeg"), (60, 60))
    bomb_img = cv2.resize(cv2.imread("bomb.jpeg"), (50, 50))
    knife_raw = cv2.imread("no_bg_knife.png", cv2.IMREAD_UNCHANGED)
    knife_img = cv2.resize(knife_raw, (50, 50))
    fruit_assets = {'apple': apple_img, 'banana': banana_img, 'cucumber': cucumber_img, 'eggplant': eggplant_img, 'orange': orange_img, 'bomb': bomb_img}
except Exception as e:
    print(f"HATA: Görseller yüklenemedi: {e}")
    sys.exit()

def reset_game():
    return [WIDTH//2, HEIGHT-50], [], [], False, 0

blade_pos, blade_history, fruits, game_over, score = reset_game()

# --- 3. ANA OYUN VE AI DÖNGÜSÜ ---
while True:
    # Boş bir tuval (Ekran) oluştur
    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    if not game_over:
        # --- A. OYUN MOTORU (Fizik ve Spawn) ---
        spawn_and_physics(fruits, frame, fruit_assets)

        # --- B. GAMER AI (Algılama ve Karar) ---
        # AI ekranın görüntüsünü alır ve içindeki her şeyi 'görür'
        detected_objects = vision.find_and_classify(frame)
        draw_detection_boxes(frame, detected_objects)
        ai_target = None
        ai_obstacles = []

        for obj in detected_objects:
            pos = obj['pos']
            label = obj['label']
            if label in ["apple", "banana", "cucumber", "eggplant", "orange"]:
                if pos[1] > WAIT_THRESHOLD:
                    # En yakın/en üstteki hedefi belirle
                    if ai_target is None or pos[1] > ai_target[1]:
                        ai_target = pos
            elif label == "avoid":
                ai_obstacles.append([pos[0], pos[1], "avoid"])

        # Yol Planlama
        # path = planner.a_star(tuple(blade_pos), ai_target, ai_obstacles)
        path = search_planner.search_path(tuple(blade_pos), ai_target, ai_obstacles)
        if path:
            step = min(len(path) - 1, BLADE_SPEED)
            blade_pos[0], blade_pos[1] = path[step]

        # --- C. ETKİLEŞİM VE ÇARPIŞMA ---
        # AI hedefe ulaştıysa kesme işlemi
        for i in range(len(fruits)-1, -1, -1):
            fx, fy, _, _, real_type = fruits[i]
            dist = np.linalg.norm(np.array(blade_pos) - np.array([fx, fy]))
            
            # 1. KESME MANTIĞI
            if dist < 45:
                features = vision.extract_features(frame, int(fx), int(fy))
                prediction = vision.classify_fruit(features)
                
                # AI tanıyorsa VEYA hafızasından hatırlıyorsa KES
                if prediction != "avoid" or vision.is_in_memory(features):
                    if real_type != 'bomb':
                        label = prediction if prediction != "avoid" else "new_fruit"
                        score += fruit_values.get(label, 10)
                        print(f"KESİLDİ! Tip: {label} | Puan: {score}")
                        cv2.circle(frame, (int(fx), int(fy)), 50, (255, 255, 255), -1)
                        fruits.pop(i)
                    else:
                        game_over = True # Bombayı kestik!

        # --- D. DÜŞENLERİ TAKİP ET (ÖĞRENME BURADA) ---
        for i in range(len(fruits)-1, -1, -1):
            if fruits[i][1] > HEIGHT: # Meyve ekranın altına düştüyse
                real_type = fruits[i][4]
                
                if real_type != 'bomb':
                    # Bilmediğimiz bir meyve düştü, puan kaybet ve ÖĞREN!
                    score -= 5
                    print(f"Meyve Kaçırıldı! -5 Puan. Güncel Score: {score}")
                    
                    # Bu meyvenin özelliklerini çıkar ve hafızaya ekle
                    missed_features = vision.extract_features(frame, int(fruits[i][0]), int(fruits[i][1]))
                    if missed_features is not None:
                        vision.missed_fruits_memory.append(missed_features)
                        print("YENİ BİR MEYVE ÖĞRENİLDİ: Gelecekte bunu keseceğim!")
                
                fruits.pop(i)
        cv2.putText(frame, f"SCORE: {score}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (50, 50, 50), 2)
        cv2.putText(frame, f"MEM: {len(vision.missed_fruits_memory)} items", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        draw_blade_and_trail(frame, blade_pos, blade_history, knife_img, TRAIL_LENGTH)

    else:
        # --- OYUN BİTTİ EKRANI ---
        overlay = frame.copy()
        # Ekranı karart (Siyah transparan katman)
        cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        # "GAME OVER" Başlığı
        cv2.putText(frame, "AI PERFORMANCE REPORT", (WIDTH//2-220, HEIGHT//2-100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        # Skor Bilgileri
        cv2.putText(frame, f"FINAL SCORE: {score}", (WIDTH//2-120, HEIGHT//2-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # Hafıza (Öğrenme) Bilgisi
        learned_count = len(vision.missed_fruits_memory)
        cv2.putText(frame, f"NEW SPECIES LEARNED: {learned_count}", (WIDTH//2-160, HEIGHT//2+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        # Kontroller
        cv2.putText(frame, "Press [R] to Restart or [Q] to Quit", (WIDTH//2-180, HEIGHT//2+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Autonomous Gamer AI", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    if key == ord('r') and game_over: blade_pos, blade_history, fruits, game_over, score = reset_game()

cv2.destroyAllWindows()