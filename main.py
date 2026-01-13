import cv2
import numpy as np
import random
import sys
import torch
from vision_module import VisionAI
from search_module import PathFinder

# --- 1. AYARLAR ---
WIDTH, HEIGHT = 800, 600
GRAVITY = 0.15
SPAWN_RATE = 0.05
WAIT_THRESHOLD = 100   # AI'nın nesneyi algılayıp tepki vermesi için gereken dikey sınır
BLADE_SPEED = 5      # AI bıçağının hızı
TRAIL_LENGTH = 10

# --- 2. SİSTEMLERİ BAŞLAT ---
# VisionAI artık hem 'find' (bulma) hem 'classify' (tanıma) işini yapacak
vision = VisionAI() 
planner = PathFinder(WIDTH, HEIGHT, 25)

try:
    apple_img = cv2.resize(cv2.imread("apple.jpg"), (60, 60))
    banana_img = cv2.resize(cv2.imread("banana.jpg"), (60, 60))
    knife_raw = cv2.imread("knife_transparent.png", cv2.IMREAD_UNCHANGED)
    if knife_raw is None: knife_raw = cv2.imread("knife.jpg", cv2.IMREAD_UNCHANGED)
    knife_img = cv2.resize(knife_raw, (50, 50))
    fruit_assets = {'apple': apple_img, 'banana': banana_img}
except Exception as e:
    print(f"HATA: Görseller yüklenemedi: {e}")
    sys.exit()

def reset_game():
    return [WIDTH//2, HEIGHT-50], [], [], False

blade_pos, blade_history, fruits, game_over = reset_game()

# --- 3. ANA OYUN VE AI DÖNGÜSÜ ---
while True:
    # Boş bir tuval (Ekran) oluştur
    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

    if not game_over:
        # --- A. OYUN MOTORU (Fizik ve Spawn) ---
        if random.random() < SPAWN_RATE:
            f_type = 'apple' if random.random() < 0.70 else 'banana'
            fruits.append([random.randint(100, WIDTH-100), -50, random.uniform(-1.2, 1.2), 0, f_type])

        for i in range(len(fruits)-1, -1, -1):
            fruits[i][1] += fruits[i][3]
            fruits[i][3] += GRAVITY
            cx, cy, real_type = int(fruits[i][0]), int(fruits[i][1]), fruits[i][4]
            
            if 30 < cx < WIDTH-30 and -50 < cy < HEIGHT-30:
                if cy > 30: frame[cy-30:cy+30, cx-30:cx+30] = fruit_assets[real_type]
            
            # Ekrandan çıkan meyveleri sil
            if fruits[i][1] > HEIGHT + 50: fruits.pop(i)

        # --- B. GAMER AI (Algılama ve Karar) ---
        # AI ekranın görüntüsünü alır ve içindeki her şeyi 'görür'
        detected_objects = vision.find_and_classify(frame)
        
        ai_target = None
        ai_obstacles = []

        for obj in detected_objects:
            pos = obj['pos']
            label = obj['label'] # 'apple', 'cucumber', 'avoid'
            
            # Tespit edilenleri ekranda kutu içine al (AI Gözü)
            color = (0, 255, 0) if label != "avoid" else (0, 0, 255)
            cv2.rectangle(frame, (pos[0]-35, pos[1]-35), (pos[0]+35, pos[1]+35), color, 1)
            cv2.putText(frame, label.upper(), (pos[0]-35, pos[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if label in ["apple", "cucumber"]:
                if pos[1] > WAIT_THRESHOLD:
                    # En yakın/en üstteki hedefi belirle
                    if ai_target is None or pos[1] > ai_target[1]:
                        ai_target = pos
            elif label == "avoid":
                ai_obstacles.append([pos[0], pos[1], "banana"])
        
        # A* Yol Planlama
        path = planner.a_star(tuple(blade_pos), ai_target, ai_obstacles)
        if path:
            step = min(len(path) - 1, BLADE_SPEED)
            blade_pos[0], blade_pos[1] = path[step]

        # --- C. ETKİLEŞİM VE ÇARPIŞMA ---
        # AI hedefe ulaştıysa kesme işlemi
        for i in range(len(fruits)-1, -1, -1):
            dist = np.linalg.norm(np.array(blade_pos) - np.array([fruits[i][0], fruits[i][1]]))
            real_type = fruits[i][4]
            
            if dist < 45:
                # AI'ın bu nesne için tahmini neydi?
                features = vision.extract_features(frame, int(fruits[i][0]), int(fruits[i][1]))
                prediction = vision.classify_fruit(features)
                
                if prediction in ["apple", "cucumber"] and real_type != 'banana':
                    cv2.circle(frame, (int(fruits[i][0]), int(fruits[i][1])), 50, (255, 255, 255), -1)
                    fruits.pop(i)
                elif real_type == 'banana':
                    game_over = True

        # Bıçağı ve İzi Çiz
        blade_history.append(tuple(blade_pos))
        if len(blade_history) > TRAIL_LENGTH: blade_history.pop(0)
        for i in range(1, len(blade_history)):
            cv2.line(frame, blade_history[i-1], blade_history[i], (150, 150, 150), i)
        
        # Bıçak görselini yerleştir (PNG Overlay)
        bx, by = int(blade_pos[0]), int(blade_pos[1])
        y1, y2, x1, x2 = max(0, by-25), min(HEIGHT, by+25), max(0, bx-25), min(WIDTH, bx+25)
        k_crop = knife_img[0:(y2-y1), 0:(x2-x1)]
        if k_crop.shape[2] == 4:
            alpha = k_crop[:,:,3]/255.0
            for c in range(3): frame[y1:y2, x1:x2, c] = (alpha*k_crop[:,:,c] + (1-alpha)*frame[y1:y2, x1:x2, c])
        else: frame[y1:y2, x1:x2] = k_crop

    else:
        # Game Over Ekranı
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "AI GAME OVER", (WIDTH//2-180, HEIGHT//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 3)
        cv2.putText(frame, "[R] Restart  [Q] Quit", (WIDTH//2-150, HEIGHT//2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Autonomous Gamer AI", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    if key == ord('r') and game_over: blade_pos, blade_history, fruits, game_over = reset_game()

cv2.destroyAllWindows()