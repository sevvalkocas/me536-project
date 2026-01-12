import cv2
import numpy as np
import random
import sys
from vision_module import VisionAI
from search_module import PathFinder

# --- 1. AYARLAR ---
WIDTH, HEIGHT = 800, 600
GRAVITY = 0.15
SPAWN_RATE = 0.04
WAIT_THRESHOLD = 100
BLADE_SPEED = 5      # Meyveleri yukarıda yakalamak için hız artırıldı
TRAIL_LENGTH = 10

# --- 2. BAŞLATMA ---
vision = VisionAI()
planner = PathFinder(WIDTH, HEIGHT, 25)

try:
    apple_img = cv2.resize(cv2.imread("apple.jpg"), (60, 60))
    banana_img = cv2.resize(cv2.imread("banana.jpg"), (60, 60))
    knife_raw = cv2.imread("knife_transparent.png", cv2.IMREAD_UNCHANGED)
    if knife_raw is None:
        knife_raw = cv2.imread("knife.png", cv2.IMREAD_UNCHANGED)
    knife_img = cv2.resize(knife_raw, (50, 50))
    fruit_assets = {'apple': apple_img, 'banana': banana_img}
except Exception as e:
    print(f"HATA: Görseller yüklenemedi: {e}")
    sys.exit()

def reset_game():
    return [WIDTH//2, HEIGHT-50], [], [], False

blade_pos, blade_history, fruits, game_over = reset_game()

# --- 4. ANA DÖNGÜ ---
while True:
    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

    if not game_over:
        if random.random() < SPAWN_RATE:
            f_type = 'apple' if random.random() < 0.75 else 'banana'
            fruits.append([random.randint(100, WIDTH-100), -50, random.uniform(-1.2, 1.2), 0, f_type])

        target = None
        apples = [f for f in fruits if f[4] == 'apple' and f[1] > WAIT_THRESHOLD]
        bananas = [f for f in fruits if f[4] == 'banana']
        
        if apples:
            f = apples[0]
            target = (int(f[0] + f[2]*5), int(f[1] + f[3]*5 + 0.5*GRAVITY*25))

        path = planner.a_star(tuple(blade_pos), target, bananas)
        if path:
            step = min(len(path) - 1, BLADE_SPEED)
            blade_pos[0], blade_pos[1] = path[step]

        blade_history.append(tuple(blade_pos))
        if len(blade_history) > TRAIL_LENGTH: blade_history.pop(0)
        for i in range(1, len(blade_history)):
            cv2.line(frame, blade_history[i-1], blade_history[i], (180, 180, 180), i)

        # GÜVENLİK KONTROLÜ: Muz yakınındayken kesme
        safe_to_slice = True
        for f in fruits:
            if f[4] == 'banana':
                if np.linalg.norm(np.array(blade_pos) - np.array([f[0], f[1]])) < 80:
                    safe_to_slice = False
                    break

        # Fizik ve Çizim
        for i in range(len(fruits)-1, -1, -1):
            fruits[i][1] += fruits[i][3]
            fruits[i][3] += GRAVITY
            cx, cy, f_real_type = int(fruits[i][0]), int(fruits[i][1]), fruits[i][4]

            if 30 < cx < WIDTH-30 and -50 < cy < HEIGHT-30:
                if cy > 30: frame[cy-30:cy+30, cx-30:cx+30] = fruit_assets[f_real_type]

            dist = np.linalg.norm(np.array(blade_pos) - np.array([cx, cy]))

            if dist < 65: 
                # --- FEATURE EXTRACTION KISMI ---
                features = vision.extract_features(frame, cx, cy)
                
                if features is not None:
                    print(f"Extracted features for {f_real_type} at ({cx}, {cy}) first 3 values: {features[:3]}")
                    # Analiz edildiğini belirtmek için mavi kare
                    cv2.rectangle(frame, (cx-35, cy-35), (cx+35, cy+35), (255, 0, 0), 2)
                    
                    if f_real_type == 'apple':
                        if safe_to_slice:
                            cv2.circle(frame, (cx, cy), 50, (255, 255, 255), -1)
                            cv2.putText(frame, "AI DECISION: APPLE (SLICE)", (cx-80, cy-50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
                            fruits.pop(i)
                            continue
                        else:
                            cv2.putText(frame, "DANGER: BANANA NEARBY!", (cx-80, cy-50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Muz Çarpışma Kontrolü
            if dist < 40 and f_real_type == 'banana':
                game_over = True; break
            
            if fruits[i][1] > HEIGHT: fruits.pop(i)

        # Bıçağı Çiz
        bx, by = int(blade_pos[0]), int(blade_pos[1])
        y1, y2, x1, x2 = max(0, by-25), min(HEIGHT, by+25), max(0, bx-25), min(WIDTH, bx+25)
        k_crop = knife_img[0:(y2-y1), 0:(x2-x1)]
        if k_crop.shape[2] == 4:
            alpha = k_crop[:,:,3]/255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha*k_crop[:,:,c] + (1-alpha)*frame[y1:y2, x1:x2, c])
        else: frame[y1:y2, x1:x2] = k_crop

    else:
        # --- OYUN SONU EKRANI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "GAME OVER", (WIDTH//2-180, HEIGHT//2-50), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 5)
        cv2.putText(frame, "[R] RESTART  [Q] QUIT", (WIDTH//2-180, HEIGHT//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Fruit Ninja AI", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    if key == ord('r') and game_over:
        blade_pos, blade_history, fruits, game_over = reset_game()

cv2.destroyAllWindows()