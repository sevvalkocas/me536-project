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

# --- 2. SİSTEMLERİ BAŞLAT ---
# VisionAI artık hem 'find' (bulma) hem 'classify' (tanıma) işini yapacak
vision = VisionAI() 
planner = PathFinder(WIDTH, HEIGHT, 25)
search_planner = SearchPlanner(WIDTH, HEIGHT, 25)

try:
    apple_img = cv2.resize(cv2.imread("apple.jpeg"), (60, 60))
    bomb_img = cv2.resize(cv2.imread("bomb.jpeg"), (60, 60))
    knife_raw = cv2.imread("no_bg_knife.png", cv2.IMREAD_UNCHANGED)
    knife_img = cv2.resize(knife_raw, (50, 50))
    fruit_assets = {'apple': apple_img, 'bomb': bomb_img}
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
            dist = np.linalg.norm(np.array(blade_pos) - np.array([fruits[i][0], fruits[i][1]]))
            real_type = fruits[i][4]
            
            if dist < 25:
                features = vision.extract_features(frame, int(fruits[i][0]), int(fruits[i][1]))
                prediction = vision.classify_fruit(features)

                if prediction in ["apple", "banana", "cucumber", "eggplant", "orange"] and real_type != 'bomb':
                    cv2.circle(frame, (int(fruits[i][0]), int(fruits[i][1])), 50, (255, 255, 255), -1)
                    fruits.pop(i)
                elif real_type == 'bomb':
                    game_over = True

        draw_blade_and_trail(frame, blade_pos, blade_history, knife_img, TRAIL_LENGTH)

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