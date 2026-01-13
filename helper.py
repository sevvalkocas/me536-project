import random
import cv2

WIDTH, HEIGHT = 800, 600
GRAVITY = 0.15
SPAWN_RATE = 0.05

def spawn_and_physics(fruits, frame, fruit_assets):
    """Meyveleri oluşturur ve aşağı düşürür."""
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

def draw_detection_boxes(frame, detected_objects):
    """AI'nın tespit ettiği nesnelerin etrafına kutu ve etiket çizer."""
    for obj in detected_objects:
        pos = obj['pos']
        label = obj['label'] # 'apple', 'cucumber', 'avoid'
        
        # Renk Belirleme: Hedefler yeşil, engeller (avoid) kırmızı
        color = (0, 255, 0) if label != "avoid" else (0, 0, 255)
        
        # Kutu Çizimi (Nesnenin merkezinden dışarı doğru 35 piksellik bir kare)
        cv2.rectangle(frame, 
                      (pos[0]-35, pos[1]-35), 
                      (pos[0]+35, pos[1]+35), 
                      color, 1)
        
        # Etiket Yazımı
        cv2.putText(frame, 
                    label.upper(), 
                    (pos[0]-35, pos[1]-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
def draw_blade_and_trail(frame, blade_pos, blade_history, knife_img, trail_length=10):
    """Bıçağın hareket izini ve kendisini (PNG) ekrana çizer."""
    # 1. İz Geçmişini Güncelle
    blade_history.append(tuple(blade_pos))
    if len(blade_history) > trail_length:
        blade_history.pop(0)

    # 2. Hareketi İzi (Trail) Çizimi
    for i in range(1, len(blade_history)):
        # i değeri kalınlık olarak kullanılıyor, iz gittikçe kalınlaşır/incelir efekti verir
        cv2.line(frame, blade_history[i-1], blade_history[i], (150, 150, 150), i)
    
    # 3. Bıçağı Çiz (PNG Overlay Mantığı)
    bx, by = int(blade_pos[0]), int(blade_pos[1])
    h, w = knife_img.shape[:2]
    
    # Ekran sınırlarını aşmamak için koordinat hesaplama
    y1, y2 = max(0, by - h//2), min(frame.shape[0], by + h//2)
    x1, x2 = max(0, bx - w//2), min(frame.shape[1], bx + w//2)
    
    k_crop = knife_img[0:(y2-y1), 0:(x2-x1)]
    
    if k_crop.shape[2] == 4: # PNG Şeffaflık (Alpha) Kontrolü
        alpha = k_crop[:, :, 3] / 255.0
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha * k_crop[:, :, c] + 
                                      (1 - alpha) * frame[y1:y2, x1:x2, c])
    else:
        frame[y1:y2, x1:x2] = k_crop[:, :, :3]

def find_safe_spot(obstacles, width, height):
    """Engellerden uzak, en güvenli bölgeyi belirler."""
    # Ekranı 3 dikey bölgeye ayır
    zones = {
        "left": (WIDTH // 6, HEIGHT - 100),
        "center": (WIDTH // 2, HEIGHT - 100),
        "right": (5 * WIDTH // 6, HEIGHT - 100)
    }
    
    # Her bölgedeki engel sayısını say
    zone_risk = {"left": 0, "center": 0, "right": 0}
    
    for obs in obstacles:
        x_pos = obs[0]
        if x_pos < WIDTH // 3:
            zone_risk["left"] += 1
        elif x_pos < 2 * WIDTH // 3:
            zone_risk["center"] += 1
        else:
            zone_risk["right"] += 1
            
    # En az riskli bölgeyi seç
    safe_zone = min(zone_risk, key=zone_risk.get)
    return zones[safe_zone]