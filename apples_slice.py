import cv2
import numpy as np
import heapq
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- 1. MODEL VE CİHAZ AYARI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()
resnet.eval()

# --- 2. AYARLAR ---
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 25
GRAVITY = 0.15      
SPAWN_RATE = 0.04   # Daha fazla elma için spawn artırıldı
WAIT_THRESHOLD = 250 # Elmaların iyice düşmesini bekle

blade_pos = [WIDTH//2, HEIGHT-50]
fruits = [] 

try:
    apple_img = cv2.resize(cv2.imread("apple.jpg"), (60, 60))
    banana_img = cv2.resize(cv2.imread("banana.jpg"), (60, 60))
    fruit_assets = {'apple': apple_img, 'banana': banana_img}
except:
    print("Görseller bulunamadı!")
    exit()

def a_star_search(start, goal):
    start_node = (int(start[0] // GRID_SIZE * GRID_SIZE), int(start[1] // GRID_SIZE * GRID_SIZE))
    goal_node = (int(goal[0] // GRID_SIZE * GRID_SIZE), int(goal[1] // GRID_SIZE * GRID_SIZE))
    open_list = []
    heapq.heappush(open_list, (0, start_node))
    came_from = {}; g_score = {start_node: 0}
    
    while open_list:
        current = heapq.heappop(open_list)[1]
        if np.linalg.norm(np.array(current) - np.array(goal_node)) < GRID_SIZE:
            path = []
            while current in came_from:
                path.append(current); current = came_from[current]
            return path[::-1]
        for dx, dy in [(0, GRID_SIZE), (0, -GRID_SIZE), (GRID_SIZE, 0), (-GRID_SIZE, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT:
                tg = g_score[current] + GRID_SIZE
                if neighbor not in g_score or tg < g_score[neighbor]:
                    g_score[neighbor] = tg
                    f = tg + np.linalg.norm(np.array(neighbor) - np.array(goal_node))
                    came_from[neighbor] = current
                    heapq.heappush(open_list, (f, neighbor))
    return []

# --- 4. ANA DÖNGÜ ---
while True:
    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    
    if random.random() < SPAWN_RATE:
        f_type = random.choice(['apple', 'banana'])
        fruits.append([random.randint(100, WIDTH-100), -50, random.uniform(-1.0, 1.0), 0, f_type])

    # HEDEFLEME: En verimli noktayı bul
    target_point = None
    apples_in_air = [f for f in fruits if f[4] == 'apple' and f[1] > WAIT_THRESHOLD]
    
    if len(apples_in_air) >= 2:
        # İki elmanın ortasını hedefle (Combo denemesi)
        a1, a2 = apples_in_air[0], apples_in_air[1]
        avg_x = (a1[0] + a2[0]) / 2
        avg_y = (a1[1] + a2[1]) / 2
        target_point = (int(avg_x), int(avg_y))
    elif len(apples_in_air) == 1:
        target_point = (int(apples_in_air[0][0]), int(apples_in_air[0][1]))

    path = a_star_search(tuple(blade_pos), target_point) if target_point else []
    
    if path:
        # Combo hızı: Hedefe daha agresif git
        step = min(len(path) - 1, 2)
        blade_pos[0], blade_pos[1] = path[step]

    # Çizim ve Kesme
    for i in range(len(fruits)-1, -1, -1):
        fruits[i][1] += fruits[i][3]; fruits[i][3] += GRAVITY
        cx, cy = int(fruits[i][0]), int(fruits[i][1])
        
        if 30 < cx < WIDTH-30 and 30 < cy < HEIGHT-30:
            frame[cy-30:cy+30, cx-30:cx+30] = fruit_assets[fruits[i][4]]
        
        dist = np.linalg.norm(np.array(blade_pos) - np.array([fruits[i][0], fruits[i][1]]))
        if dist < 50 and fruits[i][4] == 'apple':
            # Kesik Efekti
            cv2.line(frame, (cx-40, cy-40), (cx+40, cy+40), (0, 0, 255), 3)
            print("KESİLDİ!")
            fruits.pop(i)
        elif fruits[i][1] > HEIGHT:
            fruits.pop(i)

    cv2.circle(frame, tuple(blade_pos), 10, (0, 0, 0), -1)
    cv2.imshow("Multi-Cut Combo Mode", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cv2.destroyAllWindows()