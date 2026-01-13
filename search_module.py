import numpy as np
import heapq

class PathFinder:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size

    def a_star(self, start, goal, obstacles):
        if not goal: return []
        
        # 1. Başlangıç ve Hedef Noktalarını Grid Sistemine Oturt
        start_node = (int(start[0] // self.grid_size * self.grid_size), 
                    int(start[1] // self.grid_size * self.grid_size))
        goal_node = (int(goal[0] // self.grid_size * self.grid_size), 
                    int(goal[1] // self.grid_size * self.grid_size))
        
        open_list = []
        heapq.heappush(open_list, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        
        # 2. Dinamik Engel Tahmini (Düşen Muzlar)
        obstacle_nodes = set()
        for obs in obstacles:
            ox = int(obs[0] // self.grid_size * self.grid_size)
            oy = int(obs[1] // self.grid_size * self.grid_size)
            
            # Muzun olduğu yer ve çevresindeki 1 birimlik alanı kapat
            for dx in [-self.grid_size, 0, self.grid_size]:
                for dy in [-self.grid_size, 0, self.grid_size]:
                    obstacle_nodes.add((ox + dx, oy + dy))
            
            # GELECEK TAHMİNİ: Muz aşağı düştüğü için altındaki 2 hücreyi de kapat
            # Bu sayede bıçak muzun "altından" geçmeye çalışmaz, çarpmayı önler
            for future_y in range(1, 3):
                obstacle_nodes.add((ox, oy + (future_y * self.grid_size)))

        # 3. A* Arama Döngüsü
        while open_list:
            current = heapq.heappop(open_list)[1]
            
            # Hedefe yeterince yaklaştık mı?
            if np.linalg.norm(np.array(current) - np.array(goal_node)) < self.grid_size:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            # 4 Komşu Hücreyi Kontrol Et
            for dx, dy in [(0, self.grid_size), (0, -self.grid_size), 
                        (self.grid_size, 0), (-self.grid_size, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Ekran sınırları kontrolü
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    
                    # EĞER HÜCRE ENGEL İÇİNDEYSE BU ADIMI ATLA (Aşılmaz Duvar)
                    if neighbor in obstacle_nodes:
                        continue 

                    tentative_g_score = g_score[current] + self.grid_size
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        # Heuristic: Manhattan Distance (Daha hızlı hesaplama için)
                        h_score = abs(neighbor[0] - goal_node[0]) + abs(neighbor[1] - goal_node[1])
                        f_score = tentative_g_score + h_score
                        
                        came_from[neighbor] = current
                        heapq.heappush(open_list, (f_score, neighbor))
        
        return [] # Yol bulunamadı
    
class SearchPlanner:
    def __init__(self, width, height, grid_size=25):
        self.width = width
        self.height = height
        self.grid_size = grid_size

    def search_path(self, start, goal, obstacles):
        """Greedy Best-First Search kullanarak en iyi yolu arar."""
        if goal is None: return []

        # Başlangıç ve hedefi grid hücresine dönüştür
        start_node = (int(start[0] // self.grid_size), int(start[1] // self.grid_size))
        goal_node = (int(goal[0] // self.grid_size), int(goal[1] // self.grid_size))

        # Öncelik kuyruğu (Priority Queue) - (Heuristic, Koordinat)
        open_list = []
        heapq.heappush(open_list, (0, start_node))
        
        came_from = {}
        visited = {start_node}

        # Engelleri set olarak tut (Hızlı erişim için)
        obstacle_cells = set()
        for obs in obstacles:
            ox, oy = int(obs[0] // self.grid_size), int(obs[1] // self.grid_size)
            # Muzun kendisini ve altındaki 2 hücreyi (düşeceği yer) kapat
            for buffer_y in range(0, 3):
                obstacle_cells.add((ox, oy + buffer_y))
                # Yatayda da biraz güvenlik payı bırak
                obstacle_cells.add((ox-1, oy + buffer_y))
                obstacle_cells.add((ox+1, oy + buffer_y))

        while open_list:
            # En düşük maliyetli (hedefe en yakın) hücreyi seç
            _, current = heapq.heappop(open_list)

            # Hedefe ulaştık mı?
            if current == goal_node:
                return self.reconstruct_path(came_from, current)

            # 4 yöne (Yukarı, Aşağı, Sol, Sağ) komşuları ara
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Ekran sınırları ve engel kontrolü
                if (0 <= neighbor[0] < self.width // self.grid_size and 
                    0 <= neighbor[1] < self.height // self.grid_size):
                    
                    if neighbor not in visited and neighbor not in obstacle_cells:
                        visited.add(neighbor)
                        # Heuristic: Manhattan mesafesi (Arama kriterimiz)
                        h = abs(neighbor[0] - goal_node[0]) + abs(neighbor[1] - goal_node[1])
                        
                        came_from[neighbor] = current
                        heapq.heappush(open_list, (h, neighbor))

        return [] # Yol bulunamadıysa boş dön

    def reconstruct_path(self, came_from, current):
        """Bulunan yolu koordinat listesine çevirir."""
        path = []
        while current in came_from:
            # Grid hücresini tekrar piksel koordinatına çevir
            path.append((current[0] * self.grid_size, current[1] * self.grid_size))
            current = came_from[current]
        return path[::-1] # Yolu baştan sona sırala