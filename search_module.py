import numpy as np
import heapq

class PathFinder:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size

    def a_star(self, start, goal, obstacles):
        if not goal: return []
        
        start_node = (int(start[0] // self.grid_size * self.grid_size), 
                      int(start[1] // self.grid_size * self.grid_size))
        goal_node = (int(goal[0] // self.grid_size * self.grid_size), 
                     int(goal[1] // self.grid_size * self.grid_size))
        
        open_list = []
        heapq.heappush(open_list, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        
        # Muzların bulunduğu grid hücrelerini belirle
        obstacle_nodes = set()
        for obs in obstacles:
            # Muzun etrafındaki 1 gridlik alanı da tehlike bölgesi ilan et (Güvenlik payı)
            ox = int(obs[0] // self.grid_size * self.grid_size)
            oy = int(obs[1] // self.grid_size * self.grid_size)
            obstacle_nodes.add((ox, oy))
            # Komşu hücreleri de ekleyerek muzun içinden geçmeyi engelle
            for dx in [-self.grid_size, 0, self.grid_size]:
                for dy in [-self.grid_size, 0, self.grid_size]:
                    obstacle_nodes.add((ox + dx, oy + dy))

        while open_list:
            current = heapq.heappop(open_list)[1]
            
            if np.linalg.norm(np.array(current) - np.array(goal_node)) < self.grid_size:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            for dx, dy in [(0, self.grid_size), (0, -self.grid_size), 
                           (self.grid_size, 0), (-self.grid_size, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    # EĞER HÜCREDE MUZ VARSA MALİYETİ ÇOK YÜKSEK TUT
                    if neighbor in obstacle_nodes:
                        move_cost = 1000 # Muzun içinden geçmek çok "pahalı"
                    else:
                        move_cost = self.grid_size # Normal yol maliyeti

                    tentative_g_score = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        # Heuristic: Hedefe olan kuş uçuşu mesafe
                        f_score = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal_node))
                        came_from[neighbor] = current
                        heapq.heappush(open_list, (f_score, neighbor))
        return []