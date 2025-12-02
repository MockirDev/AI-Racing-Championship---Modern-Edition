import pygame
import sys
import json
import os
import math
from drawing import draw_track

# Ayarlar
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 880
FPS = 60
GRID_SIZE = 25

# Renkler
C_BG = (50, 50, 50); C_GRID = (60, 60, 60)
C_CP = (0, 255, 0, 100); C_POINT = (255, 0, 0)
C_POINT_SEL = (255, 255, 0); C_LINE_HOVER = (255, 255, 0); C_UI_BG = (40, 40, 40)
C_TEXT = (220, 220, 220); C_TEXT_SEL = (255, 215, 0)

# Veri Fonksiyonları
def get_available_tracks(folder="tracks"):
    if not os.path.exists(folder): os.makedirs(folder)
    return sorted([f for f in os.listdir(folder) if f.endswith('.json')])

def get_track_data(name, folder="tracks"):
    try:
        with open(f"{folder}/{name}", 'r') as f: return json.load(f)
    except Exception: return {"borders": [[], []], "checkpoints": []}

def save_track_data(name, data, folder="tracks"):
    if not name.endswith(".json"): name += ".json"
    try:
        with open(f"{folder}/{name}", 'w') as f: json.dump(data, f, indent=4)
        return True
    except Exception as e: print(f"Hata: {e}"); return False

def get_closest_point_on_segment(p, a, b):
    p_v, a_v, b_v = pygame.math.Vector2(p), pygame.math.Vector2(a), pygame.math.Vector2(b)
    ap, ab = p_v - a_v, b_v - a_v
    ab_len_sq = ab.length_squared()
    if ab_len_sq == 0: return a_v
    t = ap.dot(ab) / ab_len_sq
    return a_v.lerp(b_v, max(0, min(1, t)))

class Editor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Gelişmiş Harita Editörü v2.0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.menu_font = pygame.font.Font(None, 50)
        self.title_font = pygame.font.Font(None, 70)
        self.small_font = pygame.font.Font(None, 20)
        
        # States
        self.STATE_MENU, self.STATE_LOAD, self.STATE_EDITING, self.STATE_SAVE_AS = "MENU", "LOAD", "EDITING", "SAVE_AS"
        self.STATE_TEMPLATES, self.STATE_IMPORT, self.STATE_VALIDATION = "TEMPLATES", "IMPORT", "VALIDATION"
        
        self.camera = pygame.math.Vector2(0, 0)
        self.zoom = 0.5
        self.state = self.STATE_MENU
        self.running = True
        self.track_name, self.track_data = "", {}
        self.selected_point, self.dragging, self.panning = None, False, False
        
        # Advanced Tools System
        self.tools = [
            "select", "add_outer", "add_inner", "add_checkpoint", 
            "brush", "spline", "smooth", "validate"
        ]
        self.current_tool_idx = 0
        
        # Advanced Brush System
        self.brush_size = 75  # Road width in pixels
        self.brush_modes = ["road", "single", "erase", "widen"]
        self.brush_mode_idx = 0  # Default: road mode
        self.brush_path = []
        self.brush_active = False
        self.brush_corner_style = "round"  # round, sharp, beveled
        self.brush_smoothness = 0.7
        self.auto_checkpoint_enabled = True
        self.checkpoint_density = "normal"  # sparse, normal, dense
        
        # Road generation state
        self.road_center_line = []
        self.road_outer_points = []
        self.road_inner_points = []
        self.last_checkpoint_distance = 0
        self.checkpoint_spacing = {"sparse": 200, "normal": 150, "dense": 100}
        
        # Advanced Features
        self.spline_points = []
        self.temp_checkpoint_start, self.hovered_segment = None, None
        self.validation_results = []
        self.auto_smooth_enabled = True
        
        # Menu System
        self.menu_options = [
            "Yeni Harita", "Harita Düzenle", "Template'ler", 
            "Import/Export", "Ana Menüye Dön"
        ]
        self.selected_menu_option, self.available_tracks, self.selected_track_idx = 0, [], 0
        self.save_filename_input, self.save_feedback = "", 0
        
        # Templates
        self.track_templates = self._load_templates()
        self.selected_template_idx = 0
        
        # History for Undo/Redo
        self.history = []
        self.history_index = -1
        self.max_history = 50

    def _load_templates(self):
        """Track template'lerini yükler"""
        templates = {
            "Oval": {
                "borders": [
                    [[400, 200], [800, 200], [900, 300], [900, 500], [800, 600], [400, 600], [300, 500], [300, 300]],
                    [[450, 250], [750, 250], [820, 300], [820, 500], [750, 550], [450, 550], [380, 500], [380, 300]]
                ],
                "checkpoints": [[[600, 150], [600, 250]], [[850, 400], [750, 400]], [[600, 650], [600, 550]], [[350, 400], [450, 400]]]
            },
            "Figure-8": {
                "borders": [
                    [[300, 300], [500, 200], [700, 300], [600, 400], [800, 500], [600, 600], [400, 500], [500, 400]],
                    [[350, 325], [475, 250], [650, 325], [575, 375], [725, 475], [575, 525], [425, 475], [500, 375]]
                ],
                "checkpoints": [[[600, 150], [600, 200]], [[750, 400], [700, 400]], [[600, 650], [600, 600]], [[350, 400], [400, 400]]]
            },
            "S-Curve": {
                "borders": [
                    [[200, 400], [300, 300], [500, 300], [600, 200], [800, 200], [900, 300], [900, 500], [800, 600], [600, 600], [500, 500], [300, 500], [200, 600]],
                    [[250, 425], [325, 350], [475, 350], [575, 275], [750, 275], [825, 350], [825, 450], [750, 525], [575, 525], [475, 450], [325, 450], [250, 525]]
                ],
                "checkpoints": [[[150, 500], [250, 500]], [[600, 150], [600, 275]], [[950, 400], [825, 400]], [[600, 650], [600, 525]]]
            },
            "Montreal": {
                "borders": [
                    [[1725, 700], [1150, 700], [900, 700], [725, 675], [650, 625], [575, 575], [450, 550], [325, 550], [225, 550], [50, 550], [-100, 550], [-225, 550], [-350, 500], [-500, 450], [-575, 450], [-600, 500], [-625, 575], [-700, 600], [-775, 600], [-850, 550], [-875, 500], [-900, 400], [-925, 300], [-900, 200], [-800, 25], [-750, -100], [-700, -225], [-650, -325], [-625, -350], [-575, -350], [-500, -325], [-475, -350], [-450, -425], [-400, -475], [-350, -525], [-275, -550], [-225, -575], [-125, -600], [-50, -600], [50, -600], [125, -600], [175, -625], [200, -675], [175, -725], [175, -775], [225, -800], [300, -800], [400, -800], [500, -750], [600, -725], [700, -700], [775, -650], [850, -625], [925, -550], [1000, -500], [975, -450], [975, -400], [1050, -375], [1100, -350], [1250, -300], [1350, -250], [1450, -200], [1575, -125], [1700, -50], [1800, 25], [1925, 100], [2000, 150], [2100, 200], [2200, 225], [2300, 250], [2375, 325], [2375, 400], [2350, 525], [2300, 625], [2150, 675]],
                    [[350, 325], [50, 350], [-125, 350], [-225, 325], [-350, 275], [-475, 225], [-600, 250], [-650, 350], [-700, 400], [-750, 350], [-750, 275], [-700, 175], [-650, 125], [-600, 25], [-575, -75], [-550, -175], [-475, -175], [-400, -175], [-350, -225], [-325, -300], [-300, -350], [-250, -400], [-150, -425], [-50, -450], [50, -450], [125, -475], [200, -475], [275, -525], [300, -575], [325, -650], [375, -675], [450, -650], [550, -600], [625, -550], [700, -525], [775, -475], [825, -450], [825, -375], [875, -325], [950, -275], [1025, -250], [1125, -200], [1225, -150], [1325, -75], [1425, 0], [1525, 50], [1650, 125], [1750, 175], [1850, 225], [2000, 300], [2075, 325], [2150, 375], [2200, 425], [2175, 475], [2125, 500], [1950, 525], [1725, 525], [1400, 525], [1125, 550], [925, 550], [800, 525], [725, 475], [625, 400], [475, 350]]
                ],
                "checkpoints": [[[700, 450], [600, 600]], [[1475, 525], [1475, 700]], [[2200, 425], [2375, 425]], [[1975, 125], [1900, 250]], [[1325, -275], [1250, -125]], [[975, -425], [825, -425]], [[400, -800], [375, -675]], [[175, -625], [200, -475]], [[-225, -575], [-150, -425]], [[-500, -325], [-350, -225]], [[-700, -225], [-550, -175]], [[-725, 225], [-875, 150]], [[-725, 375], [-850, 550]], [[-575, 450], [-650, 350]], [[-25, 550], [-25, 350]]]
            }
        }
        return templates

    def is_ctrl_pressed(self, event):
        """Multiple methods to detect Ctrl key press - more reliable than single method"""
        
        # Method 1: Event-based modifier detection (most reliable)
        if hasattr(event, 'mod'):
            ctrl_from_event = (event.mod & pygame.KMOD_CTRL) != 0
            if ctrl_from_event:
                return True
        
        # Method 2: Legacy get_mods() method
        ctrl_from_mods = (pygame.key.get_mods() & pygame.KMOD_CTRL) != 0
        if ctrl_from_mods:
            return True
            
        # Method 3: Direct key state checking
        keys_pressed = pygame.key.get_pressed()
        ctrl_from_keys = (keys_pressed[pygame.K_LCTRL] or keys_pressed[pygame.K_RCTRL])
        if ctrl_from_keys:
            return True
        
        # Method 4: Platform-specific detection for Windows
        try:
            import sys
            if sys.platform == "win32":
                # Windows-specific Ctrl detection
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                
                # Check left and right Ctrl keys
                left_ctrl = user32.GetAsyncKeyState(0x11) & 0x8000  # VK_CONTROL
                right_ctrl = user32.GetAsyncKeyState(0xA2) & 0x8000  # VK_RCONTROL
                
                if left_ctrl or right_ctrl:
                    return True
        except:
            pass  # Fallback gracefully if platform detection fails
        
        return False

    def add_to_history(self):
        """Mevcut track state'ini history'ye ekler"""
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(json.loads(json.dumps(self.track_data)))
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1

    def undo(self):
        """Son işlemi geri alır"""
        if self.history_index > 0:
            self.history_index -= 1
            self.track_data = json.loads(json.dumps(self.history[self.history_index]))
            return True
        return False

    def redo(self):
        """Geri alınan işlemi tekrar yapar"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.track_data = json.loads(json.dumps(self.history[self.history_index]))
            return True
        return False

    def validate_track(self):
        """Track'i validate eder ve sorunları tespit eder"""
        self.validation_results = []
        
        # Outer border kontrolü
        if len(self.track_data["borders"][0]) < 3:
            self.validation_results.append("❌ Outer border en az 3 nokta içermeli")
        
        # Inner border kontrolü
        if len(self.track_data["borders"][1]) < 3:
            self.validation_results.append("⚠️ Inner border en az 3 nokta içermeli")
        
        # Checkpoint kontrolü
        if len(self.track_data["checkpoints"]) < 1:
            self.validation_results.append("❌ En az 1 checkpoint gerekli")
        
        # Self-intersection kontrolü
        outer_border = self.track_data["borders"][0]
        if len(outer_border) > 3:
            for i in range(len(outer_border)):
                p1, p2 = outer_border[i], outer_border[(i + 1) % len(outer_border)]
                for j in range(i + 2, len(outer_border)):
                    if j == len(outer_border) - 1 and i == 0:
                        continue  # Skip adjacent segments
                    p3, p4 = outer_border[j], outer_border[(j + 1) % len(outer_border)]
                    if self._lines_intersect(p1, p2, p3, p4):
                        self.validation_results.append(f"❌ Outer border self-intersection at segment {i}-{j}")
                        break
        
        # Track genişlik kontrolü
        if len(self.track_data["borders"][0]) > 0 and len(self.track_data["borders"][1]) > 0:
            min_width = float('inf')
            for outer_point in self.track_data["borders"][0]:
                for inner_point in self.track_data["borders"][1]:
                    width = math.dist(outer_point, inner_point)
                    min_width = min(min_width, width)
            
            if min_width < 50:
                self.validation_results.append("⚠️ Track çok dar (min 50px önerilir)")
            elif min_width > 200:
                self.validation_results.append("⚠️ Track çok geniş (max 200px önerilir)")
        
        if not self.validation_results:
            self.validation_results.append("✅ Track geçerli!")
        
        return len([r for r in self.validation_results if r.startswith("❌")]) == 0

    def _lines_intersect(self, p1, p2, p3, p4):
        """İki çizginin kesişip kesişmediğini kontrol eder"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def smooth_path(self, points, strength=0.5):
        """Noktaları smooth eder"""
        if len(points) < 3:
            return points
        
        smoothed = []
        for i in range(len(points)):
            prev_point = points[(i - 1) % len(points)]
            curr_point = points[i]
            next_point = points[(i + 1) % len(points)]
            
            # Average neighboring points
            smooth_x = curr_point[0] * (1 - strength) + (prev_point[0] + next_point[0]) * 0.5 * strength
            smooth_y = curr_point[1] * (1 - strength) + (prev_point[1] + next_point[1]) * 0.5 * strength
            
            smoothed.append([smooth_x, smooth_y])
        
        return smoothed

    def apply_template(self, template_name):
        """Template'i mevcut track'e uygular"""
        if template_name in self.track_templates:
            self.add_to_history()
            self.track_data = json.loads(json.dumps(self.track_templates[template_name]))

    def auto_generate_checkpoints(self):
        """Otomatik checkpoint oluşturur"""
        if len(self.track_data["borders"][0]) < 4:
            return
        
        self.add_to_history()
        outer_border = self.track_data["borders"][0]
        inner_border = self.track_data["borders"][1] if len(self.track_data["borders"][1]) > 0 else []
        
        # Track merkez hattını hesapla
        center_points = []
        for i in range(len(outer_border)):
            outer_point = outer_border[i]
            if inner_border:
                # En yakın inner point bul
                closest_inner = min(inner_border, key=lambda p: math.dist(outer_point, p))
                # Merkez nokta
                center_x = (outer_point[0] + closest_inner[0]) / 2
                center_y = (outer_point[1] + closest_inner[1]) / 2
                center_points.append([center_x, center_y])
            else:
                center_points.append(outer_point)
        
        # Her 4 noktada bir checkpoint oluştur
        self.track_data["checkpoints"] = []
        step = max(1, len(center_points) // 6)  # 6 checkpoint hedefi
        
        for i in range(0, len(center_points), step):
            center_point = center_points[i]
            next_point = center_points[(i + 1) % len(center_points)]
            
            # Perpendicular line oluştur
            dx = next_point[0] - center_point[0]
            dy = next_point[1] - center_point[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize and perpendicular
                perp_x = -dy / length * 30  # 30 pixel checkpoint genişliği
                perp_y = dx / length * 30
                
                checkpoint_start = [center_point[0] + perp_x, center_point[1] + perp_y]
                checkpoint_end = [center_point[0] - perp_x, center_point[1] - perp_y]
                
                self.track_data["checkpoints"].append([checkpoint_start, checkpoint_end])

    def generate_spline_curve(self, control_points, num_points=10):
        """Kontrol noktalarından spline eğrisi oluşturur"""
        if len(control_points) < 3:
            return control_points
        
        curve_points = []
        
        # Simple Catmull-Rom spline implementation
        for i in range(len(control_points) - 1):
            p0 = control_points[max(0, i-1)]
            p1 = control_points[i]
            p2 = control_points[i+1]
            p3 = control_points[min(len(control_points)-1, i+2)]
            
            for t in range(num_points):
                t_norm = t / num_points
                
                # Catmull-Rom formula
                q = (
                    (-t_norm**3 + 2*t_norm**2 - t_norm) * 0.5,
                    (3*t_norm**3 - 5*t_norm**2 + 2) * 0.5,
                    (-3*t_norm**3 + 4*t_norm**2 + t_norm) * 0.5,
                    (t_norm**3 - t_norm**2) * 0.5
                )
                
                x = q[0]*p0[0] + q[1]*p1[0] + q[2]*p2[0] + q[3]*p3[0]
                y = q[0]*p0[1] + q[1]*p1[1] + q[2]*p2[1] + q[3]*p3[1]
                
                # Grid snap
                snapped_point = [round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE]
                if not curve_points or math.dist(curve_points[-1], snapped_point) > GRID_SIZE/2:
                    curve_points.append(snapped_point)
        
        return curve_points

    def find_nearest_border(self, world_pos):
        """En yakın border'ı bulur"""
        min_distance = float('inf')
        nearest_border_idx = None
        
        for border_idx, border in enumerate(self.track_data["borders"]):
            if len(border) > 0:
                for point in border:
                    distance = math.dist(world_pos, point)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_border_idx = border_idx
        
        # 50 pixel mesafe içindeyse kabul et
        if min_distance < 50:
            return nearest_border_idx
        return None

    def calculate_perpendicular_vector(self, p1, p2):
        """İki nokta arasındaki dik vektörü hesaplar"""
        if p1 == p2:
            return [0, 1]  # Default upward
        
        # Direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return [0, 1]
        
        # Normalize and rotate 90 degrees
        norm_dx = dx / length
        norm_dy = dy / length
        
        # Perpendicular vector (rotated 90 degrees counter-clockwise)
        perp_x = -norm_dy
        perp_y = norm_dx
        
        return [perp_x, perp_y]

    def detect_curvature(self, points, index):
        """Belirli bir noktadaki eğrilik miktarını tespit eder"""
        if len(points) < 3 or index < 1 or index >= len(points) - 1:
            return 0
        
        p1 = points[index - 1]
        p2 = points[index]
        p3 = points[index + 1]
        
        # Calculate vectors
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        # Calculate angle between vectors
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp
        angle = math.acos(cos_angle)
        
        # Convert to curvature (0 = straight, 1 = sharp turn)
        curvature = angle / math.pi
        return curvature

    def generate_road_from_path(self, path):
        """Path'den otomatik road (outer + inner borders) oluşturur"""
        if len(path) < 2:
            return [], []
        
        current_mode = self.brush_modes[self.brush_mode_idx]
        half_width = self.brush_size / 2
        
        outer_points = []
        inner_points = []
        
        # Smooth the path first
        smoothed_path = self.smooth_path(path, strength=self.brush_smoothness)
        
        for i, point in enumerate(smoothed_path):
            if i == 0:
                # First point - use direction to next point
                perp_vec = self.calculate_perpendicular_vector(point, smoothed_path[i + 1])
            elif i == len(smoothed_path) - 1:
                # Last point - use direction from previous point
                perp_vec = self.calculate_perpendicular_vector(smoothed_path[i - 1], point)
            else:
                # Middle points - average of two perpendicular vectors
                perp1 = self.calculate_perpendicular_vector(smoothed_path[i - 1], point)
                perp2 = self.calculate_perpendicular_vector(point, smoothed_path[i + 1])
                perp_vec = [(perp1[0] + perp2[0]) / 2, (perp1[1] + perp2[1]) / 2]
                
                # Normalize
                length = math.sqrt(perp_vec[0]**2 + perp_vec[1]**2)
                if length > 0:
                    perp_vec = [perp_vec[0] / length, perp_vec[1] / length]
            
            # Adjust width based on curvature
            curvature = self.detect_curvature(smoothed_path, i)
            dynamic_width = half_width * (1 + curvature * 0.3)  # Widen on curves
            
            # Generate outer and inner points
            outer_x = point[0] + perp_vec[0] * dynamic_width
            outer_y = point[1] + perp_vec[1] * dynamic_width
            inner_x = point[0] - perp_vec[0] * dynamic_width  
            inner_y = point[1] - perp_vec[1] * dynamic_width
            
            # Grid snap
            outer_points.append([round(outer_x / GRID_SIZE) * GRID_SIZE, 
                               round(outer_y / GRID_SIZE) * GRID_SIZE])
            inner_points.append([round(inner_x / GRID_SIZE) * GRID_SIZE, 
                               round(inner_y / GRID_SIZE) * GRID_SIZE])
        
        return outer_points, inner_points

    def place_smart_checkpoints(self, center_path):
        """Akıllı checkpoint yerleştirme sistemi"""
        if not self.auto_checkpoint_enabled or len(center_path) < 2:
            return
        
        spacing = self.checkpoint_spacing[self.checkpoint_density]
        total_distance = 0
        checkpoint_distance = 0
        
        # Calculate total path length and place checkpoints
        for i in range(1, len(center_path)):
            segment_dist = math.dist(center_path[i-1], center_path[i])
            total_distance += segment_dist
            checkpoint_distance += segment_dist
            
            # Check if we should place a checkpoint
            should_place = False
            checkpoint_type = "normal"
            
            # Distance-based placement
            if checkpoint_distance >= spacing:
                should_place = True
                checkpoint_distance = 0
            
            # Curvature-based placement (for sharp turns)
            curvature = self.detect_curvature(center_path, i)
            if curvature > 0.5 and checkpoint_distance > spacing * 0.5:
                should_place = True
                checkpoint_type = "curve"
                checkpoint_distance = 0
            
            # Long straight section detection
            if i > 5:
                straight_distance = 0
                for j in range(max(0, i-5), i):
                    if self.detect_curvature(center_path, j) < 0.1:
                        straight_distance += math.dist(center_path[j], center_path[j+1]) if j+1 < len(center_path) else 0
                
                if straight_distance > spacing * 1.5:
                    checkpoint_type = "straight"
            
            if should_place:
                # Calculate perpendicular line for checkpoint
                perp_vec = self.calculate_perpendicular_vector(
                    center_path[max(0, i-1)], 
                    center_path[min(len(center_path)-1, i+1)]
                )
                
                checkpoint_width = self.brush_size * 0.8  # Slightly narrower than track
                p1 = [
                    center_path[i][0] + perp_vec[0] * checkpoint_width / 2,
                    center_path[i][1] + perp_vec[1] * checkpoint_width / 2
                ]
                p2 = [
                    center_path[i][0] - perp_vec[0] * checkpoint_width / 2,
                    center_path[i][1] - perp_vec[1] * checkpoint_width / 2
                ]
                
                # Add metadata for checkpoint type (could be used for coloring)
                checkpoint_data = [p1, p2]
                if checkpoint_type != "normal":
                    # Store type info if needed for future features
                    pass
                
                self.track_data["checkpoints"].append(checkpoint_data)
                print(f"Smart checkpoint placed: {checkpoint_type} at distance {total_distance:.1f}")

    def complete_road_generation(self):
        """Brush tool bitiminde road generation'ı tamamlar"""
        if len(self.road_center_line) < 2:
            print("Road generation: Not enough points")
            return
        
        current_mode = self.brush_modes[self.brush_mode_idx]
        
        print(f"Completing road generation: {len(self.road_center_line)} points, mode: {current_mode}")
        
        if current_mode == "road":
            # Generate both outer and inner borders
            outer_points, inner_points = self.generate_road_from_path(self.road_center_line)
            
            if outer_points and inner_points:
                # Clear existing borders and add new ones
                self.track_data["borders"][0] = outer_points
                self.track_data["borders"][1] = inner_points
                
                print(f"Road generated: {len(outer_points)} outer, {len(inner_points)} inner points")
                
                # Generate smart checkpoints
                if self.auto_checkpoint_enabled:
                    self.place_smart_checkpoints(self.road_center_line)
                    
        elif current_mode == "single":
            # Generate only outer border
            outer_points, _ = self.generate_road_from_path(self.road_center_line)
            if outer_points:
                self.track_data["borders"][0].extend(outer_points)
                print(f"Single border added: {len(outer_points)} points")
                
        elif current_mode == "erase":
            # Erase mode - remove nearby points (simplified implementation)
            erase_radius = self.brush_size
            points_removed = 0
            
            for center_point in self.road_center_line:
                # Remove outer border points
                self.track_data["borders"][0] = [
                    p for p in self.track_data["borders"][0] 
                    if math.dist(p, center_point) > erase_radius
                ]
                # Remove inner border points
                self.track_data["borders"][1] = [
                    p for p in self.track_data["borders"][1] 
                    if math.dist(p, center_point) > erase_radius
                ]
                points_removed += 1
            
            print(f"Erase mode: Cleared {points_removed} areas with radius {erase_radius}px")
            
        elif current_mode == "widen":
            # Widen existing track (simplified implementation)
            widen_amount = self.brush_size / 4  # Quarter of brush size for widening
            
            # Find nearest borders and widen them
            for center_point in self.road_center_line:
                # Find nearby border points and push them outward
                for border_idx, border in enumerate(self.track_data["borders"]):
                    for i, point in enumerate(border):
                        if math.dist(point, center_point) < self.brush_size:
                            # Calculate direction from center to point
                            dx = point[0] - center_point[0]
                            dy = point[1] - center_point[1]
                            length = math.sqrt(dx*dx + dy*dy)
                            
                            if length > 0:
                                # Push point outward
                                norm_dx = dx / length
                                norm_dy = dy / length
                                
                                new_x = point[0] + norm_dx * widen_amount
                                new_y = point[1] + norm_dy * widen_amount
                                
                                self.track_data["borders"][border_idx][i] = [
                                    round(new_x / GRID_SIZE) * GRID_SIZE,
                                    round(new_y / GRID_SIZE) * GRID_SIZE
                                ]
            
            print(f"Widen mode: Expanded track by {widen_amount}px")
        
        # Clear road generation state
        self.road_center_line = []
        self.road_outer_points = []
        self.road_inner_points = []
        
        # Set feedback
        self.save_feedback = FPS * 2
        
        print("Road generation completed successfully!")

    def world_to_screen(self, pos):
        return (pygame.math.Vector2(pos) - self.camera) * self.zoom + pygame.math.Vector2(self.screen.get_size()) / 2

    def screen_to_world(self, pos):
        return (pygame.math.Vector2(pos) - pygame.math.Vector2(self.screen.get_size()) / 2) / self.zoom + self.camera

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        print("Harita editöründen çıkılıyor...")

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        world_pos = self.screen_to_world(mouse_pos)
        snapped_pos = [round(c / GRID_SIZE) * GRID_SIZE for c in world_pos]

        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            if self.state == self.STATE_EDITING:
                if event.type == pygame.MOUSEWHEEL:
                    zoom_change = 1.1 if event.y > 0 else 1/1.1
                    old_world_pos = self.screen_to_world(mouse_pos)
                    self.zoom = max(0.1, min(self.zoom * zoom_change, 5.0))
                    self.camera += old_world_pos - self.screen_to_world(mouse_pos)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2: self.panning = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 2: self.panning = False
            if event.type == pygame.MOUSEMOTION and self.panning:
                self.camera -= pygame.math.Vector2(event.rel) / self.zoom

            if self.state == self.STATE_EDITING: self.handle_editing_events(event, world_pos, snapped_pos)
            elif self.state == self.STATE_MENU: self.handle_menu_events(event)
            elif self.state == self.STATE_LOAD: self.handle_load_events(event)
            elif self.state == self.STATE_SAVE_AS: self.handle_save_as_events(event)
            elif self.state == self.STATE_TEMPLATES: self.handle_templates_events(event)

    def handle_editing_events(self, event, world_pos, snapped_pos):
        current_tool = self.tools[self.current_tool_idx]
        if event.type == pygame.KEYDOWN:
            # Debug modifier detection
            ctrl_pressed = self.is_ctrl_pressed(event)
            if ctrl_pressed:
                print(f"Ctrl detected with key: {pygame.key.name(event.key)}")
            
            if event.key == pygame.K_ESCAPE: self.state = self.STATE_MENU
            # Support keys 1-8 for tools
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]: 
                new_idx = event.key - pygame.K_1
                if new_idx < len(self.tools):
                    self.current_tool_idx = new_idx
                    print(f"Tool değiştirildi: {self.tools[self.current_tool_idx]}")
            # Additional hotkeys
            elif event.key == pygame.K_9: self.state = self.STATE_SAVE_AS  # Save As hotkey
            elif event.key == pygame.K_v: self.validate_track(); print("Track validation:", self.validation_results)
            elif event.key == pygame.K_g: self.auto_generate_checkpoints(); print("Checkpoints otomatik oluşturuldu!")
            
            # Enhanced Ctrl shortcuts with multiple detection methods
            elif event.key == pygame.K_z and ctrl_pressed: 
                if self.undo(): 
                    print("✅ Undo yapıldı!")
                    self.save_feedback = FPS * 1  # Visual feedback
                else:
                    print("❌ Undo: History boş!")
                    
            elif event.key == pygame.K_y and ctrl_pressed: 
                if self.redo(): 
                    print("✅ Redo yapıldı!")
                    self.save_feedback = FPS * 1  # Visual feedback
                else:
                    print("❌ Redo: History sonunda!")
                    
            elif event.key == pygame.K_s and ctrl_pressed:
                if self.track_name and save_track_data(self.track_name, self.track_data): 
                    self.save_feedback = FPS * 2
                    print("✅ Harita kaydedildi!")
                else:
                    print("❌ Kayıt başarısız - harita adı eksik!")
                    self.state = self.STATE_SAVE_AS
                    
            # New Ctrl shortcuts
            elif event.key == pygame.K_n and ctrl_pressed:
                print("✅ Yeni harita oluşturuluyor...")
                self.track_data = {"borders": [[], []], "checkpoints": []}
                self.add_to_history()  # Add initial state to history
                self.track_name = ""
                self.save_filename_input = "yeni_harita"
                self.state = self.STATE_SAVE_AS
                
            elif event.key == pygame.K_o and ctrl_pressed:
                print("✅ Harita yükleme menüsü açılıyor...")
                self.available_tracks = get_available_tracks()
                if self.available_tracks:
                    self.state = self.STATE_LOAD
                    self.selected_track_idx = 0
                else:
                    print("❌ Yüklenecek harita bulunamadı!")

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
            self.dragging = False
            # Complete road generation when brush drawing finishes
            if self.brush_active and current_tool == "brush":
                self.complete_road_generation()
            self.brush_active = False  # Stop brush drawing
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging and self.selected_point:
                pt_type, l_idx, p_idx = self.selected_point["type"], self.selected_point["list_idx"], self.selected_point["point_idx"]
                if pt_type == "border": self.track_data["borders"][l_idx][p_idx] = snapped_pos
                elif pt_type == "checkpoint": self.track_data["checkpoints"][l_idx][p_idx] = snapped_pos
            elif self.brush_active and current_tool == "brush":
                # Advanced brush drawing - Continue road center line
                if len(self.road_center_line) == 0 or math.dist(self.road_center_line[-1], snapped_pos) > GRID_SIZE:
                    self.road_center_line.append(snapped_pos)
                    print(f"Road center line: {len(self.road_center_line)} points")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.hovered_segment and current_tool in ["add_outer", "add_inner"]:
                    b_idx, p2_idx = self.hovered_segment['b_idx'], self.hovered_segment['p2_idx']
                    self.track_data["borders"][b_idx].insert(p2_idx, snapped_pos)
                elif current_tool == "select":
                    found = False
                    points = [("border",i,j,p) for i,b in enumerate(self.track_data["borders"]) for j,p in enumerate(b)] + [("checkpoint",i,j,p) for i,c in enumerate(self.track_data["checkpoints"]) for j,p in enumerate(c)]
                    for p_type, l_idx, p_idx, point in sorted(points, key=lambda x: math.dist(world_pos, x[3])):
                        if math.dist(world_pos, point) * self.zoom < 10:
                            self.selected_point = {"type": p_type, "list_idx": l_idx, "point_idx": p_idx}; self.dragging = True; found = True; break
                    if not found: self.selected_point = None
                elif current_tool == "add_outer": self.track_data["borders"][0].append(snapped_pos)
                elif current_tool == "add_inner": self.track_data["borders"][1].append(snapped_pos)
                elif current_tool == "add_checkpoint":
                    if not self.temp_checkpoint_start: self.temp_checkpoint_start = snapped_pos
                    else: self.track_data["checkpoints"].append([self.temp_checkpoint_start, snapped_pos]); self.temp_checkpoint_start = None
                elif current_tool == "brush":
                    # Advanced Brush tool - Start road generation
                    self.add_to_history()
                    self.brush_active = True
                    self.brush_path = [snapped_pos]
                    self.road_center_line = [snapped_pos]
                    current_mode = self.brush_modes[self.brush_mode_idx]
                    print(f"Advanced Brush: {current_mode} mode - Width: {self.brush_size}px")
                elif current_tool == "spline":
                    # Spline tool - Add control point
                    self.spline_points.append(snapped_pos)
                    print(f"Spline nokta eklendi: {len(self.spline_points)} nokta")
                    if len(self.spline_points) >= 3:
                        # Generate spline curve
                        self.add_to_history()
                        spline_curve = self.generate_spline_curve(self.spline_points)
                        self.track_data["borders"][0].extend(spline_curve)
                        self.spline_points = []  # Reset
                        print("Spline eğrisi oluşturuldu!")
                elif current_tool == "smooth":
                    # Smooth tool - Apply smoothing to nearest border
                    border_to_smooth = self.find_nearest_border(world_pos)
                    if border_to_smooth is not None:
                        self.add_to_history()
                        border_idx = border_to_smooth
                        if len(self.track_data["borders"][border_idx]) > 3:
                            self.track_data["borders"][border_idx] = self.smooth_path(
                                self.track_data["borders"][border_idx], strength=0.3
                            )
                            print(f"Border {border_idx} yumuşatıldı!")
                elif current_tool == "validate":
                    # Validate tool - Run validation and show results
                    self.validate_track()
                    print("Track validation sonuçları:")
                    for result in self.validation_results:
                        print(f"  {result}")
                    self.save_feedback = FPS * 3  # Show feedback longer
            elif event.button == 3:
                to_delete = None
                points_to_check = [("border",i,j,p) for i,b in enumerate(self.track_data["borders"]) for j,p in enumerate(b)] + [("checkpoint_line",i,c) for i,c in enumerate(self.track_data["checkpoints"])]
                for item in points_to_check:
                    p_type, l_idx = item[0], item[1]
                    if p_type == "border" and math.dist(world_pos, item[3]) * self.zoom < 10: to_delete = item; break
                    elif p_type == "checkpoint_line":
                        p1, p2 = item[2]; closest_pt = get_closest_point_on_segment(world_pos, p1, p2)
                        if pygame.math.Vector2(world_pos).distance_to(closest_pt) * self.zoom < 10: to_delete = item; break
                if to_delete:
                    if to_delete[0] == "border": del self.track_data["borders"][to_delete[1]][to_delete[2]]
                    elif to_delete[0] == "checkpoint_line": del self.track_data["checkpoints"][to_delete[1]]

    def handle_menu_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: self.selected_menu_option = (self.selected_menu_option - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN: self.selected_menu_option = (self.selected_menu_option + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN:
                if self.selected_menu_option == 0: 
                    self.track_data = {"borders": [[], []], "checkpoints": []}
                    self.save_filename_input = "yeni_harita"
                    self.state = self.STATE_SAVE_AS
                elif self.selected_menu_option == 1: 
                    self.available_tracks = get_available_tracks()
                    self.state = self.STATE_LOAD
                elif self.selected_menu_option == 2: 
                    self.state = self.STATE_TEMPLATES
                elif self.selected_menu_option == 3: 
                    print("Import/Export özelliği yakında eklenecek!")
                    # self.state = self.STATE_IMPORT  # Şimdilik disabled
                elif self.selected_menu_option == 4: 
                    self.running = False

    def handle_load_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: self.state = self.STATE_MENU
            if self.available_tracks:
                if event.key == pygame.K_UP: self.selected_track_idx = (self.selected_track_idx - 1) % len(self.available_tracks)
                elif event.key == pygame.K_DOWN: self.selected_track_idx = (self.selected_track_idx + 1) % len(self.available_tracks)
                elif event.key == pygame.K_RETURN:
                    self.track_name = self.available_tracks[self.selected_track_idx]; self.track_data = get_track_data(self.track_name); self.state = self.STATE_EDITING
    
    def handle_save_as_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.save_filename_input:
                    self.track_name = self.save_filename_input
                    if save_track_data(self.track_name, self.track_data): self.save_feedback = FPS * 2
                    self.state = self.STATE_EDITING
            elif event.key == pygame.K_BACKSPACE: self.save_filename_input = self.save_filename_input[:-1]
            elif event.key == pygame.K_ESCAPE: self.state = self.STATE_MENU
            else: self.save_filename_input += event.unicode

    def handle_templates_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: self.state = self.STATE_MENU
            elif event.key == pygame.K_UP: 
                self.selected_template_idx = (self.selected_template_idx - 1) % len(self.track_templates)
            elif event.key == pygame.K_DOWN: 
                self.selected_template_idx = (self.selected_template_idx + 1) % len(self.track_templates)
            elif event.key == pygame.K_RETURN:
                template_names = list(self.track_templates.keys())
                selected_template = template_names[self.selected_template_idx]
                self.apply_template(selected_template)
                self.track_name = f"template_{selected_template.lower()}"
                self.save_feedback = FPS * 2  # Show feedback
                self.state = self.STATE_EDITING
                print(f"Template '{selected_template}' uygulandı!")

    def update(self):
        keys = pygame.key.get_pressed()
        pan_speed = 20 / self.zoom
        if self.state == self.STATE_EDITING:
            if keys[pygame.K_UP] or keys[pygame.K_w]: self.camera.y -= pan_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: self.camera.y += pan_speed
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: self.camera.x -= pan_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.camera.x += pan_speed
        if self.save_feedback > 0: self.save_feedback -= 1

    def draw(self):
        self.screen.fill(C_BG)
        if self.state == self.STATE_EDITING: self.draw_editing_screen()
        elif self.state == self.STATE_MENU: self.draw_menu_screen()
        elif self.state == self.STATE_LOAD: self.draw_load_screen()
        elif self.state == self.STATE_SAVE_AS: self.draw_save_as_screen()
        elif self.state == self.STATE_TEMPLATES: self.draw_templates_screen()
        pygame.display.flip()

    def draw_editing_screen(self):
        self.draw_grid()
        
        world_borders = [[self.world_to_screen(p) for p in path] for path in self.track_data["borders"]]
        world_cps = [[self.world_to_screen(p) for p in pair] for pair in self.track_data["checkpoints"]]
        draw_track(self.screen, world_borders, world_cps)

        for cp in world_cps:
            pygame.draw.line(self.screen, C_CP, cp[0], cp[1], 3)
        
        mouse_pos = pygame.mouse.get_pos()
        world_pos = self.screen_to_world(mouse_pos)
        snapped_pos = [round(c / GRID_SIZE) * GRID_SIZE for c in world_pos]
        
        self.hovered_segment = None
        current_tool = self.tools[self.current_tool_idx]
        if current_tool in ["add_outer", "add_inner"]:
            world_pos = self.screen_to_world(pygame.mouse.get_pos())
            border_idx = 0 if current_tool == "add_outer" else 1
            if border_idx < len(self.track_data["borders"]):
                border = self.track_data["borders"][border_idx]
                for i in range(len(border)):
                    p1, p2 = border[i], border[(i + 1) % len(border)]
                    if not p1 or not p2: continue
                    closest_pt = get_closest_point_on_segment(world_pos, p1, p2)
                    if pygame.math.Vector2(world_pos).distance_to(closest_pt) * self.zoom < 10:
                        self.hovered_segment = {"b_idx": border_idx, "p2_idx": (i + 1) % len(border)}
                        snapped_pos = [round(c / GRID_SIZE) * GRID_SIZE for c in world_pos]
                        pygame.draw.line(self.screen, C_LINE_HOVER, self.world_to_screen(p1), self.world_to_screen(p2), 4)
                        pygame.draw.circle(self.screen, C_LINE_HOVER, self.world_to_screen(snapped_pos), 8)
                        break

        for i, border in enumerate(self.track_data["borders"]):
            for j, point in enumerate(border):
                is_selected = self.selected_point and self.selected_point["type"] == "border" and self.selected_point["list_idx"] == i and self.selected_point["point_idx"] == j
                pygame.draw.circle(self.screen, C_POINT_SEL if is_selected else C_POINT, self.world_to_screen(point), 6)
        
        for i, cp in enumerate(self.track_data["checkpoints"]):
            for j, point in enumerate(cp):
                is_selected = self.selected_point and self.selected_point["type"] == "checkpoint" and self.selected_point["list_idx"] == i and self.selected_point["point_idx"] == j
                pygame.draw.circle(self.screen, C_POINT_SEL if is_selected else C_POINT, self.world_to_screen(point), 6)
        
        if self.temp_checkpoint_start: pygame.draw.line(self.screen, C_CP, self.world_to_screen(self.temp_checkpoint_start), mouse_pos, 2)
        
        # Bigger UI panel for all tools
        ui_surf = pygame.Surface((320, 400), pygame.SRCALPHA); ui_surf.fill(C_UI_BG)
        
        # Tool list
        for i, tool_name in enumerate(self.tools):
            text = self.font.render(f"[{i+1}] {tool_name.replace('_',' ').title()}", True, C_TEXT_SEL if i == self.current_tool_idx else C_TEXT)
            ui_surf.blit(text, (10, 10 + i * 30))
        
        # Additional shortcuts
        shortcuts = [
            "[9] Save As",
            "[V] Validate Track", 
            "[G] Auto Checkpoints",
            "[Ctrl+N] New Track",
            "[Ctrl+O] Open Track",
            "[Ctrl+S] Save",
            "[Ctrl+Z] Undo",
            "[Ctrl+Y] Redo"
        ]
        
        y_offset = 10 + len(self.tools) * 30 + 20  # After tools + gap
        shortcut_title = self.small_font.render("--- Shortcuts ---", True, C_TEXT_SEL)
        ui_surf.blit(shortcut_title, (10, y_offset))
        y_offset += 25
        
        for shortcut in shortcuts:
            text = self.small_font.render(shortcut, True, C_TEXT)
            ui_surf.blit(text, (15, y_offset))
            y_offset += 20
        
        self.screen.blit(ui_surf, (10, 10))

        if self.save_feedback > 0:
            save_text = self.font.render("Kaydedildi!", True, (0,200,0)); self.screen.blit(save_text, (240, 10))

    def draw_grid(self):
        start_world = self.screen_to_world((0, 0)); end_world = self.screen_to_world((SCREEN_WIDTH, SCREEN_HEIGHT))
        start_x = math.floor(start_world.x / GRID_SIZE) * GRID_SIZE; end_x = math.ceil(end_world.x / GRID_SIZE) * GRID_SIZE
        start_y = math.floor(start_world.y / GRID_SIZE) * GRID_SIZE; end_y = math.ceil(end_world.y / GRID_SIZE) * GRID_SIZE
        for x in range(start_x, end_x, GRID_SIZE):
            pygame.draw.line(self.screen, C_GRID, self.world_to_screen((x, start_y)), self.world_to_screen((x, end_y)))
        for y in range(start_y, end_y, GRID_SIZE):
            pygame.draw.line(self.screen, C_GRID, self.world_to_screen((start_x, y)), self.world_to_screen((end_x, y)))

    def draw_menu_screen(self):
        title = self.title_font.render("Harita Editörü", True, C_TEXT_SEL)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH/2, 150)))
        for i, option in enumerate(self.menu_options):
            color = C_TEXT_SEL if i == self.selected_menu_option else C_TEXT
            text = self.menu_font.render(option, True, color); rect = text.get_rect(center=(SCREEN_WIDTH/2, 350 + i * 100)); self.screen.blit(text, rect)

    def draw_load_screen(self):
        title = self.title_font.render("Harita Yükle", True, C_TEXT_SEL)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH/2, 100)))
        if not self.available_tracks:
            err = self.menu_font.render("'tracks' klasörü boş.", True, C_TEXT); self.screen.blit(err, err.get_rect(center=(SCREEN_WIDTH/2, 300)))
        else:
            for i, name in enumerate(self.available_tracks):
                color = C_TEXT_SEL if i == self.selected_track_idx else C_TEXT
                text = self.font.render(name, True, color); self.screen.blit(text, text.get_rect(center=(SCREEN_WIDTH/2, 200 + i * 40)))
        help_text = self.font.render("[ESC] ile Ana Menüye Dön", True, C_TEXT); self.screen.blit(help_text, (10, SCREEN_HEIGHT - 40))

    def draw_save_as_screen(self):
        self.draw_editing_screen()
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0,0))
        box_rect = pygame.Rect(0, 0, 600, 200); box_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2); pygame.draw.rect(self.screen, C_UI_BG, box_rect, border_radius=10)
        prompt = self.font.render("Yeni Harita Adı:", True, C_TEXT); self.screen.blit(prompt, (box_rect.x + 20, box_rect.y + 30))
        input_box = pygame.Rect(box_rect.x + 20, box_rect.y + 70, 560, 50); pygame.draw.rect(self.screen, C_BG, input_box)
        input_text = self.menu_font.render(self.save_filename_input, True, C_TEXT); self.screen.blit(input_text, (input_box.x + 10, input_box.y + 5))

    def draw_templates_screen(self):
        title = self.title_font.render("Template Seç", True, C_TEXT_SEL)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH/2, 100)))
        
        template_names = list(self.track_templates.keys())
        
        # Template listesi
        for i, template_name in enumerate(template_names):
            color = C_TEXT_SEL if i == self.selected_template_idx else C_TEXT
            text = self.menu_font.render(template_name, True, color)
            rect = text.get_rect(center=(SCREEN_WIDTH/2, 250 + i * 80))
            self.screen.blit(text, rect)
            
            # Template önizlemesi (küçük)
            if i == self.selected_template_idx:
                preview_size = 200
                preview_surf = pygame.Surface((preview_size, preview_size), pygame.SRCALPHA)
                preview_surf.fill((20, 20, 20, 200))
                
                template_data = self.track_templates[template_name]
                if template_data["borders"] and template_data["borders"][0]:
                    # Bounding box hesapla
                    all_points = []
                    for border in template_data["borders"]:
                        all_points.extend(border)
                    
                    if all_points:
                        min_x = min(p[0] for p in all_points)
                        max_x = max(p[0] for p in all_points)
                        min_y = min(p[1] for p in all_points)
                        max_y = max(p[1] for p in all_points)
                        
                        # Scale to fit preview
                        scale_x = preview_size * 0.8 / max(1, max_x - min_x)
                        scale_y = preview_size * 0.8 / max(1, max_y - min_y)
                        scale = min(scale_x, scale_y)
                        
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        
                        # Draw borders
                        for j, border in enumerate(template_data["borders"]):
                            if len(border) > 2:
                                scaled_points = []
                                for point in border:
                                    px = (point[0] - center_x) * scale + preview_size/2
                                    py = (point[1] - center_y) * scale + preview_size/2
                                    scaled_points.append([px, py])
                                
                                color = (100, 100, 100) if j == 0 else (80, 80, 80)
                                if len(scaled_points) > 2:
                                    pygame.draw.polygon(preview_surf, color, scaled_points, 2)
                        
                        # Draw checkpoints
                        for checkpoint in template_data["checkpoints"]:
                            if len(checkpoint) == 2:
                                p1x = (checkpoint[0][0] - center_x) * scale + preview_size/2
                                p1y = (checkpoint[0][1] - center_y) * scale + preview_size/2
                                p2x = (checkpoint[1][0] - center_x) * scale + preview_size/2
                                p2y = (checkpoint[1][1] - center_y) * scale + preview_size/2
                                pygame.draw.line(preview_surf, (0, 200, 0), (p1x, p1y), (p2x, p2y), 2)
                
                # Önizleme konumu
                preview_rect = preview_surf.get_rect(center=(SCREEN_WIDTH - 150, SCREEN_HEIGHT/2))
                self.screen.blit(preview_surf, preview_rect)
        
        # Help text
        help_text = self.font.render("[ENTER] Seç  [ESC] Ana Menü", True, C_TEXT)
        help_rect = help_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT - 50))
        self.screen.blit(help_text, help_rect)

def editor_main():
    Editor().run()

if __name__ == '__main__':
    editor_main()
