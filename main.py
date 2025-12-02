import pygame
import sys
import torch
import math
import itertools
import json
import os
from car import Car
from ai import Agent, MultiAgentManager
from editor import editor_main
from drawing import draw_track
from telemetry import telemetry

# --- Ayarlar ve Sabitler ---
CAR_NAMES = [
    "Türkiye", "Almanya", "Amerika", "Japonya",
    "İtalya", "İngiltere", "Fransa", "Brezilya", "İspanya", "Kanada"
]
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 880
FPS = 60

from drawing import draw_track, COLOR_WHITE

# Modern Color Scheme
COLOR_BACKGROUND = (15, 15, 25)  # Dark navy
COLOR_GRADIENT_START = (25, 25, 45)  # Darker navy
COLOR_GRADIENT_END = (35, 35, 65)  # Lighter navy
COLOR_CHECKPOINT = (0, 255, 0, 100)
COLOR_MENU_TEXT = (220, 220, 220)  # Light gray
COLOR_MENU_SELECTED = (0, 255, 150)  # Neon green
COLOR_MENU_HOVER = (100, 100, 255)  # Blue accent
COLOR_GOLD = (255, 215, 0)
COLOR_NEON_BLUE = (0, 200, 255)
COLOR_NEON_PURPLE = (200, 100, 255)
COLOR_SUCCESS = (50, 255, 50)
COLOR_WARNING = (255, 200, 50)
COLOR_ERROR = (255, 100, 100)

# AI Ayarları
TARGET_UPDATE = 10

# Oyun Durumları
STATE_MENU = "MENU"
STATE_SETTINGS = "SETTINGS"
STATE_EDITOR = "EDITOR"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RACING = "RACING"
STATE_GAME_OVER = "GAME_OVER"
STATE_PAUSE = "PAUSE"

# UI Animation System
class UIAnimation:
    def __init__(self):
        self.animations = {}
        
    def add_fade_in(self, element_id, duration=30):
        self.animations[element_id] = {
            'type': 'fade_in',
            'timer': 0,
            'duration': duration,
            'alpha': 0
        }
    
    def add_slide_in(self, element_id, start_pos, end_pos, duration=30):
        self.animations[element_id] = {
            'type': 'slide_in',
            'timer': 0,
            'duration': duration,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'current_pos': start_pos
        }
    
    def update(self):
        to_remove = []
        for element_id, anim in self.animations.items():
            anim['timer'] += 1
            progress = min(anim['timer'] / anim['duration'], 1.0)
            
            if anim['type'] == 'fade_in':
                anim['alpha'] = int(255 * progress)
            elif anim['type'] == 'slide_in':
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                anim['current_pos'] = (current_x, current_y)
            
            if progress >= 1.0:
                to_remove.append(element_id)
        
        for element_id in to_remove:
            del self.animations[element_id]
    
    def get_alpha(self, element_id):
        return self.animations.get(element_id, {}).get('alpha', 255)
    
    def get_pos(self, element_id):
        return self.animations.get(element_id, {}).get('current_pos', (0, 0))

def draw_gradient_rect(surface, color1, color2, rect):
    """Draw a vertical gradient rectangle"""
    for y in range(rect.height):
        ratio = y / rect.height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), 
                        (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))

def draw_modern_button(surface, text, rect, font, selected=False, hover=False):
    """Draw a modern button with gradient and effects"""
    # Button background
    if selected:
        color1, color2 = COLOR_NEON_BLUE, COLOR_NEON_PURPLE
        text_color = COLOR_BACKGROUND
    elif hover:
        color1, color2 = COLOR_MENU_HOVER, COLOR_GRADIENT_END
        text_color = COLOR_MENU_TEXT
    else:
        color1, color2 = COLOR_GRADIENT_START, COLOR_GRADIENT_END
        text_color = COLOR_MENU_TEXT
    
    # Draw gradient background
    draw_gradient_rect(surface, color1, color2, rect)
    
    # Draw border
    border_color = COLOR_NEON_BLUE if selected else COLOR_MENU_HOVER if hover else COLOR_MENU_TEXT
    pygame.draw.rect(surface, border_color, rect, 2)
    
    # Draw text
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)
    
    return rect

def draw_modern_panel(surface, rect, title="", alpha=200):
    """Draw a modern panel with transparency"""
    panel_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel_surface.fill((20, 20, 40, alpha))
    
    # Border gradient effect
    border_color = (*COLOR_NEON_BLUE, 150)
    pygame.draw.rect(panel_surface, border_color, panel_surface.get_rect(), 2)
    
    surface.blit(panel_surface, rect.topleft)
    
    if title:
        font = pygame.font.Font(None, 28)
        title_surface = font.render(title, True, COLOR_NEON_BLUE)
        title_rect = title_surface.get_rect(centerx=rect.centerx, y=rect.y + 10)
        surface.blit(title_surface, title_rect)

def show_notification(message, notification_type="info", duration=180):
    """Add a notification to the queue"""
    global notification_queue
    colors = {
        "info": COLOR_NEON_BLUE,
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARNING,
        "error": COLOR_ERROR
    }
    
    notification_queue.append({
        'message': message,
        'color': colors.get(notification_type, COLOR_NEON_BLUE),
        'timer': duration,
        'y_offset': 0
    })

# --- Veri ve Yardımcı Fonksiyonlar ---
def get_available_tracks():
    if not os.path.exists("tracks"):
        os.makedirs("tracks")
        return []
    return [f.split('.')[0] for f in os.listdir("tracks") if f.endswith('.json')]

def get_track_data(name):
    path = f"tracks/{name}.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        borders = [[pygame.math.Vector2(p) for p in path] for path in data["borders"]]
        checkpoints = [[pygame.math.Vector2(p) for p in pair] for pair in data["checkpoints"]]
        
        # Boost zones (optional)
        boost_zones = []
        if "boost_zones" in data:
            boost_zones = [[pygame.math.Vector2(p) for p in zone] for zone in data["boost_zones"]]
        
        return borders, checkpoints, boost_zones
    except Exception as e:
        print(f"Hata: {path} yüklenemedi veya formatı bozuk. {e}")
        return None, None, []

def get_track_bounding_box(track_borders, track_checkpoints):
    """Haritanın bounding box'ını hesapla"""
    all_points = []
    
    # Track borders'dan tüm noktaları ekle
    for border in track_borders:
        for point in border:
            all_points.append(point)
    
    # Checkpoints'lerden tüm noktaları ekle
    for checkpoint in track_checkpoints:
        for point in checkpoint:
            all_points.append(point)
    
    if not all_points:
        return 0, 0, 100, 100
    
    min_x = min(point.x for point in all_points)
    max_x = max(point.x for point in all_points)
    min_y = min(point.y for point in all_points)
    max_y = max(point.y for point in all_points)
    
    return min_x, min_y, max_x, max_y

def calculate_auto_fit_zoom(track_borders, track_checkpoints, screen_width, screen_height, margin_ratio=0.1):
    """Haritanın tamamen ekrana sığması için gerekli zoom seviyesini hesapla"""
    min_x, min_y, max_x, max_y = get_track_bounding_box(track_borders, track_checkpoints)
    
    track_width = max_x - min_x
    track_height = max_y - min_y
    
    if track_width == 0 or track_height == 0:
        return 1.0, pygame.math.Vector2(0, 0)
    
    # Margin için yer bırak
    available_width = screen_width * (1 - margin_ratio)
    available_height = screen_height * (1 - margin_ratio)
    
    # En kısıtlayıcı boyuta göre zoom hesapla
    zoom_x = available_width / track_width
    zoom_y = available_height / track_height
    zoom = min(zoom_x, zoom_y)
    
    # Haritanın merkezini hesapla
    track_center = pygame.math.Vector2((min_x + max_x) / 2, (min_y + max_y) / 2)
    
    return zoom, track_center

def world_to_screen(pos, camera, zoom, screen_width, screen_height):
    """Dünya koordinatlarını ekran koordinatlarına dönüştür"""
    return (pygame.math.Vector2(pos) - camera) * zoom + pygame.math.Vector2(screen_width, screen_height) / 2

def screen_to_world(pos, camera, zoom, screen_width, screen_height):
    """Ekran koordinatlarını dünya koordinatlarına dönüştür"""
    return (pygame.math.Vector2(pos) - pygame.math.Vector2(screen_width, screen_height) / 2) / zoom + camera

def resolve_collision(car1, car2):
    collision_normal = car1.position - car2.position
    if collision_normal.length() == 0: return
    unit_normal = collision_normal.normalize()
    unit_tangent = pygame.math.Vector2(-unit_normal.y, unit_normal.x)
    v1n = car1.velocity.dot(unit_normal); v1t = car1.velocity.dot(unit_tangent)
    v2n = car2.velocity.dot(unit_normal); v2t = car2.velocity.dot(unit_tangent)
    m1, m2 = car1.mass, car2.mass
    new_v1n = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
    new_v2n = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)
    car1.velocity = new_v1n * unit_normal + v1t * unit_tangent
    car2.velocity = new_v2n * unit_normal + v2t * unit_tangent
    overlap = (car1.rect.width + car2.rect.width) / 2 - collision_normal.length()
    if overlap > 0:
        car1.position += unit_normal * (overlap / 2)
        car2.position -= unit_normal * (overlap / 2)
    car1.reward -= 2.0
    car2.reward -= 2.0

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Racing Championship - Modern Edition")
    clock = pygame.time.Clock()
    
    # Modern font system
    font = pygame.font.Font(None, 36)
    leaderboard_font = pygame.font.Font(None, 24)
    countdown_font = pygame.font.Font(None, 200)
    large_font = pygame.font.Font(None, 100)
    small_font = pygame.font.Font(None, 50)
    menu_font = pygame.font.Font(None, 60)
    title_font = pygame.font.Font(None, 120)
    modern_font = pygame.font.Font(None, 40)
    
    # UI System
    ui_animations = UIAnimation()
    global notification_queue
    notification_queue = []

    available_tracks = get_available_tracks()
    if not available_tracks:
        print("Uyarı: 'tracks' klasöründe harita bulunamadı.")
        available_tracks = ["harita_yok"]

    settings = {
        "track": {"options": available_tracks, "current": 0},
        "laps": {"options": list(range(1, 11)), "current": 0},
        "cars": {"options": list(range(1, 9)), "current": 7},
        "collisions": {"options": ["Açık", "Kapalı"], "current": 0},
        "ai_training": {"options": ["Her Frame", "Her 2 Frame", "Her 3 Frame", "Her 5 Frame"], "current": 2},
        "ai_difficulty": {"options": ["Easy", "Medium", "Hard", "Expert"], "current": 1},
        "fps": {"options": [30, 45, 60], "current": 2},
    }
    
    cars = pygame.sprite.Group()
    agents, track_borders, track_checkpoints, boost_zones, car_list_for_agents = [], [], [], [], []
    ataturk_img, ataturk_rect = None, None
    
    # Kamera ve zoom değişkenleri
    camera = pygame.math.Vector2(0, 0)
    zoom = 1.0
    panning = False
    
    game_state = STATE_MENU
    menu_options = ["Başlat", "Harita Editörü", "Ayarlar", "Çıkış"]
    selected_menu_option = 0
    selected_setting_option = 0
    winner, frame_count, countdown_timer, last_countdown_time = None, 0, 5, 0
    draw_sensors, running = False, True
    
    # Performance monitoring variables
    fps_counter = 0
    fps_last_time = pygame.time.get_ticks()
    current_fps = 0
    show_performance = False
    show_ai_stats = False
    show_mini_map = False
    show_hud = True
    paused = False
    
    # Modern UI variables
    mouse_pos = (0, 0)
    button_hover_states = {}
    settings_categories = ["Graphics", "Gameplay", "Controls", "AI"]
    selected_settings_category = 0
    
    # Performance preset system
    performance_presets = ["Balanced", "Speed", "Handling"]
    current_performance_preset = 0

    def setup_simulation():
        nonlocal cars, agents, track_borders, track_checkpoints, boost_zones, game_state, ataturk_img, ataturk_rect, car_list_for_agents, winner, countdown_timer, last_countdown_time, camera, zoom
        
        track_name = settings["track"]["options"][settings["track"]["current"]]
        track_borders, track_checkpoints, boost_zones = get_track_data(track_name)
        if not track_borders:
            print(f"Hata: '{track_name}' yüklenemedi, ana menüye dönülüyor.")
            game_state = STATE_MENU
            return
        
        if boost_zones:
            print(f"Boost zones yüklendi: {len(boost_zones)} adet")
        
        # Otomatik zoom ve kamera pozisyonu hesapla
        auto_zoom, track_center = calculate_auto_fit_zoom(track_borders, track_checkpoints, SCREEN_WIDTH, SCREEN_HEIGHT)
        zoom = auto_zoom
        camera = track_center
        print(f"Auto-fit zoom: {zoom:.2f}, Camera center: ({camera.x:.1f}, {camera.y:.1f})")
        
        try:
            ataturk_img = pygame.image.load("Atatürk.jpg")
            ataturk_img = pygame.transform.scale(ataturk_img, (120, 150))
            ataturk_rect = ataturk_img.get_rect(topright=(SCREEN_WIDTH - 10, 10))
        except pygame.error as e:
            ataturk_img = None
        
        num_cars = settings["cars"]["options"][settings["cars"]["current"]]
        
        start_line = track_checkpoints[-1] 
        start_pos_p1, start_pos_p2 = start_line
        
        # Başlangıç açısını, başlangıç çizgisine dik olarak ayarla.
        # Bu, aracın ilk kontrol noktasına değil, pistin akış yönüne bakmasını sağlar.
        start_line_direction = start_pos_p2 - start_pos_p1
        # -90 derece rotasyon, pist tanımına göre doğru "ileri" yönünü verir.
        start_angle_vec = start_line_direction.rotate(90)
        start_angle = (math.degrees(math.atan2(-start_angle_vec.y, start_angle_vec.x)) + 360) % 360
        
        # Arabaları başlangıç çizgisine diz
        start_line_center = (start_pos_p1 + start_pos_p2) / 2

        cars.empty()
        car_list_for_agents.clear()
        for i in range(num_cars):
            t = (i + 1) / (num_cars + 1)
            start_pos = start_pos_p1.lerp(start_pos_p2, t)
            car_name = CAR_NAMES[i] if i < len(CAR_NAMES) else f"Car {i+1}"
            car = Car(i + 1, name=car_name, start_pos=start_pos, start_angle=start_angle)
            cars.add(car)
            car_list_for_agents.append(car)
        
        sample_state = car_list_for_agents[0].get_state(track_checkpoints)
        n_observations = len(sample_state)
        n_actions = 3 # Sabit olarak 3 eylem (İleri, Sol, Sağ)
        
        # AI difficulty setting
        difficulty_level = settings["ai_difficulty"]["options"][settings["ai_difficulty"]["current"]].lower()
        agents[:] = [Agent(n_observations, n_actions, difficulty=difficulty_level) for _ in range(num_cars)]
        
        print(f"AI Difficulty: {difficulty_level.title()}")
        
        winner, countdown_timer = None, 5
        last_countdown_time = pygame.time.get_ticks()
        game_state = STATE_COUNTDOWN

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # Mouse wheel zoom kontrolü (yarış ve countdown sırasında)
            if event.type == pygame.MOUSEWHEEL and game_state in [STATE_RACING, STATE_COUNTDOWN]:
                mouse_pos = pygame.mouse.get_pos()
                old_world_pos = screen_to_world(mouse_pos, camera, zoom, SCREEN_WIDTH, SCREEN_HEIGHT)
                zoom_change = 1.1 if event.y > 0 else 1/1.1
                zoom = max(0.1, min(zoom * zoom_change, 5.0))
                new_world_pos = screen_to_world(mouse_pos, camera, zoom, SCREEN_WIDTH, SCREEN_HEIGHT)
                camera += old_world_pos - new_world_pos
            
            # Orta mouse button ile panning (yarış ve countdown sırasında)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2 and game_state in [STATE_RACING, STATE_COUNTDOWN]:
                panning = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                panning = False
            if event.type == pygame.MOUSEMOTION and panning:
                camera -= pygame.math.Vector2(event.rel) / zoom
            
            if event.type == pygame.KEYDOWN:
                if game_state == STATE_MENU:
                    if event.key == pygame.K_UP: selected_menu_option = (selected_menu_option - 1) % len(menu_options)
                    elif event.key == pygame.K_DOWN: selected_menu_option = (selected_menu_option + 1) % len(menu_options)
                    elif event.key == pygame.K_RETURN:
                        if selected_menu_option == 0: setup_simulation()
                        elif selected_menu_option == 1: 
                            editor_main()
                            available_tracks = get_available_tracks()
                            if not available_tracks: available_tracks = ["harita_yok"]
                            settings["track"]["options"] = available_tracks
                            settings["track"]["current"] = 0
                        elif selected_menu_option == 2: game_state = STATE_SETTINGS
                        elif selected_menu_option == 3: running = False
                elif game_state == STATE_SETTINGS:
                    if event.key == pygame.K_ESCAPE: game_state = STATE_MENU
                    elif event.key == pygame.K_UP: selected_setting_option = (selected_setting_option - 1) % len(settings)
                    elif event.key == pygame.K_DOWN: selected_setting_option = (selected_setting_option + 1) % len(settings)
                    elif event.key == pygame.K_LEFT:
                        key = list(settings.keys())[selected_setting_option]
                        settings[key]["current"] = (settings[key]["current"] - 1) % len(settings[key]["options"])
                    elif event.key == pygame.K_RIGHT:
                        key = list(settings.keys())[selected_setting_option]
                        settings[key]["current"] = (settings[key]["current"] + 1) % len(settings[key]["options"])
                elif game_state in [STATE_RACING, STATE_COUNTDOWN]:
                    if event.key == pygame.K_s: draw_sensors = not draw_sensors
                    elif event.key == pygame.K_p: show_performance = not show_performance  # Toggle performance overlay
                    elif event.key == pygame.K_i: show_ai_stats = not show_ai_stats  # Toggle AI statistics overlay
                    elif event.key == pygame.K_m: show_mini_map = not show_mini_map  # Toggle mini-map
                    elif event.key == pygame.K_h: show_hud = not show_hud  # Toggle HUD
                    elif event.key == pygame.K_ESCAPE: 
                        paused = not paused
                        game_state = STATE_PAUSE if paused else STATE_RACING  # Pause menu
                    elif event.key == pygame.K_r:  # Reset view - otomatik fit
                        auto_zoom, track_center = calculate_auto_fit_zoom(track_borders, track_checkpoints, SCREEN_WIDTH, SCREEN_HEIGHT)
                        zoom = auto_zoom
                        camera = track_center
                        show_notification("View Reset", "info")
                    # Performance Preset Controls
                    elif event.key == pygame.K_1:  # Balanced Mode
                        current_performance_preset = 0
                        for car in cars:
                            car.apply_performance_preset("balanced")
                        show_notification("Performance: Balanced Mode", "success")
                    elif event.key == pygame.K_2:  # Speed Mode  
                        current_performance_preset = 1
                        for car in cars:
                            car.apply_performance_preset("speed")
                        show_notification("Performance: Speed Mode", "warning")
                    elif event.key == pygame.K_3:  # Handling Mode
                        current_performance_preset = 2
                        for car in cars:
                            car.apply_performance_preset("handling")
                        show_notification("Performance: Handling Mode", "info")
                    elif event.key == pygame.K_t:  # Cycle through presets
                        current_performance_preset = (current_performance_preset + 1) % len(performance_presets)
                        preset_name = performance_presets[current_performance_preset].lower()
                        for car in cars:
                            car.apply_performance_preset(preset_name)
                        show_notification(f"Performance: {performance_presets[current_performance_preset]} Mode", "success")
                    elif event.key == pygame.K_F5 and agents:  # Save AI models
                        show_notification("Saving AI models...", "info")
                        for i, agent in enumerate(agents):
                            filepath = f"ai_models/agent_{i}_model.pth"
                            agent.save_model(filepath)
                        show_notification("All AI models saved!", "success")
                    elif event.key == pygame.K_F9 and agents:  # Load AI models
                        show_notification("Loading AI models...", "info")
                        models_loaded = 0
                        for i, agent in enumerate(agents):
                            filepath = f"ai_models/agent_{i}_model.pth"
                            if agent.load_model(filepath):
                                models_loaded += 1
                        show_notification(f"Loaded {models_loaded}/{len(agents)} AI models!", "success" if models_loaded > 0 else "warning")
                elif game_state == STATE_PAUSE:
                    if event.key == pygame.K_ESCAPE: 
                        paused = False
                        game_state = STATE_RACING
                    elif event.key == pygame.K_q: 
                        game_state = STATE_MENU  # Quit to menu
                elif game_state == STATE_GAME_OVER: game_state = STATE_MENU
        
        # Klavye ile kamera hareketi (yarış ve countdown sırasında)
        if game_state in [STATE_RACING, STATE_COUNTDOWN]:
            keys = pygame.key.get_pressed()
            pan_speed = 20 / zoom  # Zoom seviyesine göre hız ayarla
            if keys[pygame.K_w] or keys[pygame.K_UP]: camera.y -= pan_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: camera.y += pan_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]: camera.x -= pan_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: camera.x += pan_speed
        
        if game_state == STATE_RACING:
            total_laps = settings["laps"]["options"][settings["laps"]["current"]]
            # Oyunu bitirme koşulunu tur sayısı 0'dan büyükse uygula
            if total_laps > 0:
                for car in cars:
                    if car.laps_completed >= total_laps:
                        winner = car.name
                        game_state = STATE_GAME_OVER
                        break
            if settings["collisions"]["options"][settings["collisions"]["current"]] == "Açık":
                for car1, car2 in itertools.combinations(cars, 2):
                    if car1.alive and car2.alive and pygame.sprite.collide_mask(car1, car2):
                        resolve_collision(car1, car2)

            # AI Training frequency control
            ai_training_intervals = [1, 2, 3, 5]  # Her Frame, Her 2 Frame, Her 3 Frame, Her 5 Frame
            ai_interval = ai_training_intervals[settings["ai_training"]["current"]]
            should_train_ai = (frame_count % ai_interval) == 0

            for i, car in enumerate(cars):
                if not car.alive: car.reset()
                agent = agents[i]
                state = car.get_state(track_checkpoints)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action_tensor = agent.select_action(state_tensor)
                action = action_tensor.item()
                car.move(action)
                lap_completed = car.update(track_borders, track_checkpoints, boost_zones)
                
                # Only train AI based on the frequency setting
                if should_train_ai:
                    reward = car.reward
                    reward_tensor = torch.tensor([reward], device=agent.device)
                    next_state = car.get_state(track_checkpoints) if car.alive else None
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0) if next_state is not None else None
                    agent.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
                    agent.optimize_model()
                    
                    # Update AI statistics
                    agent.update_stats(reward, lap_completed)

            if frame_count % TARGET_UPDATE == 0:
                for agent in agents:
                    agent.update_target_net()

        # Modern gradient background
        draw_gradient_rect(screen, COLOR_GRADIENT_START, COLOR_GRADIENT_END, 
                          pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        
        if game_state in [STATE_MENU, STATE_SETTINGS]:
            # Modern title with glow effect
            title_text = title_font.render("AI RACING", True, COLOR_GOLD)
            subtitle_text = modern_font.render("CHAMPIONSHIP", True, COLOR_NEON_BLUE)
            title_rect = title_text.get_rect(center=(SCREEN_WIDTH/2, 120))
            subtitle_rect = subtitle_text.get_rect(center=(SCREEN_WIDTH/2, 170))
            
            # Glow effect
            glow_surface = pygame.Surface((title_text.get_width() + 20, title_text.get_height() + 20), pygame.SRCALPHA)
            glow_surface.fill((255, 215, 0, 30))
            glow_rect = glow_surface.get_rect(center=title_rect.center)
            screen.blit(glow_surface, glow_rect)
            
            screen.blit(title_text, title_rect)
            screen.blit(subtitle_text, subtitle_rect)
            
            if game_state == STATE_MENU:
                # Modern button system
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                for i, option in enumerate(menu_options):
                    y_pos = 320 + i * 90
                    button_rect = pygame.Rect(SCREEN_WIDTH//2 - 200, y_pos - 30, 400, 60)
                    
                    # Check hover
                    is_hover = button_rect.collidepoint(mouse_x, mouse_y)
                    is_selected = i == selected_menu_option
                    
                    draw_modern_button(screen, option, button_rect, menu_font, is_selected, is_hover)
                
                # Add some info text
                info_text = leaderboard_font.render("Use ↑↓ arrows or mouse to navigate • Enter to select", True, COLOR_MENU_TEXT)
                info_rect = info_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT - 50))
                screen.blit(info_text, info_rect)
                
            elif game_state == STATE_SETTINGS:
                # Modern settings with categories
                settings_panel_rect = pygame.Rect(100, 250, SCREEN_WIDTH - 200, 500)
                draw_modern_panel(screen, settings_panel_rect, "Settings Configuration")
                
                y_start = 290
                for i, (key, value) in enumerate(settings.items()):
                    y_pos = y_start + i * 60
                    
                    # Setting name
                    key_text_str = key.replace("_", " ").title()
                    is_selected = i == selected_setting_option
                    
                    if is_selected:
                        # Highlight background for selected setting
                        highlight_rect = pygame.Rect(120, y_pos - 15, SCREEN_WIDTH - 240, 40)
                        highlight_surface = pygame.Surface((highlight_rect.width, highlight_rect.height), pygame.SRCALPHA)
                        highlight_surface.fill((*COLOR_NEON_BLUE, 50))
                        screen.blit(highlight_surface, highlight_rect)
                    
                    color = COLOR_NEON_BLUE if is_selected else COLOR_MENU_TEXT
                    key_text = modern_font.render(key_text_str, True, color)
                    key_rect = key_text.get_rect(midright=(SCREEN_WIDTH/2 - 100, y_pos))
                    screen.blit(key_text, key_rect)
                    
                    # Setting value with modern styling
                    value_text_str = f"◀ {value['options'][value['current']]} ▶"
                    value_color = COLOR_MENU_SELECTED if is_selected else COLOR_MENU_TEXT
                    value_text = modern_font.render(value_text_str, True, value_color)
                    value_rect = value_text.get_rect(midleft=(SCREEN_WIDTH/2 + 100, y_pos))
                    screen.blit(value_text, value_rect)
                
                # Help text
                help_panel_rect = pygame.Rect(150, SCREEN_HEIGHT - 120, SCREEN_WIDTH - 300, 80)
                draw_modern_panel(screen, help_panel_rect, alpha=150)
                
                help_texts = [
                    "↑↓ Navigate • ←→ Change Value • ESC Return to Menu",
                    "Changes are applied immediately"
                ]
                
                for i, help_text in enumerate(help_texts):
                    text_surface = leaderboard_font.render(help_text, True, COLOR_MENU_TEXT)
                    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT - 90 + i * 25))
                    screen.blit(text_surface, text_rect)
        else:
            # Track'i transform et ve çiz
            if track_borders and track_checkpoints:
                # Track borders'ı world-to-screen transform et
                transformed_borders = []
                for border in track_borders:
                    transformed_border = [world_to_screen(point, camera, zoom, SCREEN_WIDTH, SCREEN_HEIGHT) for point in border]
                    transformed_borders.append(transformed_border)
                
                # Track checkpoints'leri transform et
                transformed_checkpoints = []
                for checkpoint in track_checkpoints:
                    transformed_checkpoint = [world_to_screen(point, camera, zoom, SCREEN_WIDTH, SCREEN_HEIGHT) for point in checkpoint]
                    transformed_checkpoints.append(transformed_checkpoint)
                
                draw_track(screen, transformed_borders, transformed_checkpoints)
                
                if draw_sensors:
                    for i, (cp_start, cp_end) in enumerate(transformed_checkpoints[:-1]):
                        pygame.draw.line(screen, COLOR_CHECKPOINT, cp_start, cp_end, 15)
            
            # Araçları transform et ve çiz
            for car in cars:
                # Araç pozisyonunu transform et
                screen_pos = world_to_screen(car.position, camera, zoom, SCREEN_WIDTH, SCREEN_HEIGHT)
                # Yeni draw metodunu kullan
                car.draw(screen, screen_pos=screen_pos, camera=camera, zoom=zoom, 
                        screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT, draw_sensors=draw_sensors)
            if ataturk_img:
                screen.blit(ataturk_img, ataturk_rect)
            if game_state == STATE_RACING:
                total_laps = settings["laps"]["options"][settings["laps"]["current"]]
                def get_car_rank(car):
                    target_pos = (track_checkpoints[car.next_checkpoint_id][0] + track_checkpoints[car.next_checkpoint_id][1]) / 2
                    dist = car.position.distance_to(target_pos)
                    return (-car.laps_completed, -car.next_checkpoint_id, dist)
                sorted_cars = sorted(list(cars), key=get_car_rank)
                title_text = font.render("Sıralama:", True, COLOR_WHITE)
                screen.blit(title_text, (10, 10))
                y_offset = 40
                for i, car in enumerate(sorted_cars):
                    if i >= 8: break
                    rank_text = f"{i+1}. {car.name}" + (f" (Tur: {car.laps_completed})" if total_laps > 1 else "")
                    text_surface = leaderboard_font.render(rank_text, True, COLOR_WHITE)
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 25
                if total_laps > 1:
                    total_laps_text = leaderboard_font.render(f"Toplam Tur: {total_laps}", True, COLOR_WHITE)
                    screen.blit(total_laps_text, (10, y_offset + 10))
                
                # Performance overlay
                if show_performance:
                    # Calculate FPS
                    fps_counter += 1
                    if current_time - fps_last_time >= 1000:  # Update every second
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_last_time = current_time
                    
                    # Performance info
                    perf_y = SCREEN_HEIGHT - 180
                    ai_interval = [1, 2, 3, 5][settings["ai_training"]["current"]]
                    target_fps = settings["fps"]["options"][settings["fps"]["current"]]
                    
                    # Performance overlay background
                    perf_surf = pygame.Surface((300, 170), pygame.SRCALPHA)
                    perf_surf.fill((0, 0, 0, 180))
                    screen.blit(perf_surf, (SCREEN_WIDTH - 310, perf_y))
                    
                    # Performance text
                    perf_texts = [
                        f"FPS: {current_fps} / {target_fps}",
                        f"AI Training: {settings['ai_training']['options'][settings['ai_training']['current']]}",
                        f"Cars: {len(cars)}",
                        f"Zoom: {zoom:.2f}x",
                        f"Sensors: {'ON' if draw_sensors else 'OFF'}",
                        f"",
                        f"[P] Toggle Performance",
                        f"[I] AI Statistics",
                        f"[F5] Save AI Models",
                        f"[F9] Load AI Models"
                    ]
                    
                    for i, text in enumerate(perf_texts):
                        if text:  # Skip empty strings
                            color = COLOR_GOLD if i == 0 else COLOR_WHITE
                            text_surf = leaderboard_font.render(text, True, color)
                            screen.blit(text_surf, (SCREEN_WIDTH - 300, perf_y + 10 + i * 20))
                
                # AI Statistics overlay
                if show_ai_stats and agents:
                    ai_y = 180
                    
                    # AI overlay background
                    ai_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
                    ai_surf.fill((0, 0, 0, 180))
                    screen.blit(ai_surf, (10, ai_y))
                    
                    # Get stats from first agent (representative)
                    agent_stats = agents[0].get_learning_stats()
                    
                    # AI Statistics text
                    ai_texts = [
                        "=== AI LEARNING STATS ===",
                        f"Episodes: {agent_stats['episodes']}",
                        f"Total Laps: {agent_stats['laps']}",
                        f"Avg Reward: {agent_stats['avg_reward']:.2f}",
                        f"Best Reward: {agent_stats['best_reward']:.2f}",
                        f"Exploration: {agent_stats['exploration']:.1%}",
                        f"Memory: {agent_stats['memory_usage']}/20000",
                        f"Train Steps: {agent_stats['total_steps']}",
                        "",
                        "[I] Toggle AI Stats",
                        "[P] Performance Info"
                    ]
                    
                    for i, text in enumerate(ai_texts):
                        if text:  # Skip empty strings
                            if text.startswith("==="):
                                color = COLOR_GOLD
                            elif i == 0 or text.startswith("["):
                                color = COLOR_GOLD
                            else:
                                color = COLOR_WHITE
                            text_surf = leaderboard_font.render(text, True, color)
                            screen.blit(text_surf, (20, ai_y + 10 + i * 20))
            elif game_state == STATE_COUNTDOWN:
                if current_time - last_countdown_time > 1000:
                    countdown_timer -= 1
                    last_countdown_time = current_time
                    if countdown_timer < 0:
                        game_state = STATE_RACING
                text_to_show = str(countdown_timer) if countdown_timer > 0 else "GO!"
                countdown_text = countdown_font.render(text_to_show, True, COLOR_GOLD)
                text_rect = countdown_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
                screen.blit(countdown_text, text_rect)
            elif game_state == STATE_PAUSE:
                # Modern pause menu overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                screen.blit(overlay, (0, 0))
                
                # Pause menu panel
                pause_panel_rect = pygame.Rect(SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT//2 - 200, 500, 400)
                draw_modern_panel(screen, pause_panel_rect, "GAME PAUSED")
                
                # Pause menu options
                pause_options = [
                    "Resume Race",
                    "Performance Settings",
                    "View Controls",
                    "Main Menu"
                ]
                
                y_start = pause_panel_rect.y + 80
                for i, option in enumerate(pause_options):
                    y_pos = y_start + i * 60
                    option_rect = pygame.Rect(pause_panel_rect.x + 50, y_pos, pause_panel_rect.width - 100, 40)
                    
                    is_selected = i == 0  # Default to first option for now
                    draw_modern_button(screen, option, option_rect, modern_font, is_selected)
                
                # Performance preset indicator
                preset_text = f"Current Mode: {performance_presets[current_performance_preset]}"
                preset_surface = leaderboard_font.render(preset_text, True, COLOR_NEON_BLUE)
                preset_rect = preset_surface.get_rect(center=(pause_panel_rect.centerx, pause_panel_rect.bottom - 50))
                screen.blit(preset_surface, preset_rect)
                
                # Controls help
                controls_help = [
                    "[ESC] Resume • [Q] Main Menu",
                    "[1][2][3] Performance Modes • [T] Cycle Modes"
                ]
                
                for i, help_text in enumerate(controls_help):
                    help_surface = leaderboard_font.render(help_text, True, COLOR_MENU_TEXT)
                    help_rect = help_surface.get_rect(center=(pause_panel_rect.centerx, pause_panel_rect.bottom - 20 + i * 15))
                    screen.blit(help_surface, help_rect)
                    
            elif game_state == STATE_GAME_OVER:
                # Modern game over screen
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))
                
                # Game over panel
                game_over_rect = pygame.Rect(SCREEN_WIDTH//2 - 350, SCREEN_HEIGHT//2 - 150, 700, 300)
                draw_modern_panel(screen, game_over_rect, "RACE FINISHED")
                
                # Winner announcement with glow
                winner_text = large_font.render(f"🏆 WINNER: {winner} 🏆", True, COLOR_GOLD)
                winner_rect = winner_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 30))
                
                # Glow effect for winner
                glow_surface = pygame.Surface((winner_text.get_width() + 40, winner_text.get_height() + 20), pygame.SRCALPHA)
                glow_surface.fill((255, 215, 0, 40))
                glow_rect = glow_surface.get_rect(center=winner_rect.center)
                screen.blit(glow_surface, glow_rect)
                
                screen.blit(winner_text, winner_rect)
                
                # Return instruction
                return_text = modern_font.render("Press any key to return to Menu", True, COLOR_MENU_TEXT)
                return_rect = return_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 50))
                screen.blit(return_text, return_rect)
                
        # Update UI animations
        ui_animations.update()
        
        # Draw notifications
        if notification_queue:
            notification_y = 100
            notifications_to_remove = []
            
            for i, notification in enumerate(notification_queue):
                # Update notification timer
                notification['timer'] -= 1
                
                if notification['timer'] <= 0:
                    notifications_to_remove.append(i)
                    continue
                
                # Notification appearance animation
                alpha = min(255, notification['timer'] * 3) if notification['timer'] < 85 else 255
                if notification['timer'] < 30:
                    alpha = notification['timer'] * 8  # Fade out
                
                # Create notification surface
                notification_text = notification['message']
                text_surface = modern_font.render(notification_text, True, notification['color'])
                
                notification_width = text_surface.get_width() + 40
                notification_height = text_surface.get_height() + 20
                notification_rect = pygame.Rect(SCREEN_WIDTH - notification_width - 20, 
                                               notification_y + i * (notification_height + 10), 
                                               notification_width, notification_height)
                
                # Notification background with alpha
                notif_surface = pygame.Surface((notification_rect.width, notification_rect.height), pygame.SRCALPHA)
                bg_color = (*COLOR_BACKGROUND, min(200, alpha))
                border_color = (*notification['color'], min(255, alpha))
                
                notif_surface.fill(bg_color)
                pygame.draw.rect(notif_surface, border_color, notif_surface.get_rect(), 2)
                
                screen.blit(notif_surface, notification_rect)
                
                # Notification text with alpha
                text_surface_alpha = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
                text_surface_alpha.blit(text_surface, (0, 0))
                text_surface_alpha.set_alpha(alpha)
                
                text_rect = text_surface_alpha.get_rect(center=notification_rect.center)
                screen.blit(text_surface_alpha, text_rect)
            
            # Remove expired notifications
            for i in reversed(notifications_to_remove):
                notification_queue.pop(i)
        
        # Enhanced HUD elements
        if game_state == STATE_RACING and show_hud:
            # Performance preset indicator (top-right corner)
            preset_panel_rect = pygame.Rect(SCREEN_WIDTH - 200, 10, 180, 60)
            draw_modern_panel(screen, preset_panel_rect, alpha=150)
            
            preset_title = leaderboard_font.render("Performance Mode", True, COLOR_NEON_BLUE)
            preset_value = modern_font.render(performance_presets[current_performance_preset], True, COLOR_MENU_SELECTED)
            
            screen.blit(preset_title, (preset_panel_rect.x + 10, preset_panel_rect.y + 10))
            screen.blit(preset_value, (preset_panel_rect.x + 10, preset_panel_rect.y + 30))
            
            # Quick controls indicator (bottom-center)
            if frame_count % 300 < 150:  # Blinking hint
                controls_panel_rect = pygame.Rect(SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT - 50, 400, 30)
                controls_surface = pygame.Surface((controls_panel_rect.width, controls_panel_rect.height), pygame.SRCALPHA)
                controls_surface.fill((0, 0, 0, 100))
                screen.blit(controls_surface, controls_panel_rect)
                
                controls_text = "[1][2][3] Performance • [ESC] Pause • [P] Stats"
                controls_render = leaderboard_font.render(controls_text, True, COLOR_MENU_TEXT)
                controls_rect = controls_render.get_rect(center=controls_panel_rect.center)
                screen.blit(controls_render, controls_rect)
        
        # Mini-map placeholder (if enabled)
        if show_mini_map and game_state == STATE_RACING:
            minimap_rect = pygame.Rect(SCREEN_WIDTH - 220, SCREEN_HEIGHT - 220, 200, 200)
            draw_modern_panel(screen, minimap_rect, "Mini Map", alpha=180)
            
            # Mini-map content placeholder
            minimap_text = leaderboard_font.render("Mini-map", True, COLOR_MENU_TEXT)
            minimap_text_rect = minimap_text.get_rect(center=minimap_rect.center)
            screen.blit(minimap_text, minimap_text_rect)
        
        pygame.display.flip()
        # Dynamic FPS control based on settings
        target_fps = settings["fps"]["options"][settings["fps"]["current"]]
        clock.tick(target_fps)
        frame_count += 1

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
