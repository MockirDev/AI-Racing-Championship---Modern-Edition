import pygame
import math
import numpy as np

def line_intersection(p1, p2, p3, p4):
    """
    Checks if line segment 'p1p2' intersects with line segment 'p3p4'.
    Returns the intersection point or None.
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None # Lines are parallel

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    
    t = t_num / den
    u = u_num / den

    if 0 < t < 1 and u > 0:
        # We only care about u > 0 for the sensor ray, and 0 < t < 1 for the wall segment
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return pygame.math.Vector2(intersect_x, intersect_y)
        
    return None

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

class Car(pygame.sprite.Sprite):
    def __init__(self, car_id, name, start_pos, start_angle=0):
        super().__init__()
        self.name = name
        self.start_pos = start_pos
        self.start_angle = start_angle

        try:
            self.original_image = pygame.image.load(f'car_images/car{car_id}.png').convert_alpha()
        except pygame.error:
            self.original_image = pygame.Surface((40, 20), pygame.SRCALPHA)
            self.original_image.fill((255, 0, 0))
            
        self.image = pygame.transform.scale(self.original_image, (40, 20))
        self.mask = pygame.mask.from_surface(self.image)
        self.font = pygame.font.Font(None, 20)
        
        # Fizik özellikleri
        self.mass = 1.0

        # AI ve Durum özellikleri
        self.alive = True
        self.next_checkpoint_id = 0
        self.laps_completed = 0
        self.reward = 0.0

        # Boost Zone System
        self.boost_active = False
        self.boost_timer = 0
        self.boost_max_duration = 60  # frames
        self.boost_multiplier = 1.5
        self.last_boost_zone = None

        # Car Customization System
        self.customization = {
            "performance": {
                "preset": "balanced",  # speed, balanced, handling
                "acceleration_modifier": 1.0,
                "max_speed_modifier": 1.0,
                "rotation_speed_modifier": 1.0,
                "friction_modifier": 1.0
            },
            "visual": {
                "color": (255, 255, 255),  # Default white
                "trail_enabled": False,
                "trail_color": (0, 255, 255),
                "theme": "default"
            },
            "progression": {
                "xp": 0,
                "level": 1,
                "unlocked_upgrades": ["basic_tuning"]
            }
        }
        
        # Performance presets
        self.performance_presets = {
            "speed": {
                "acceleration_modifier": 1.3,
                "max_speed_modifier": 1.4,
                "rotation_speed_modifier": 0.8,
                "friction_modifier": 0.95
            },
            "balanced": {
                "acceleration_modifier": 1.0,
                "max_speed_modifier": 1.0,
                "rotation_speed_modifier": 1.0,
                "friction_modifier": 1.0
            },
            "handling": {
                "acceleration_modifier": 0.9,
                "max_speed_modifier": 0.8,
                "rotation_speed_modifier": 1.4,
                "friction_modifier": 1.1
            }
        }

        # Trail system for visual effects
        self.trail_points = []
        self.max_trail_length = 20

        # Sensörler
        self.sensor_angles = [-90, -45, 0, 45, 90]
        self.sensor_len = 300
        self.sensors = [0.0] * len(self.sensor_angles)

        self.reset()

    def apply_performance_preset(self, preset_name):
        """Performans preset'ini uygular"""
        if preset_name in self.performance_presets:
            preset = self.performance_presets[preset_name]
            self.customization["performance"] = preset.copy()
            self.customization["performance"]["preset"] = preset_name
            self._update_performance_stats()
            print(f"{self.name} performans preset'i değiştirildi: {preset_name.title()}")

    def set_custom_performance(self, **kwargs):
        """Özel performans ayarlarını uygular"""
        for key, value in kwargs.items():
            if key in self.customization["performance"]:
                self.customization["performance"][key] = value
        self._update_performance_stats()

    def _update_performance_stats(self):
        """Performans modifierlarını gerçek stats'lara uygular"""
        perf = self.customization["performance"]
        
        # Base values (default)
        base_acceleration = 0.15
        base_max_speed = 4.0
        base_rotation_speed = 4.0
        base_friction = 0.98
        
        # Apply modifiers
        self.acceleration = base_acceleration * perf["acceleration_modifier"]
        self.max_speed = base_max_speed * perf["max_speed_modifier"]
        self.rotation_speed = base_rotation_speed * perf["rotation_speed_modifier"]
        self.friction = base_friction * perf["friction_modifier"]

    def set_visual_customization(self, **kwargs):
        """Görsel özelleştirmeleri ayarlar"""
        for key, value in kwargs.items():
            if key in self.customization["visual"]:
                self.customization["visual"][key] = value
        
        # Renk değişimi için image'i güncelle
        if "color" in kwargs:
            self._update_car_color(kwargs["color"])

    def _update_car_color(self, color):
        """Araç rengini değiştirir"""
        # Original image'i al ve renk uygula
        colored_image = self.original_image.copy()
        
        # Renk overlay uygula
        color_surface = pygame.Surface(colored_image.get_size(), pygame.SRCALPHA)
        color_surface.fill((*color, 128))  # Alpha ile karıştır
        colored_image.blit(color_surface, (0, 0), special_flags=pygame.BLEND_MULTIPLY)
        
        self.image = pygame.transform.scale(colored_image, (40, 20))

    def _update_trail_system(self):
        """Trail sistemi güncelleme"""
        if self.customization["visual"]["trail_enabled"] and self.alive:
            # Mevcut pozisyonu trail'e ekle
            self.trail_points.append(self.position.copy())
            
            # Trail uzunluğunu sınırla
            if len(self.trail_points) > self.max_trail_length:
                self.trail_points.pop(0)
        else:
            # Trail kapalıysa listeyi temizle
            if self.trail_points:
                self.trail_points.clear()

    def _draw_trail(self, screen, camera, zoom, screen_width, screen_height):
        """Trail efektini çizer"""
        if len(self.trail_points) < 2:
            return
            
        trail_color = self.customization["visual"]["trail_color"]
        
        for i in range(len(self.trail_points) - 1):
            # Trail alpha efekti - eskileri soluklaştır
            alpha = int((i / len(self.trail_points)) * 255)
            
            # World to screen dönüşümü
            start_screen = self.world_to_screen_local(self.trail_points[i], camera, zoom, screen_width, screen_height)
            end_screen = self.world_to_screen_local(self.trail_points[i + 1], camera, zoom, screen_width, screen_height)
            
            # Trail line çiz
            trail_surface = pygame.Surface((abs(end_screen[0] - start_screen[0]) + 4, abs(end_screen[1] - start_screen[1]) + 4), pygame.SRCALPHA)
            pygame.draw.line(trail_surface, (*trail_color, alpha), (2, 2), (end_screen[0] - start_screen[0] + 2, end_screen[1] - start_screen[1] + 2), 3)
            screen.blit(trail_surface, (min(start_screen[0], end_screen[0]) - 2, min(start_screen[1], end_screen[1]) - 2))

    def add_xp(self, amount):
        """XP ekler ve level kontrolü yapar"""
        old_level = self.customization["progression"]["level"]
        self.customization["progression"]["xp"] += amount
        
        # Level hesaplama (her level için 100 XP)
        new_level = (self.customization["progression"]["xp"] // 100) + 1
        self.customization["progression"]["level"] = new_level
        
        if new_level > old_level:
            print(f"{self.name} seviye atladı! Yeni seviye: {new_level}")
            self._unlock_new_upgrades(new_level)

    def _unlock_new_upgrades(self, level):
        """Yeni level'da upgrade'leri unlock eder"""
        unlocks = self.customization["progression"]["unlocked_upgrades"]
        
        if level >= 2 and "color_customization" not in unlocks:
            unlocks.append("color_customization")
            print(f"{self.name} renk özelleştirmesi unlocked!")
            
        if level >= 3 and "trail_effects" not in unlocks:
            unlocks.append("trail_effects")
            print(f"{self.name} trail efektleri unlocked!")
            
        if level >= 5 and "advanced_tuning" not in unlocks:
            unlocks.append("advanced_tuning")
            print(f"{self.name} gelişmiş tuning unlocked!")

    def update(self, track_borders, checkpoints, boost_zones=None):
        if not self.alive:
            return False

        # Boost zone kontrolü
        if boost_zones:
            self._update_boost_zones(boost_zones)

        # Boost timer güncelle
        if self.boost_active:
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.boost_active = False
                print(f"{self.name} boost sona erdi!")

        # Hız ve pozisyonu güncelle
        self.velocity *= self.friction
        if self.velocity.length() < 0.05:
            self.velocity.x = 0
            self.velocity.y = 0
        self.position += self.velocity

        # Döndürülmüş görüntü, rect ve maskeyi güncelle (çarpışma tespiti için)
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.rotated_image)

        # Sensörleri ve çarpışmaları kontrol et
        self._update_sensors(track_borders)
        self.check_collision(track_borders)
        
        # Trail sistemi güncelle
        self._update_trail_system()
        
        return self.check_checkpoints(checkpoints)

    def _update_boost_zones(self, boost_zones):
        """Boost zone algılama ve aktivasyon"""
        for i, boost_zone in enumerate(boost_zones):
            if point_in_polygon(self.position, boost_zone):
                if self.last_boost_zone != i and not self.boost_active:
                    self.boost_active = True
                    self.boost_timer = self.boost_max_duration
                    self.last_boost_zone = i
                    self.reward += 5.0  # Boost zone reward
                    print(f"{self.name} boost zone girdi! Hız artırıldı!")
                return
        
        # Hiçbir boost zone'da değilse
        self.last_boost_zone = None

    def draw(self, screen, screen_pos=None, camera=None, zoom=None, screen_width=None, screen_height=None, draw_sensors=False):
        if not self.alive:
            return
        
        # Eğer screen_pos verilmişse, transform edilmiş koordinatları kullan
        if screen_pos is not None:
            display_pos = screen_pos
        else:
            display_pos = self.position
        
        # Trail efekti çiz (araçtan önce)
        if camera is not None and zoom is not None:
            self._draw_trail(screen, camera, zoom, screen_width, screen_height)
            
        # Boost efekti - araç rengini değiştir
        if self.boost_active:
            # Boost aktifken turuncu/sarı glow efekti
            boost_surface = pygame.Surface((50, 30), pygame.SRCALPHA)
            glow_color = (255, 200, 0, 100)  # Turuncu glow
            pygame.draw.ellipse(boost_surface, glow_color, (0, 0, 50, 30))
            glow_rect = boost_surface.get_rect(center=display_pos)
            screen.blit(boost_surface, glow_rect.topleft)
            
        # Araç görüntüsünü sabit boyutta çiz (40x20 px)
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        
        # Boost aktifken araç rengini değiştir
        if self.boost_active:
            # Boost sırasında daha parlak renk
            boost_tint = pygame.Surface(rotated_image.get_size(), pygame.SRCALPHA)
            boost_tint.fill((255, 255, 150, 80))  # Sarımsı parlaklık
            rotated_image.blit(boost_tint, (0, 0), special_flags=pygame.BLEND_ADD)
            
        img_rect = rotated_image.get_rect(center=display_pos)
        screen.blit(rotated_image, img_rect.topleft)

        # İsmi çizdir (araç üzerinde sabit pozisyonda)
        name_color = (255, 255, 100) if self.boost_active else (255, 255, 255)
        name_text = self.font.render(self.name, True, name_color)
        text_rect = name_text.get_rect(center=(display_pos.x, display_pos.y - 25))
        screen.blit(name_text, text_rect)

        # Level ve XP göster (araç isminin altında)
        level_text = f"Lvl:{self.customization['progression']['level']} XP:{self.customization['progression']['xp']%100}/100"
        level_surface = pygame.font.Font(None, 16).render(level_text, True, (200, 200, 200))
        level_rect = level_surface.get_rect(center=(display_pos.x, display_pos.y - 10))
        screen.blit(level_surface, level_rect)

        # Boost timer göster
        if self.boost_active:
            timer_text = f"BOOST: {self.boost_timer//10 + 1}"
            timer_surface = self.font.render(timer_text, True, (255, 255, 0))
            timer_rect = timer_surface.get_rect(center=(display_pos.x, display_pos.y + 35))
            screen.blit(timer_surface, timer_rect)

        # Performance preset göster (kısa süre için)
        if hasattr(self, '_preset_display_timer') and self._preset_display_timer > 0:
            preset_text = f"Mode: {self.customization['performance']['preset'].title()}"
            preset_surface = self.font.render(preset_text, True, (0, 255, 255))
            preset_rect = preset_surface.get_rect(center=(display_pos.x, display_pos.y + 50))
            screen.blit(preset_surface, preset_rect)
            self._preset_display_timer -= 1

        # Sensörler için transform gerekli
        if draw_sensors and camera is not None and zoom is not None:
            for i, angle in enumerate(self.sensor_angles):
                sensor_angle_rad = math.radians(self.angle + angle)
                # World coordinates
                world_start_pos = self.position
                world_end_pos = world_start_pos + pygame.math.Vector2(
                    math.cos(sensor_angle_rad) * self.sensors[i],
                    math.sin(sensor_angle_rad) * self.sensors[i]
                )
                # Transform to screen coordinates
                screen_start = self.world_to_screen_local(world_start_pos, camera, zoom, screen_width, screen_height)
                screen_end = self.world_to_screen_local(world_end_pos, camera, zoom, screen_width, screen_height)
                
                pygame.draw.line(screen, (255, 255, 0, 100), screen_start, screen_end, 1)
    
    def world_to_screen_local(self, pos, camera, zoom, screen_width, screen_height):
        """Koordinat dönüşümü için yardımcı fonksiyon"""
        return (pygame.math.Vector2(pos) - camera) * zoom + pygame.math.Vector2(screen_width, screen_height) / 2

    def move(self, action):
        # Eylemler: 
        # 0: İleri
        # 1: Sola Dön
        # 2: Sağa Dön
        
        # Boost aktifse hızlanma artır
        current_acceleration = self.acceleration
        current_max_speed = self.max_speed
        
        if self.boost_active:
            current_acceleration *= self.boost_multiplier
            current_max_speed *= self.boost_multiplier
        
        # Araç her zaman ileri gitmeye çalışır (agresif mod)
        forward_x = current_acceleration * math.cos(math.radians(self.angle))
        forward_y = current_acceleration * math.sin(math.radians(self.angle))
        
        self.velocity.x += forward_x
        self.velocity.y += forward_y
        
        # Anti-reverse mechanism: Geriye doğru gitme cezası
        forward_dir = pygame.math.Vector2(math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
        if self.velocity.length() > 0:
            velocity_normalized = self.velocity.normalize()
            # Eğer hız vektörü ileri yönün tersindeyse (dot product < 0), geriye gidiyor demektir
            dot_product = forward_dir.dot(velocity_normalized)
            if dot_product < -0.5:  # Geriye doğru gidiyor
                self.velocity *= 0.8  # Hızı azalt
                self.reward -= 0.1    # Ceza ver
        
        # Hız limitini uygula (boost durumuna göre)
        if self.velocity.length() > current_max_speed:
            self.velocity.scale_to_length(current_max_speed)

        # Dönüşler (Sadece hareket halindeyken)
        if self.velocity.length() > 0.2:
            if action == 1: # Sola
                self.angle -= self.rotation_speed
            elif action == 2: # Sağa
                self.angle += self.rotation_speed
    
    def _update_sensors(self, track_borders):
        for i, angle in enumerate(self.sensor_angles):
            sensor_angle_rad = math.radians(self.angle + angle)
            start_pos = self.position
            end_pos = start_pos + pygame.math.Vector2(
                math.cos(sensor_angle_rad) * self.sensor_len,
                math.sin(sensor_angle_rad) * self.sensor_len
            )

            min_dist = self.sensor_len
            for border in track_borders:
                for j in range(len(border)):
                    p1 = border[j]
                    p2 = border[(j + 1) % len(border)]
                    intersect_point = line_intersection(p1, p2, start_pos, end_pos)
                    if intersect_point:
                        dist = self.position.distance_to(intersect_point)
                        if dist < min_dist:
                            min_dist = dist
            
            self.sensors[i] = min_dist

    def check_collision(self, track_borders):
        if min(self.sensors) < 10:
            self.alive = False
            self.reward = -10.0
    
    def check_checkpoints(self, checkpoints):
        if not self.alive:
            return False

        target_checkpoint_line = checkpoints[self.next_checkpoint_id]
        target_pos = (target_checkpoint_line[0] + target_checkpoint_line[1]) / 2
        
        # Base reward/penalty per step
        self.reward = -0.01  # Small time penalty to encourage faster completion

        # === SPEED REWARDS ===
        speed_ratio = self.velocity.length() / self.max_speed
        if speed_ratio > 0.3:  # Reward for maintaining good speed
            speed_reward = speed_ratio * 0.03
            self.reward += speed_reward
        else:
            self.reward -= 0.02  # Penalty for being too slow

        # === BOOST REWARDS ===
        if self.boost_active:
            self.reward += 0.02  # Extra reward for using boost effectively

        # === DIRECTION & PROGRESS REWARDS ===
        vec_to_target = target_pos - self.position
        distance_to_target = vec_to_target.length()
        
        # Track previous distance for progress measurement
        if not hasattr(self, 'prev_distance_to_target'):
            self.prev_distance_to_target = distance_to_target
        
        # Progress reward: getting closer to target
        progress = self.prev_distance_to_target - distance_to_target
        if progress > 0:
            self.reward += progress * 0.01  # Reward progress toward checkpoint
        else:
            self.reward -= abs(progress) * 0.005  # Small penalty for moving away
        
        self.prev_distance_to_target = distance_to_target
        
        # Direction alignment reward
        if vec_to_target.length() > 0:
            car_forward_vec = pygame.math.Vector2(math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
            direction_reward = car_forward_vec.dot(vec_to_target.normalize())
            
            if direction_reward > 0:
                self.reward += direction_reward * 0.04  # Reward for facing target
            else:
                self.reward += direction_reward * 0.06  # Penalty for facing away

        # === SMOOTH DRIVING REWARDS ===
        # Penalty for excessive turning (jerky movements)
        if hasattr(self, 'prev_angle'):
            angle_change = abs(self.angle - self.prev_angle)
            if angle_change > 180:  # Handle angle wraparound
                angle_change = 360 - angle_change
            
            if angle_change > 5:  # Penalize sharp turns
                self.reward -= (angle_change / 180) * 0.03
        
        self.prev_angle = self.angle

        # === SENSOR-BASED REWARDS ===
        # Penalty for getting too close to walls
        min_sensor_dist = min(self.sensors)
        if min_sensor_dist < 50:
            danger_penalty = (50 - min_sensor_dist) / 50 * 0.05
            self.reward -= danger_penalty

        # === CHECKPOINT COMPLETION ===
        car_line_p1 = self.position
        car_line_p2 = self.position + self.velocity * 2
        
        if self.velocity.length() > 0 and line_intersection(car_line_p1, car_line_p2, target_checkpoint_line[0], target_checkpoint_line[1]):
            # Dynamic checkpoint reward based on performance
            base_checkpoint_reward = 10.0
            speed_bonus = speed_ratio * 2.0  # Bonus for completing checkpoint at high speed
            boost_bonus = 3.0 if self.boost_active else 0.0  # Extra bonus for boost usage
            self.reward = base_checkpoint_reward + speed_bonus + boost_bonus
            
            # XP reward for checkpoint completion
            checkpoint_xp = int(5 + (speed_bonus * 2) + boost_bonus)
            self.add_xp(checkpoint_xp)
            
            self.next_checkpoint_id += 1
            
            if self.next_checkpoint_id >= len(checkpoints):
                self.next_checkpoint_id = 0
                self.laps_completed += 1
                # Lap completion bonus
                lap_bonus = 50.0 + (speed_ratio * 10.0) + (5.0 if self.boost_active else 0.0)
                self.reward = lap_bonus
                
                # Extra XP for lap completion
                lap_xp = int(25 + (speed_ratio * 15) + (10 if self.boost_active else 0))
                self.add_xp(lap_xp)
                
                print(f"{self.name} bir tur tamamladı!")
                return True
        
        return False

    def get_state(self, checkpoints):
        # Sensör verileri
        state = [s / self.sensor_len for s in self.sensors]
        
        # Hız bilgisi
        state.append(self.velocity.length() / self.max_speed)
        
        # Boost durumu
        state.append(1.0 if self.boost_active else 0.0)
        state.append(self.boost_timer / self.boost_max_duration if self.boost_active else 0.0)
        
        # --- Birinci Hedef Bilgisi ---
        target_checkpoint_line_1 = checkpoints[self.next_checkpoint_id]
        target_pos_1 = (target_checkpoint_line_1[0] + target_checkpoint_line_1[1]) / 2
        
        vec_to_target_1 = target_pos_1 - self.position
        dist_to_target_1 = vec_to_target_1.length()
        
        # Hedefin açısı (aracın kendi açısına göre normalize edilmiş)
        # Pygame'de y ekseni ters olduğu için -vec.y kullanılır
        angle_to_target_1 = math.degrees(math.atan2(-vec_to_target_1.y, vec_to_target_1.x))
        relative_angle_1 = (angle_to_target_1 - self.angle + 180) % 360 - 180
        
        state.append(relative_angle_1 / 180.0) # -1 ile 1 arasında normalize et
        state.append(dist_to_target_1 / 1000.0) # Uzaklığı kabaca normalize et

        # --- İkinci Hedef Bilgisi (Vizyon) ---
        next_next_checkpoint_id = (self.next_checkpoint_id + 1) % len(checkpoints)
        target_checkpoint_line_2 = checkpoints[next_next_checkpoint_id]
        target_pos_2 = (target_checkpoint_line_2[0] + target_checkpoint_line_2[1]) / 2

        vec_to_target_2 = target_pos_2 - self.position
        dist_to_target_2 = vec_to_target_2.length()

        angle_to_target_2 = math.degrees(math.atan2(-vec_to_target_2.y, vec_to_target_2.x))
        relative_angle_2 = (angle_to_target_2 - self.angle + 180) % 360 - 180
        
        state.append(relative_angle_2 / 180.0)
        state.append(dist_to_target_2 / 1000.0)
        
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.rotated_image = pygame.transform.rotate(self.image, -self.start_angle)
        self.rect = self.rotated_image.get_rect(center=self.start_pos)
        self.position = pygame.math.Vector2(self.start_pos)
        self.velocity = pygame.math.Vector2(0, 0)
        self.angle = self.start_angle
        self.alive = True
        self.next_checkpoint_id = 0
        self.laps_completed = 0
        self.reward = 0.0
        
        # Boost zone reset
        self.boost_active = False
        self.boost_timer = 0
        self.last_boost_zone = None
        
        # Trail sistemi reset
        self.trail_points.clear()
        
        # Customization ayarlarını uygula
        self._update_performance_stats()
        
        # Preset display timer reset
        self._preset_display_timer = 0
