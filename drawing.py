import pygame

# Renkler
COLOR_GRASS = (20, 140, 40)
COLOR_ASPHALT = (50, 50, 50)
COLOR_BORDERS = (200, 200, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

def draw_finish_line(screen, line):
    p1, p2 = line
    length = p1.distance_to(p2)
    vec = (p2 - p1).normalize()
    perp_vec = pygame.math.Vector2(-vec.y, vec.x) * 7.5
    num_checkers = 10
    checker_size = length / num_checkers
    for i in range(num_checkers):
        color = COLOR_WHITE if i % 2 == 0 else COLOR_BLACK
        start_pos = p1 + vec * (i * checker_size)
        end_pos = p1 + vec * ((i + 1) * checker_size)
        points = [start_pos - perp_vec, start_pos + perp_vec, end_pos + perp_vec, end_pos - perp_vec]
        pygame.draw.polygon(screen, color, points)

def draw_track(screen, track_borders, track_checkpoints):
    screen.fill(COLOR_GRASS)
    if track_borders:
        # Only draw polygons if there are at least 3 points
        if len(track_borders[0]) >= 3:
            pygame.draw.polygon(screen, COLOR_ASPHALT, track_borders[0])
        if len(track_borders) > 1 and len(track_borders[1]) >= 3:
            pygame.draw.polygon(screen, COLOR_GRASS, track_borders[1])
        
        # Draw border lines only if there are at least 2 points
        for border in track_borders:
            if len(border) >= 2:
                pygame.draw.lines(screen, COLOR_BORDERS, True, border, 3)
    
    if track_checkpoints and len(track_checkpoints[-1]) == 2:
        draw_finish_line(screen, track_checkpoints[-1])
