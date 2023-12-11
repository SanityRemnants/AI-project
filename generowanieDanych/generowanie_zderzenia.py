import pygame
import sys
import random
import math
import time


from pygame.math import Vector2
from ElasticCollision.ec_game import momentum_trigonometry


def angle_with_x_axis(vector):
    angle = math.atan2(vector[1], vector[0])
    return angle


# Inicjalizacja Pygame
pygame.init()

# Ustawienia okna
window_width = 600
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Poruszające się kule")
balls_number = 20

# Ustawienia kuli
ball_radius = 10
ball_color = (255, 0, 0)

ball_positions = [[0,0] for _ in range(balls_number)]
for i in range(balls_number):
    m = 50*i
    a = 0
    while m > window_width - 100:
        m =  m - (window_width - 50)
        a +=1
    ball_positions[i] = [50 + m, window_height // 2 + a*50]

ball_speed_magnitude = 1.5 # Magnituda prędkości kulki

balls_speed_x = [0] * balls_number
balls_speed_y = [0] * balls_number

# Ustawienia ramki
frame_thickness = 10
frame_color = (0, 0, 0)
frame_rect = pygame.Rect(
    frame_thickness, frame_thickness, window_width - 2 * frame_thickness, window_height - 2 * frame_thickness
)

# Ustawienia pola do odbijania
field_color = (0, 255, 0)
field_rect = pygame.Rect(
    2 * frame_thickness,
    2 * frame_thickness,
    window_width - 4 * frame_thickness,
    window_height - 4 * frame_thickness,
)
last_hit = [0] * balls_number
# Główna pętla programu
petla = True
counter = 0
f = open("data.csv", "a")
while counter<20000:
    start_time = time.time()

    # Wyczyszczenie ekranu
    window.fill((255, 255, 255))

    # Narysowanie ramki
    pygame.draw.rect(window, frame_color, frame_rect)

    # Narysowanie pola do odbijania
    pygame.draw.rect(window, field_color, field_rect)

    # Narysowanie kul
    for nr in range(balls_number):
        pos = ball_positions[nr]
        if nr == 0:
            pygame.draw.circle(window, (255,255,255), (int(pos[0]), int(pos[1])), ball_radius)
        else:
            pygame.draw.circle(window, ball_color, (int(pos[0]), int(pos[1])), ball_radius)

    # Aktualizacja ekranu
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Wygenerowanie losowego kierunku prędkości kulki po naciśnięciu przycisku myszy

            ball_speed_directions = random.uniform(0, 2 * math.pi)
            balls_speed_x[0] = ball_speed_magnitude * math.cos(ball_speed_directions)
            balls_speed_y[0] = ball_speed_magnitude * math.sin(ball_speed_directions)
    prev_time = [time.time()]*balls_number
    for i in range(len(ball_positions)):
        if balls_speed_x[i] + balls_speed_y[i] != 0:
            # Poruszanie kulki
            cur_time = time.time()
            delta = cur_time - prev_time[i]
            prev_time[i] = cur_time
            ball_positions[i][0] += balls_speed_x[i]
            ball_positions[i][1] += balls_speed_y[i]



                # Sprawdzanie zderzenia między kulami
            for j in range(len(ball_positions)):
                if i != j:
                    distance = math.sqrt((ball_positions[i][0] - ball_positions[j][0]) ** 2 +
                                         (ball_positions[i][1] - ball_positions[j][1]) ** 2)
                    if distance <= 2 * ball_radius:
                        if last_hit[i] != j + 1:
                            last_hit[i] = j + 1
                            last_hit[j] = i + 1
                            # Obliczanie wektora między kulami
                            collision_vector = [ball_positions[i][0] - ball_positions[j][0],
                                                ball_positions[i][1] - ball_positions[j][1]]
                            collision_distance = math.sqrt(collision_vector[0] ** 2 + collision_vector[1] ** 2)

                            # Normalizowanie wektora kolizji
                            normalized_collision_vector = [collision_vector[0] / collision_distance,
                                                           collision_vector[1] / collision_distance]

                            if distance < 2 * ball_radius:
                                c = 2 * ball_radius - abs(distance)
                                przes_y = (c * abs(ball_positions[i][1] - ball_positions[j][1]))
                                c1 = (c ** 2)
                                c2 = (przes_y ** 2)
                                przes_x = math.sqrt(abs(c1 - c2))
                                if ball_positions[i][1] < ball_positions[j][1]:
                                    ball_positions[i][1] -= przes_y
                                elif ball_positions[i][1] > ball_positions[j][1]:
                                    ball_positions[i][1] += przes_y
                                if ball_positions[i][0] < ball_positions[j][0]:
                                    ball_positions[i][0] -= przes_x
                                elif ball_positions[i][0] > ball_positions[j][0]:
                                    ball_positions[i][0] += przes_x
                            print(counter)
                            counter +=1
                            f.write(str(normalized_collision_vector[0]) + "," + str(normalized_collision_vector[1]) + "," + str(balls_speed_x[i]) +
                                    "," + str(balls_speed_y[i]) + "," + str(balls_speed_x[j]) + "," + str(balls_speed_y[j]) + ",")

                            vector1 = Vector2(balls_speed_x[i], balls_speed_y[i])
                            centre1 = Vector2(ball_positions[i][0], ball_positions[i][1])
                            vector2 = Vector2(balls_speed_x[j], balls_speed_y[j])
                            centre2 = Vector2(ball_positions[j][0], ball_positions[j][1])
                            mass1 = 1.0
                            mass2 = 1.0

                            v11, v12 = momentum_trigonometry(centre1, centre2, vector1, vector2, mass1, mass2, False)

                            balls_speed_x[i] = v11[0]
                            balls_speed_y[i] = v11[1]
                            balls_speed_x[j] = v12[0]
                            balls_speed_y[j] = v12[1]

                            f.write(str(balls_speed_x[i]) +
                                    "," + str(balls_speed_y[i]) + "," + str(balls_speed_x[j]) + "," + str(
                                balls_speed_y[j]) + "\n")

# Odbijanie kulki od ścianek
            if ball_positions[i][0] <= field_rect.left + ball_radius:
                if balls_speed_x[i] < 0:
                    ball_positions[i][0] = field_rect.left + ball_radius
                    last_hit[i] = 0
                    balls_speed_x[i] = abs(balls_speed_x[i])
            elif ball_positions[i][0] >= field_rect.right - ball_radius:
                if balls_speed_x[i] > 0:
                    ball_positions[i][0] = field_rect.right - ball_radius
                    balls_speed_x[i] = -abs(balls_speed_x[i])
                    last_hit[i] = 0

            if ball_positions[i][1] <= field_rect.top + ball_radius:
                if balls_speed_y[i] < 0:
                    ball_positions[i][1] = field_rect.top + ball_radius
                    balls_speed_y[i] = abs(balls_speed_y[i])
                    last_hit[i] = 0
            elif ball_positions[i][1] >= field_rect.bottom - ball_radius:
                if balls_speed_y[i] > 0:
                    ball_positions[i][1] = field_rect.bottom - ball_radius
                    balls_speed_y[i] = -abs(balls_speed_y[i])
                    last_hit[i] = 0