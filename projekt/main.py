import pygame
import sys
import random
import math
import time

from torch import tensor

from AIcollider import AIcolider
from AImover import AImover
import torch

from pygame.math import Vector2
from ElasticCollision.ec_game import momentum_trigonometry


def angle_with_x_axis(vector):
    angle = math.atan2(vector[1], vector[0])
    return angle
aicollider = AIcolider()
aicollider.load_state_dict(torch.load("model_best"))
aimover = AImover()
aimover.load_state_dict(torch.load("model2_best"))
aicollider.train(False)
aimover.train(False)

aiMOVE = False

# Inicjalizacja Pygame
pygame.init()

# Ustawienia okna
screen_width = 600
window_height = 400
window = pygame.display.set_mode((2*screen_width, window_height+75))
pygame.display.set_caption("Bile")
balls_number = 3

# Ustawienia kuli
ball_radius = 20
ball_color = (255, 0, 0)

ball_positions1 = [[0,0] for _ in range(balls_number)]
for i in range(balls_number):
    r = 5*ball_radius
    m = r*i
    a = 0
    while m > screen_width - 100:
        m =  m - (screen_width - r)
        a +=1
    ball_positions1[i] = [r + m, window_height // 2 + a*r]

ball_positions2 = [[0,0] for _ in range(balls_number)]
for i in range(balls_number):
    r = 5 * ball_radius
    m = r*i
    a = 0
    while m > 2*screen_width - 100:
        m =  m - (2*screen_width - r)
        a +=1
    ball_positions2[i] = [screen_width + r + m, window_height // 2 + a*50]

ball_speed_magnitude = 1.0 # Magnituda prędkości kulki

balls1_speed_x = [0] * balls_number
balls1_speed_y = [0] * balls_number

balls2_speed_x = [0] * balls_number
balls2_speed_y = [0] * balls_number

# Ustawienia ramki
frame_thickness = 10
frame_color = (0, 0, 0)
frame_rect = pygame.Rect(
    frame_thickness, frame_thickness, screen_width - 2 * frame_thickness, window_height - 2 * frame_thickness
)
frame_rect2 = pygame.Rect(
    frame_thickness + screen_width, frame_thickness, screen_width - 2 * frame_thickness, window_height - 2 * frame_thickness
)

# Ustawienia pola do odbijania
field_color = (0, 255, 0)
field_rect1 = pygame.Rect(
    2 * frame_thickness,
    2 * frame_thickness,
    screen_width - 4 * frame_thickness,
    window_height - 4 * frame_thickness,
)
field_rect2 = pygame.Rect(
    2 * frame_thickness + screen_width,
    2 * frame_thickness,
    screen_width - 4 * frame_thickness,
    window_height - 4 * frame_thickness,
)
last_hit1 = [0] * balls_number
last_hit2 = [0] * balls_number
# Główna pętla programu
petla = True
counter = 0
f = open("data.csv", "a")
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
font = pygame.font.Font('freesansbold.ttf', 32)
text = font.render('', True, green, blue)
przebiegi = 0
while True:
    przebiegi +=1
    start_time = time.time()

    # Wyczyszczenie ekranu
    window.fill((255, 255, 255))

    # Narysowanie ramki
    pygame.draw.rect(window, frame_color, frame_rect)
    pygame.draw.rect(window, frame_color, frame_rect2)

    # Narysowanie pola do odbijania
    pygame.draw.rect(window, field_color, field_rect1)
    pygame.draw.rect(window, field_color, field_rect2)

    # Narysowanie kul
    for nr in range(balls_number):
        pos = ball_positions1[nr]
        if nr == 0:
            pygame.draw.circle(window, (255,255,255), (int(pos[0]), int(pos[1])), ball_radius)
        else:
            pygame.draw.circle(window, ball_color, (int(pos[0]), int(pos[1])), ball_radius)
        pos = ball_positions2[nr]
        if nr == 0:
            pygame.draw.circle(window, (255, 255, 255), (int(pos[0]), int(pos[1])), ball_radius)
        else:
            pygame.draw.circle(window, ball_color, (int(pos[0]), int(pos[1])), ball_radius)

    textRect = text.get_rect()
    textRect.center = (screen_width, window_height + 50)
    window.blit(text, textRect)
    info1 = font.render("model oparty na sieci neuronowej", True, white, black)
    info2 = font.render("model oparty na fizyce", True, white, black)
    i1 = info1.get_rect()
    i2 = info2.get_rect()
    i1.center = (screen_width//2, window_height + 10)
    i2.center = (screen_width // 2 + screen_width, window_height + 10)
    window.blit(info1, i1)
    window.blit(info2, i2)

    # Aktualizacja ekranu
    pygame.display.flip()



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Wygenerowanie losowego kierunku prędkości kulki po naciśnięciu przycisku myszy

            ball_speed_directions = random.uniform(0, 2 * math.pi)
            balls1_speed_x[0] = ball_speed_magnitude * math.cos(ball_speed_directions)
            balls1_speed_y[0] = ball_speed_magnitude * math.sin(ball_speed_directions)
            balls2_speed_x[0] = ball_speed_magnitude * math.cos(ball_speed_directions)
            balls2_speed_y[0] = ball_speed_magnitude * math.sin(ball_speed_directions)

    prev_time = [time.time()]*balls_number
    keys = pygame.key.get_pressed()  # Checking pressed keys
    if keys[pygame.K_SPACE]:
        for i in range(len(ball_positions1)):
            if balls1_speed_x[i] **2 + balls1_speed_y[i]**2 != 0:
                # Poruszanie kulki
                speed = math.sqrt(balls1_speed_x[i] ** 2 + balls1_speed_y[i] ** 2)
                speed2 = aimover(torch.FloatTensor([speed])).tolist()[0]

                slowdown = speed2 / speed
                balls1_speed_x[i] = balls1_speed_x[i] * slowdown

                balls1_speed_y[i] = balls1_speed_y[i] * slowdown

                ball_positions1[i][0] += balls1_speed_x[i]
                ball_positions1[i][1] += balls1_speed_y[i]



                    # Sprawdzanie zderzenia między kulami
                for j in range(len(ball_positions1)):
                    if i != j:
                        distance = math.sqrt((ball_positions1[i][0] - ball_positions1[j][0]) ** 2 +
                                             (ball_positions1[i][1] - ball_positions1[j][1]) ** 2)
                        if distance <= 2 * ball_radius:
                            if last_hit1[i] != j + 1:
                                last_hit1[i] = j + 1
                                last_hit1[j] = i + 1
                                # Obliczanie wektora między kulami
                                collision_vector = [ball_positions1[i][0] - ball_positions1[j][0],
                                                    ball_positions1[i][1] - ball_positions1[j][1]]
                                collision_distance = math.sqrt(collision_vector[0] ** 2 + collision_vector[1] ** 2)

                                # Normalizowanie wektora kolizji
                                normalized_collision_vector = [collision_vector[0] / collision_distance,
                                                               collision_vector[1] / collision_distance]

                                if distance < 2 * ball_radius:
                                    c = 2 * ball_radius - abs(distance)
                                    przes_y = (c * abs(ball_positions1[i][1] - ball_positions1[j][1]))
                                    c1 = (c ** 2)
                                    c2 = (przes_y ** 2)
                                    przes_x = math.sqrt(abs(c1 - c2))
                                    if ball_positions1[i][1] < ball_positions1[j][1]:
                                        ball_positions1[i][1] -= przes_y
                                    elif ball_positions1[i][1] > ball_positions1[j][1]:
                                        ball_positions1[i][1] += przes_y
                                    if ball_positions1[i][0] < ball_positions1[j][0]:
                                        ball_positions1[i][0] -= przes_x
                                    elif ball_positions1[i][0] > ball_positions1[j][0]:
                                        ball_positions1[i][0] += przes_x

                                a = aicollider(torch.FloatTensor([normalized_collision_vector[0],normalized_collision_vector[1],balls1_speed_x[i],balls1_speed_y[i],balls1_speed_x[j],balls1_speed_y[j]])).tolist()


                                balls1_speed_x[i] = a[0]
                                balls1_speed_y[i] = a[1]
                                balls1_speed_x[j] = a[2]
                                balls1_speed_y[j] = a[3]

    # Odbijanie kulki od ścianek
                if ball_positions1[i][0] <= field_rect1.left + ball_radius:
                    if balls1_speed_x[i] < 0:
                        ball_positions1[i][0] = field_rect1.left + ball_radius
                        last_hit1[i] = 0
                        balls1_speed_x[i] = abs(balls1_speed_x[i])
                elif ball_positions1[i][0] >= field_rect1.right - ball_radius:
                    if balls1_speed_x[i] > 0:
                        ball_positions1[i][0] = field_rect1.right - ball_radius
                        balls1_speed_x[i] = -abs(balls1_speed_x[i])
                        last_hit1[i] = 0

                if ball_positions1[i][1] <= field_rect1.top + ball_radius:
                    if balls1_speed_y[i] < 0:
                        ball_positions1[i][1] = field_rect1.top + ball_radius
                        balls1_speed_y[i] = abs(balls1_speed_y[i])
                        last_hit1[i] = 0
                elif ball_positions1[i][1] >= field_rect1.bottom - ball_radius:
                    if balls1_speed_y[i] > 0:
                        ball_positions1[i][1] = field_rect1.bottom - ball_radius
                        balls1_speed_y[i] = -abs(balls1_speed_y[i])
                        last_hit1[i] = 0

        for i in range(len(ball_positions2)):
                if balls2_speed_x[i] **2 + balls2_speed_y[i] **2 != 0:
                    # Poruszanie kulki
                    speed = math.sqrt(balls2_speed_x[i] ** 2 + balls2_speed_y[i] ** 2)
                    speed2 = speed - 0.0005
                    if speed2 <= 0.0003 and speed2 >= -0.0003:
                        speed = 0
                        balls2_speed_x[i] = 0
                        balls2_speed_y[i] = 0
                    else:
                        slowdown = speed2 / speed
                        balls2_speed_x[i] = balls2_speed_x[i] * slowdown

                        balls2_speed_y[i] = balls2_speed_y[i] * slowdown

                    ball_positions2[i][0] += balls2_speed_x[i]
                    ball_positions2[i][1] += balls2_speed_y[i]

                    # Sprawdzanie zderzenia między kulami
                    for j in range(len(ball_positions2)):
                        if i != j:
                            distance = math.sqrt((ball_positions2[i][0] - ball_positions2[j][0]) ** 2 +
                                                 (ball_positions2[i][1] - ball_positions2[j][1]) ** 2)
                            if distance <= 2 * ball_radius:
                                if last_hit2[i] != j + 1:
                                    last_hit2[i] = j + 1
                                    last_hit2[j] = i + 1
                                    # Obliczanie wektora między kulami
                                    collision_vector = [ball_positions2[i][0] - ball_positions2[j][0],
                                                        ball_positions2[i][1] - ball_positions2[j][1]]
                                    collision_distance = math.sqrt(collision_vector[0] ** 2 + collision_vector[1] ** 2)

                                    # Normalizowanie wektora kolizji
                                    normalized_collision_vector = [collision_vector[0] / collision_distance,
                                                                   collision_vector[1] / collision_distance]

                                    if distance < 2 * ball_radius:
                                        c = 2 * ball_radius - abs(distance)
                                        przes_y = (c * abs(ball_positions2[i][1] - ball_positions2[j][1]))
                                        c1 = (c ** 2)
                                        c2 = (przes_y ** 2)
                                        przes_x = math.sqrt(abs(c1 - c2))
                                        if ball_positions2[i][1] < ball_positions2[j][1]:
                                            ball_positions2[i][1] -= przes_y
                                        elif ball_positions2[i][1] > ball_positions2[j][1]:
                                            ball_positions2[i][1] += przes_y
                                        if ball_positions2[i][0] < ball_positions2[j][0]:
                                            ball_positions2[i][0] -= przes_x
                                        elif ball_positions2[i][0] > ball_positions2[j][0]:
                                            ball_positions2[i][0] += przes_x

                                    vector1 = Vector2(balls2_speed_x[i], balls2_speed_y[i])
                                    centre1 = Vector2(ball_positions2[i][0], ball_positions2[i][1])
                                    vector2 = Vector2(balls2_speed_x[j], balls2_speed_y[j])
                                    centre2 = Vector2(ball_positions2[j][0], ball_positions2[j][1])
                                    mass1 = 1.0
                                    mass2 = 1.0

                                    v11, v12 = momentum_trigonometry(centre1, centre2, vector1, vector2, mass1, mass2,
                                                                     False)

                                    balls2_speed_x[i] = v11[0]
                                    balls2_speed_y[i] = v11[1]
                                    balls2_speed_x[j] = v12[0]
                                    balls2_speed_y[j] = v12[1]

                    # Odbijanie kulki od ścianek
                    if ball_positions2[i][0] <= field_rect2.left + ball_radius:
                        if balls2_speed_x[i] < 0:
                            ball_positions2[i][0] = field_rect2.left + ball_radius
                            last_hit2[i] = 0
                            balls2_speed_x[i] = abs(balls2_speed_x[i])
                    elif ball_positions2[i][0] >= field_rect2.right - ball_radius:
                        if balls2_speed_x[i] > 0:
                            ball_positions2[i][0] = field_rect2.right - ball_radius
                            balls2_speed_x[i] = -abs(balls2_speed_x[i])
                            last_hit2[i] = 0

                    if ball_positions2[i][1] <= field_rect2.top + ball_radius:
                        if balls2_speed_y[i] < 0:
                            ball_positions2[i][1] = field_rect2.top + ball_radius
                            balls2_speed_y[i] = abs(balls2_speed_y[i])
                            last_hit2[i] = 0
                    elif ball_positions2[i][1] >= field_rect2.bottom - ball_radius:
                        if balls2_speed_y[i] > 0:
                            ball_positions2[i][1] = field_rect2.bottom - ball_radius
                            balls2_speed_y[i] = -abs(balls2_speed_y[i])
                            last_hit2[i] = 0
    if przebiegi >=100:
        przebiegi = 0
        suma = 0
        for b in range(len(ball_positions1)):
            x_sc = ball_positions1[b][0] - (ball_positions2[b][0] - screen_width)
            y_sc = ball_positions1[b][1] - ball_positions2[b][1]
            odl = math.sqrt(x_sc ** 2 + y_sc ** 2)
            suma += odl

        suma = round(suma, 4)
        t = ("Sumaryczna różnica odległości bil pomiędzy modelami: " + str(suma))
        text = font.render(t, False, green, blue)

