import pygame
import sys
import random
import math
import time



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
balls_number = 1

# Ustawienia kuli
ball_radius = 20
ball_color = (255, 0, 0)

ball_positions = [[0,0] for _ in range(balls_number)]
for i in range(balls_number):
    m = 50*i
    a = 0
    while m > window_width - 100:
        m =  m - (window_width - 50)
        a +=1
    ball_positions[i] = [50 + m, window_height // 2 + a*50]

ball_speed_magnitude = 1.0 # Magnituda prędkości kulki

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
f = open("dane_spowolnienie.csv", "a")
while counter<1000000:
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

            print(counter)
            counter += 1


            speed = math.sqrt(balls_speed_x[i] **2 + balls_speed_y[i]**2)
            speed2 = speed - 0.0005

            if speed2 <= 0.0003 and speed2 >= -0.0003:
                speed2 = 0
                balls_speed_x[i] = 0
                balls_speed_y[i] = 0
            else:
                slowdown = speed2/speed
                balls_speed_x[i] = balls_speed_x[i]*slowdown
                balls_speed_y[i] = balls_speed_y[i] * slowdown


            ball_positions[i][0] += balls_speed_x[i]
            ball_positions[i][1] += balls_speed_y[i]

            f.write(str(speed) + "," + str(speed2) + "\n")



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
        else:
            ball_speed_directions = random.uniform(0, 2 * math.pi)
            balls_speed_x[0] = ball_speed_magnitude * math.cos(ball_speed_directions)
            balls_speed_y[0] = ball_speed_magnitude * math.sin(ball_speed_directions)