import pygame
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygame.locals import *


# Deep Q-Network Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Game Constants
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15
GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100
PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
SENSOR_ANGLES = [45, 30, 15, 0, -15, -30, -45]

# Initialize PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(len(SENSOR_ANGLES) + 1, 2).to(device)  # Input: 7 sensors + velocity
model.load_state_dict(torch.load('dqn_model.pth', map_location=device))  # Load trained model
model.eval()


class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def calculate_sensor_distances(self, pipes):
        sensor_distances = {}
        bird_center = self.rect.center
        max_distance = max(SCREEN_WIDHT, SCREEN_HEIGHT)

        for angle in SENSOR_ANGLES:
            radians = math.radians(angle)
            dx = math.cos(radians)
            dy = -math.sin(radians)
            distance = 0
            hit_found = False

            while not hit_found and distance < max_distance:
                distance += 1
                x = int(bird_center[0] + dx * distance)
                y = int(bird_center[1] + dy * distance)

                if x < 0 or x >= SCREEN_WIDHT or y < 0 or y >= SCREEN_HEIGHT:
                    break

                for pipe in pipes:
                    if pipe.rect.collidepoint(x, y):
                        hit_found = True
                        break

            sensor_distances[angle] = distance / max_distance  # Normalize
        return sensor_distances


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    """Generate pipes with a minimum height for the lower pipe"""
    min_lower_pipe_height = 200  # Minimum height for the lower pipe
    max_lower_pipe_height = SCREEN_HEIGHT - PIPE_GAP #- min_lower_pipe_height
    lower_pipe_height = random.randint(min_lower_pipe_height, max_lower_pipe_height)

    pipe = Pipe(False, xpos, lower_pipe_height)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - lower_pipe_height - PIPE_GAP)
    return pipe, pipe_inverted


def draw_sensors(screen, bird, sensor_distances):
    bird_center = bird.rect.center
    for angle, distance in sensor_distances.items():
        real_distance = distance * max(SCREEN_WIDHT, SCREEN_HEIGHT)
        radians = math.radians(angle)
        end_x = int(bird_center[0] + math.cos(radians) * real_distance)
        end_y = int(bird_center[1] - math.sin(radians) * real_distance)
        pygame.draw.line(screen, (255, 0, 0), bird_center, (end_x, end_y), 1)

    font = pygame.font.SysFont(None, 24)
    x_offset = SCREEN_WIDHT - 120
    y_offset = 10
    for angle, distance in sensor_distances.items():
        text = font.render(f"{angle}°: {int(distance * max(SCREEN_WIDHT, SCREEN_HEIGHT))}", True, (255, 255, 255))
        screen.blit(text, (x_offset, y_offset))
        y_offset += 20


def reset_game():
    """Reset all game objects and groups"""
    # Reset bird
    bird_group.empty()
    new_bird = Bird()
    bird_group.add(new_bird)

    # Reset ground
    ground_group.empty()
    for i in range(2):
        ground = Ground(GROUND_WIDHT * i)
        ground_group.add(ground)

    # Reset pipes
    pipe_group.empty()
    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    return new_bird


# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird AI')

# Load assets
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

# Initialize game objects
bird_group = pygame.sprite.Group()
ground_group = pygame.sprite.Group()
pipe_group = pygame.sprite.Group()

# Main game loop
while True:
    # Initialize game state
    bird = reset_game()
    clock = pygame.time.Clock()
    running = True

    # Game starts immediately
    while running:
        clock.tick(30)

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        # AI Gameplay
        # Get game state
        sensor_distances = bird.calculate_sensor_distances(pipe_group.sprites())
        velocity_normalized = bird.speed / SPEED

        # Create state vector
        state = np.array([sensor_distances[angle] for angle in SENSOR_ANGLES] + [velocity_normalized], dtype=np.float32)

        # AI decision
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        # Execute action
        if action == 1:
            bird.bump()

        # Update game state
        bird.update()
        ground_group.update()
        pipe_group.update()

        # Check collisions
        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            time.sleep(1)
            running = False  # Will trigger game restart

        # Drawing
        screen.blit(BACKGROUND, (0, 0))
        bird_group.draw(screen)
        pipe_group.draw(screen)
        draw_sensors(screen, bird, sensor_distances)
        ground_group.draw(screen)
        pygame.display.update()

        # Generate new pipes
        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDHT * 2)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        # Update ground
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)

pygame.quit()