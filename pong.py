import pygame
import sys
import random

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
width, height = 600, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong")

# Colores
WHITE = (255, 255, 255)

# Jugador
player_width, player_height = 15, 60
player_x, player_y = 10, height // 2 - player_height // 2
player_speed = 5

# Computadora
computer_width, computer_height = 15, 60
computer_x, computer_y = width - 30, height // 2 - computer_height // 2
computer_speed = 3

# Pelota
ball_size = 15
ball_x, ball_y = width // 2, height // 2
ball_speed_x = random.choice([-5, 5])
ball_speed_y = random.choice([-5, 5])

# Marcador
player_score = 0
computer_score = 0
font = pygame.font.Font(None, 36)

# Bucle principal
clock = pygame.time.Clock()

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      sys.exit()


  # Movimiento del jugador
  keys = pygame.key.get_pressed()
  if keys[pygame.K_UP] and player_y > 0:
    player_y -= player_speed
  if keys[pygame.K_DOWN] and player_y < height - player_height:
    player_y += player_speed

  # Movimiento de la computadora
  if computer_y + computer_height // 2 < ball_y:
    computer_y += computer_speed
  elif computer_y + computer_height // 2 > ball_y:
    computer_y -= computer_speed

  # Movimiento de la pelota
  ball_x += ball_speed_x
  ball_y += ball_speed_y

  # Rebotes en las paredes
  if ball_y <= 0 or ball_y >= height - ball_size:
    ball_speed_y = -ball_speed_y

  # Rebotes en los jugadores
  if((player_x <= ball_x <= player_x + player_width and
      player_y <= ball_y <= player_y + player_height) or
     (computer_x - ball_size <= ball_x <= computer_x and
      computer_y <= ball_y <= computer_y + computer_height)):
      ball_speed_x = -ball_speed_x

  # Punto para la computadora
  if ball_x <= 0:
    ball_x, ball_y = width // 2, height // 2
    ball_speed_x = random.choice([-5, 5])
    ball_speed_y = random.choice([-5, 5])
    computer_score += 1

  # Punto para el jugador
  if ball_x >= width - ball_size:
    ball_x, ball_y = width // 2, height // 2
    ball_speed_x = random.choice([-5, 5])
    ball_speed_y = random.choice([-5, 5])
    player_score += 1

  # Dibujar en la pantalla
  screen.fill((0, 0, 0))
  pygame.draw.rect(screen, WHITE, (player_x, player_y, player_width, player_height))
  pygame.draw.rect(screen, WHITE, (computer_x - computer_width, computer_y, computer_width, computer_height))
  pygame.draw.ellipse(screen, WHITE, (ball_x, ball_y, ball_size, ball_size))


  # Mostrar el marcador
  player_text = font.render(str(player_score), True, WHITE)
  computer_text = font.render(str(computer_score), True, WHITE)
  screen.blit(player_text, (width // 4, 20))
  screen.blit(computer_text, (3 * width // 4 - player_text.get_width(), 20))

  # Actualizar la pantalla
  pygame.display.flip()

  # Controlar la velocidad del bucle
  clock.tick(60)
