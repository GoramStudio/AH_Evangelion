import numpy as np
import tensorflow as tf
import random
import pickle
import pyautogui
import os
import pygame
from collections import deque
import time

# Constantes de l'environnement
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CIRCLE_RADIUS = 30
ACTION_SPACE = 5  # Haut, Bas, Gauche, Droite, Clic

# Fichiers de sauvegarde
MODEL_PATH = "dqn_model_mouse.h5"
MEMORY_PATH = "dqn_memory_mouse.pkl"

# Paramètres d'apprentissage
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000

# Création du modèle DQN
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=4, activation='relu'),  # x, y, cible_x, cible_y
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(ACTION_SPACE, activation='linear')  # Q-values pour chaque action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Environnement graphique avec PyGame
class SimulatedClickEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simulated Click Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Réinitialise la position de la souris et la cible."""
        self.mouse_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        self.target = [random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)]
        self.done = False
        self.reward = 0
        pyautogui.moveTo(self.mouse_pos[0], self.mouse_pos[1])  # Positionne la souris réelle

    def render(self):
        """Affiche l'environnement dans une fenêtre PyGame."""
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (0, 255, 0), self.target, CIRCLE_RADIUS)  # Cible
        pygame.draw.circle(self.screen, (255, 0, 0), self.mouse_pos, 5)  # Position de la souris
        pygame.display.flip()

    def step(self, action):
        """Met à jour l'état en fonction de l'action."""
        if action == 0:  # Haut
            self.mouse_pos[1] = max(0, self.mouse_pos[1] - 10)
        elif action == 1:  # Bas
            self.mouse_pos[1] = min(SCREEN_HEIGHT, self.mouse_pos[1] + 10)
        elif action == 2:  # Gauche
            self.mouse_pos[0] = max(0, self.mouse_pos[0] - 10)
        elif action == 3:  # Droite
            self.mouse_pos[0] = min(SCREEN_WIDTH, self.mouse_pos[0] + 10)
        elif action == 4:  # Clic
            distance = np.linalg.norm(np.array(self.mouse_pos) - np.array(self.target))
            if distance < CIRCLE_RADIUS:  # Clic réussi
                self.reward = 10
                self.target = [random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)]
            else:
                self.reward = -5
        else:
            self.reward = -1  # Mouvement inutile

        # Déplacer physiquement la souris (non bloquant avec 0 délai)
        pyautogui.moveTo(self.mouse_pos[0], self.mouse_pos[1])

        # Retourner l'état et le feedback
        return self.mouse_pos + self.target, self.reward, self.done

    def handle_events(self):
        """Gère les événements PyGame pour éviter le freeze."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.model = create_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.load()

    def save(self):
        """Sauvegarde le modèle et la mémoire."""
        self.model.save(MODEL_PATH)
        with open(MEMORY_PATH, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self):
        """Charge le modèle et la mémoire si disponible."""
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, 'rb') as f:
                self.memory = pickle.load(f)

    def act(self, state):
        """Choisit une action en fonction de l'état."""
        if np.random.rand() <= self.epsilon:
            return random.randint(0, ACTION_SPACE - 1)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Apprend à partir d'un batch d'expériences."""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += GAMMA * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Main function
def main():
    env = SimulatedClickEnvironment()
    agent = DQNAgent()

    for episode in range(500):
        state = np.array(env.mouse_pos + env.target)
        env.reset()

        for t in range(200):  # Limite de pas par épisode
            env.handle_events()
            env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        agent.replay()
        print(f"Episode {episode + 1} completed.")
        if episode % 10 == 0:
            agent.save()

    pygame.quit()

if __name__ == "__main__":
    main()
