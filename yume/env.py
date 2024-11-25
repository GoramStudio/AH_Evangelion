import gym
from gym import spaces
import numpy as np
import pyautogui
import cv2
import sys

class MouseClickEnv(gym.Env):
    def __init__(self):
        super(MouseClickEnv, self).__init__()
        
        # L'espace d'action contient 5 actions : 4 directions de déplacement et 1 clic
        self.action_space = spaces.Discrete(5)  # Déplacer (haut, bas, gauche, droite), clic
        
        # L'espace d'observation est l'image de l'écran
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Variables de récompenses
        self.reward_positive = 1
        self.reward_negative = -1
        self.reward_no_action = 0

    def reset(self):
        # Capturer l'écran et redimensionner l'image à (64, 64)
        self.state = np.array(pyautogui.screenshot(region=(0, 0, self.screen_width, self.screen_height)))
        self.state = cv2.cvtColor(self.state, cv2.COLOR_RGB2BGR)
        
        # Redimensionner l'image à 64x64 pour qu'elle corresponde à l'espace d'observation
        self.state = cv2.resize(self.state, (64, 64))
        
        return self.state

    def step(self, action):
        # Appliquer l'action choisie (déplacement ou clic)
        if action == 0:  # Déplacer vers la gauche
            pyautogui.move(-10, 0)
        elif action == 1:  # Déplacer vers la droite
            pyautogui.move(10, 0)
        elif action == 2:  # Déplacer vers le haut
            pyautogui.move(0, -10)
        elif action == 3:  # Déplacer vers le bas
            pyautogui.move(0, 10)
        elif action == 4:  # Cliquer
            pyautogui.click()

        # Capturer l'écran après l'action et redimensionner
        self.state = np.array(pyautogui.screenshot(region=(0, 0, self.screen_width, self.screen_height)))
        self.state = cv2.cvtColor(self.state, cv2.COLOR_RGB2BGR)
        
        # Redimensionner l'image à 64x64 pour qu'elle corresponde à l'espace d'observation
        self.state = cv2.resize(self.state, (64, 64))

        # Détection des cercles dans l'image
        gray = cv2.cvtColor(self.state, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50
        )

        reward = self.reward_no_action
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Vérifier si l'agent a cliqué sur un cercle
                cursor_x, cursor_y = pyautogui.position()
                if x - r < cursor_x < x + r and y - r < cursor_y < y + r:
                    reward = self.reward_positive  # Récompense pour avoir cliqué sur un cercle
                    print("Bonus : Vous avez cliqué sur un cercle !")
                    break
                else:
                    reward = self.reward_negative  # Malus si l'agent a cliqué hors du cercle
                    print("Malus : Pas de cercle, action incorrecte.")

        done = False  # Pas de condition de fin spécifique ici
        return self.state, reward, done, {}
