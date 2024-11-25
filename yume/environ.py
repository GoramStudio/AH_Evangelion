import pyautogui
import random
import cv2
import numpy as np

class Environnement:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_position = (self.screen_width // 2, self.screen_height // 2)
        self.capture = None

    def get_state(self):
        # Retourne la position actuelle de la souris
        return pyautogui.position()

    def is_round(self, position):
        # Détection d'un rond (ici, on suppose qu'un rond peut être détecté avec OpenCV)
        screenshot = pyautogui.screenshot(region=(position[0] - 20, position[1] - 20, 40, 40))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        # Application d'un filtre pour détecter des cercles
        circles = cv2.HoughCircles(screenshot, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=20)

        if circles is not None:
            return True
        return False

    def reset(self):
        # Réinitialise la position du curseur à une position centrale
        pyautogui.moveTo(self.screen_width // 2, self.screen_height // 2, duration=0.1)
