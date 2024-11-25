import cv2
import numpy as np
import pyautogui

class Environnement:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def is_round(self, position):
        """
        Vérifie si un rond est présent autour de la position donnée.
        """
        try:
            # Charger le modèle du rond
            template = cv2.imread("circle.png", cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError("L'image 'circle.png' est introuvable ou invalide.")

            w, h = template.shape[::-1]
            region_size = 100

            # Définir une région autour de la position
            region = (
                max(0, position[0] - region_size // 2),
                max(0, position[1] - region_size // 2),
                min(region_size, self.screen_width - position[0]),
                min(region_size, self.screen_height - position[1]),
            )
            screenshot = pyautogui.screenshot(region=region)
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

            # Correspondance avec le modèle
            res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.8)  # Correspondances >= 80%
            loc_list = list(zip(*loc[::-1]))  # Convertir en liste avant d'utiliser len()

            return len(loc_list) > 0

        except Exception as e:
            print(f"Erreur de détection d'image : {e}")
            return False

    def detect_all_ronds(self):
        """
        Détecte tous les ronds sur l'écran et retourne leurs positions.
        """
        try:
            # Charger le modèle du rond
            template = cv2.imread("circle.png", cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError("L'image 'circle.png' est introuvable ou invalide.")

            w, h = template.shape[::-1]

            # Capture de l'écran
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

            # Recherche des correspondances
            res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            # Convertir en liste avant de retourner
            detected_positions = list(zip(*loc[::-1]))
            return detected_positions

        except Exception as e:
            print(f"Erreur lors de la détection des ronds : {e}")
            return []
