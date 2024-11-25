import pyautogui
import cv2
import numpy as np

class Environnement:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def is_round(self, position):
        """
        Vérifie si un rond est présent autour de la position donnée.
        """
        try:
            region_size = 200
            screenshot = pyautogui.screenshot(region=(
                max(0, position[0] - region_size // 2),  # Empêcher les coordonnées négatives
                max(0, position[1] - region_size // 2),
                min(region_size, self.screen_width - position[0]),  # Limiter à l'écran
                min(region_size, self.screen_height - position[1])
            ))
            circle_position = pyautogui.locate("circle.png", screenshot)
            return circle_position is not None
        except Exception as e:
            print(f"Erreur de détection d'image : {e}")
            return False

    def is_near_round(self, position, threshold=100):
        """
        Vérifie si un rond est proche de la position donnée.
        """
        try:
            region_size = threshold * 2
            # Vérifie les limites de l'écran pour éviter une erreur de dimension
            region = (
                max(0, position[0] - region_size // 2),
                max(0, position[1] - region_size // 2),
                min(region_size, self.screen_width - position[0]),
                min(region_size, self.screen_height - position[1]),
            )
            screenshot = pyautogui.screenshot(region=region)
            circle_position = pyautogui.locate("circle.png", screenshot)
            return circle_position is not None
        except Exception as e:
            print(f"Erreur de détection de proximité : {e}")
            return False

    def detect_and_annotate_ronds(self):
        """
        Détecte tous les ronds sur l'écran et affiche une superposition.
        """
        try:
            # Capture de l'écran complet
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)  # Conversion en tableau numpy
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Conversion en BGR pour OpenCV

            # Détection des ronds (tous les correspondances avec circle.png)
            detected_positions = list(pyautogui.locateAll("circle.png", screenshot))

            # Dessiner des rectangles autour des positions détectées
            for pos in detected_positions:
                top_left = (pos.left, pos.top)
                bottom_right = (pos.left + pos.width, pos.top + pos.height)
                cv2.rectangle(screenshot_bgr, top_left, bottom_right, (0, 255, 0), 2)

            # Affichage du résultat
            cv2.imshow("Ronds détectés", screenshot_bgr)
            cv2.waitKey(0)  # Attendre une touche pour fermer la fenêtre
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Erreur lors de la détection des ronds : {e}")
