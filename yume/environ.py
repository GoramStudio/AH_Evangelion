import pyautogui

class Environnement:
    def __init__(self):
        # Récupérer les dimensions de l'écran
        self.screen_width, self.screen_height = pyautogui.size()

    def is_round(self, position):
        """
        Vérifie si un rond est présent autour de la position donnée.
        :param position: tuple (x, y) - position actuelle de la souris
        :return: bool - True si un rond est détecté, False sinon
        """
        try:
            # Taille de la région capturée (200x200 autour de la position de la souris)
            region_size = 200
            screenshot = pyautogui.screenshot(region=(position[0] - region_size // 2, 
                                                      position[1] - region_size // 2, 
                                                      region_size, region_size))
            screenshot.save("debug_screenshot.png")  # Debug : sauvegarder pour vérification
            # Chercher une correspondance avec l'image de référence
            circle_position = pyautogui.locate("circle.png", screenshot)
            return circle_position is not None
        except Exception as e:
            print(f"Erreur de détection d'image : {e}")
            return False
