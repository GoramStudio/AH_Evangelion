import pyautogui
import random

class Agent:
    def __init__(self):
        self.q_table = {}  # Cette table contient les états et actions

    def get_state(self):
        """
        Retourne l'état actuel de l'agent (position actuelle de la souris)
        """
        return pyautogui.position()

    def choose_action(self, state):
        """
        Choisit une action parmi une liste d'actions possibles
        (par exemple, un déplacement aléatoire dans le voisinage de l'état actuel).
        """
        # Pour simplifier, l'agent choisit une direction aléatoire
        actions = ['up', 'down', 'left', 'right', 'click']
        return random.choice(actions)

    def perform_action(self, action):
        """
        Exécute l'action choisie par l'agent. Cela peut être un mouvement de souris ou un clic.
        """
        x, y = pyautogui.position()  # Obtenir la position actuelle de la souris

        if action == 'up':
            pyautogui.moveTo(x, y - 50)  # Déplacer la souris vers le haut
        elif action == 'down':
            pyautogui.moveTo(x, y + 50)  # Déplacer la souris vers le bas
        elif action == 'left':
            pyautogui.moveTo(x - 50, y)  # Déplacer la souris vers la gauche
        elif action == 'right':
            pyautogui.moveTo(x + 50, y)  # Déplacer la souris vers la droite
        elif action == 'click':
            pyautogui.click()  # Effectuer un clic

    def reward(self, value):
        """
        Applique une récompense (ou malus) à l'agent.
        """
        # Ici, vous pouvez mettre à jour la table Q ou tout autre système de récompense.
        print(f"Récompense ou malus de {value}")
        # Pour l'instant, nous n'ajoutons pas de logique Q-learning dans cette version.
