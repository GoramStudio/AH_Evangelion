import pyautogui
import random
import pickle

class Agent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-table pour stocker les valeurs Q
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'actualisation
        self.epsilon = epsilon  # Taux d'exploration
        self.load()  # Charger la Q-table sauvegardée

    def get_state(self):
        # Obtenir la position actuelle de la souris
        x, y = pyautogui.position()
        return (x, y)

    def choose_action(self, state):
        # Stratégie epsilon-greedy
        if state not in self.q_table:
            self.q_table[state] = {"move": 0, "click": 0}  # Initialiser les actions pour cet état

        if random.random() < self.epsilon:  # Exploration
            return random.choice(["move", "click"])
        else:  # Exploitation
            return max(self.q_table[state], key=self.q_table[state].get)

    def move_mouse(self):
        # Déplacer la souris de manière aléatoire
        screen_width, screen_height = pyautogui.size()
        new_x = random.randint(0, screen_width - 1)
        new_y = random.randint(0, screen_height - 1)
        pyautogui.moveTo(new_x, new_y, duration=0.1)  # Déplacement rapide mais fluide

    def click(self):
        # Clic gauche
        pyautogui.click()

    def learn(self, state, action, reward, next_state):
        # Initialiser l'état et le prochain état s'ils n'existent pas
        if state not in self.q_table:
            self.q_table[state] = {"move": 0, "click": 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {"move": 0, "click": 0}

        # Mise à jour de la Q-table
        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state][action])

    def save(self):
        # Sauvegarder la Q-table
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self):
        # Charger la Q-table si elle existe
        try:
            with open("q_table.pkl", "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}
