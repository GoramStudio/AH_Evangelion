import random
import pickle
import numpy as np
from environ import Environnement
import pyautogui

class Agent:
    def __init__(self):
        self.environment = Environnement()
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1  # 10% du temps, l'agent va explorer de manière aléatoire
        self.q_table = {}  # Table de Q pour mémoriser les actions
        self.load()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(["move", "click"])
        else:
            if state not in self.q_table:
                self.q_table[state] = {"move": 0, "click": 0}
            action = max(self.q_table[state], key=self.q_table[state].get)
        return action

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {"move": 0, "click": 0}

        old_q_value = self.q_table[state][action]
        future_q_value = max(self.q_table.get(next_state, {"move": 0, "click": 0}).values())
        self.q_table[state][action] = old_q_value + self.learning_rate * (reward + self.discount_factor * future_q_value - old_q_value)

    def save(self):
        with open("agent.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self):
        try:
            with open("agent.pkl", "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("Aucune donnée d'apprentissage n'a été trouvée. Démarrage à zéro.")

    def move_mouse(self):
        new_x = random.randint(0, self.environment.screen_width)
        new_y = random.randint(0, self.environment.screen_height)
        pyautogui.moveTo(new_x, new_y, duration=0.1)

    def click(self):
        pyautogui.click()

