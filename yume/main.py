from agent import Agent
from environ import Environnement
import time

def run():
    agent = Agent()
    env = Environnement()

    for episode in range(1000):  # Nombre d'épisodes d'apprentissage
        state = agent.get_state()  # Récupérer la position actuelle de la souris
        action = agent.choose_action(str(state))  # Choisir une action : "move" ou "click"

        if action == "move":
            agent.move_mouse()
            reward = -0.1  # Petite pénalité pour encourager les clics utiles
            print(f"Episode {episode}: Déplacement. Récompense: {reward}")
        elif action == "click":
            agent.click()
            if env.is_round(state):
                reward = 1  # Récompense pour avoir cliqué sur un rond
                print(f"Episode {episode}: Clic sur un rond ! Récompense: {reward}")
            else:
                reward = -1  # Malus pour un clic incorrect
                print(f"Episode {episode}: Clic incorrect. Malus: {reward}")

        next_state = agent.get_state()  # Nouvel état après l'action
        agent.learn(str(state), action, reward, str(next_state))  # Mise à jour de la Q-table

        time.sleep(0.01)  # Pause entre chaque action

    # Sauvegarde de la Q-table à la fin d'un épisode
    agent.save()
    print("Apprentissage sauvegardé.")

if __name__ == "__main__":
    run()
