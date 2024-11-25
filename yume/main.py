from agent import Agent
from environ import Environnement
import time

def run():
    agent = Agent()
    env = Environnement()

    for episode in range(1000):  # Nombre d'épisodes
        state = agent.get_state()
        action = agent.choose_action(str(state))

        if action == "move":
            agent.move_mouse()
            agent.no_click_moves += 1  # Incrémenter le compteur de déplacements
            reward = -0.1  # Petite pénalité pour déplacement

            # Vérifier si le déplacement est proche d'un rond
            if env.is_near_round(agent.get_state()):
                reward += 0.5  # Bonus pour être proche d'un rond
                print(f"Episode {episode}: Déplacement proche d'un rond ! Récompense: {reward}")
            else:
                print(f"Episode {episode}: Déplacement normal. Récompense: {reward}")

            # Appliquer un malus si l'agent ne clique pas après 10 déplacements
            if agent.no_click_moves > 10:
                reward -= 1
                print(f"Episode {episode}: Trop de déplacements sans clic. Malus: -1")

        elif action == "click":
            agent.click()
            agent.no_click_moves = 0  # Réinitialiser le compteur de déplacements
            if env.is_round(state):
                reward = 1  # Récompense pour clic sur un rond
                print(f"Episode {episode}: Clic sur un rond ! Récompense: {reward}")
            else:
                reward = -1  # Malus pour clic incorrect
                print(f"Episode {episode}: Clic incorrect. Malus: {reward}")

        next_state = agent.get_state()
        agent.learn(str(state), action, reward, str(next_state))

        time.sleep(0.05)  # Pause entre chaque action

    agent.save()
    print("Apprentissage sauvegardé.")

if __name__ == "__main__":
    run()
