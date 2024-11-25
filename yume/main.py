from environ import Environnement
from agent import Agent
import time

def run():
    env = Environnement()
    agent = Agent()

    print("Démarrage de l'agent.")
    for step in range(1000):  # Nombre de cycles d'apprentissage
        # L'agent choisit une action (déplacement de la souris)
        state = agent.get_state()
        action = agent.choose_action(state)
        agent.perform_action(action)

        # Vérification après chaque clic
        if env.is_round(agent.get_state()):
            print(f"Bonus : Rond détecté à l'étape {step}.")
            agent.reward(1)  # Récompense positive
        else:
            print(f"Malus : Pas de rond détecté à l'étape {step}.")
            agent.reward(-0.5)  # Malus

        # Vérification de proximité pour encourager les bons mouvements
        if env.detect_all_ronds():
            print(f"Proximité détectée : Encouragement.")
            agent.reward(0.5)  # Encouragement

        # Temps d'attente pour observer les mouvements
        time.sleep(0.1)  # Ajustez pour accélérer ou ralentir

    print("Apprentissage terminé. Sauvegarde des données...")
    agent.save()

if __name__ == "__main__":
    run()
