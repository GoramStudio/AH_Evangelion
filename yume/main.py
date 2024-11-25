import time
from agent import Agent

def run():
    agent = Agent()

    for episode in range(1000):
        state = agent.environment.get_state()
        action = agent.choose_action(str(state))

        if action == "move":
            agent.move_mouse()
            next_state = agent.environment.get_state()
            reward = -0.1  # Pénalité pour un déplacement
        elif action == "click":
            agent.click()
            next_state = agent.environment.get_state()
            if agent.environment.is_round(state):
                reward = 1  # Récompense pour avoir cliqué sur un rond
            else:
                reward = -1  # Malus pour avoir cliqué ailleurs

        agent.learn(str(state), action, reward, str(next_state))

        time.sleep(0.5)  # Pause pour éviter que le programme tourne trop vite

        if episode % 100 == 0:
            print(f"Episode {episode}, Récompense cumulative : {reward}")
    
    agent.save()

if __name__ == "__main__":
    run()
