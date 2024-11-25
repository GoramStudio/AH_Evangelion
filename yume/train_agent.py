from stable_baselines3 import PPO
import gym
from env import MouseClickEnv

def train_model():
    # Initialiser l'environnement
    env = MouseClickEnv()

    # Créer ou charger le modèle PPO
    model_path = 'models/ppo_mouse_click_model'
    try:
        model = PPO.load(model_path)  # Charger le modèle si il existe déjà
        print("Modèle chargé !")
    except:
        model = PPO('CnnPolicy', env, verbose=1)  # Sinon, créer un nouveau modèle
        print("Nouveau modèle créé !")

    # Entraîner le modèle pendant 10000 étapes
    model.learn(total_timesteps=10000)

    # Sauvegarder le modèle après l'entraînement
    model.save(model_path)
    print("Modèle sauvegardé !")

def evaluate_model():
    # Charger le modèle sauvegardé
    model = PPO.load('models/ppo_mouse_click_model')

    # Initialiser l'environnement
    env = MouseClickEnv()

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Récompense totale après évaluation : {total_reward}")

if __name__ == "__main__":
    # Entraîner le modèle
    train_model()

    # Évaluer le modèle
    evaluate_model()
