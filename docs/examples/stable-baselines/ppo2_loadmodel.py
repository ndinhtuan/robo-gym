import numpy as np
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def evaluate(model, env, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  print(obs)
  #obs_vect = np.zeros((1, len(obs)))
  #obs_vect[0,:] = obs
  _states = None

  for i in range(num_steps):
      print(i)
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs, _states)
      print("action: ", action)

      obs, reward, done, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward
  
if __name__=="__main__":

    target_machine_ip = 'localhost'
    env = gym.make('ObstacleAvoidanceMir100Sim-v0', ip=target_machine_ip, gui=True, gazebo_gui=True)
    env = ExceptionHandling(env)

    model = PPO2.load("tmp/best_model")
    evaluate(model, env)

