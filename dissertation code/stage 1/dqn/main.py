import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import gym
import numpy as np
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback


env = gym.make('DebrisAvoidance-v0')
# env = Monitor(env, "./") <-- path of the log file


# define a model
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='./ofc_stage1-tensorboard')

# learn a model
# model.learn(total_timesteps=int(2e6), callback=callback)

# load a model
model = DQN.load("model")

# saves a model 
# model.save("modeldqn")

for j in range(15):
	obs = env.reset()
	dones = False
	while not dones:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()
    



