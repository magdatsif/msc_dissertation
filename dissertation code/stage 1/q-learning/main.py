import gym
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 

env = gym.make('DebrisAvoidance-v0')

SHOW = 150
load_q_table = 0
window_height = 360

# loads or creates q table
def create_Q_table():
	q_table = None
	# if q_table is None:
	# 	q_table = {}
	# 	for i in range(0, 5):
	# 		for j in range(0, 5):
	# 			for l in range(0, 5):
	# 				q_table[(i, j, l)] = [np.random.uniform(-5, 0) for z in range(3)]

	load_q_table = 1
	with open('pkl.pickle', "rb") as f:
	    q_table = pickle.load(f)
	return q_table

def get_tuple(state):
    return tuple(state) 

def test_agent(q_table):
	state = get_tuple(env.reset())
	for i in range(1000):
		action = np.argmax(q_table[state])	
		observation, reward, done, info = env.step(action)
		observation = get_tuple(observation)
		env.render()

		if done: 
			break
		
		state = observation

# train the agent
def train(EPISODES, LEARNING_RATE, EPSILON, EPSILON_DECAY, GAMMA, q_table):
	episode_rewards = []
	episode_rewards_stats = {'ep': [], 'avg': [], 'max': [], 'min': [], 'rewards': [], 'timesteps':[]}
	timesteps = []
	episodes = []
	epsilons = []

	for episode in range(EPISODES):
		epsilons.append(EPSILON)
		state = get_tuple(env.reset())
		done = False
		
		print(f"Running episode {episode}")
		print(f"EPSILON {EPSILON}")

		episode_reward = 0
		for i in range(1000):

			# for training
			if np.random.random() > EPSILON:
				action = np.argmax(q_table[state])
			else:
				action = np.random.randint(0, 3)

			#for testing q_table
			observation, reward, done, info = env.step(action)
			observation = get_tuple(observation)
			env.render()

			new_action = np.argmax(q_table[observation])
			episode_reward += reward

			# update q values
			if not done:
				max_future_q = np.max(q_table[observation])
				q_table[state][action] = q_table[state][action] + LEARNING_RATE*(reward + GAMMA * q_table[observation][new_action] - q_table[state][action])
			else:
				break
			
			state = observation

		EPSILON *= EPSILON_DECAY

		# plotting
		episodes.append(episode)
		episode_rewards.append(episode_reward)
		
		if not episode % SHOW:
			average_reward = sum(episode_rewards[-SHOW:])/SHOW
			episode_rewards_stats['ep'].append(episode)
			episode_rewards_stats['avg'].append(average_reward)
			episode_rewards_stats['max'].append(max(episode_rewards[-SHOW:]))
			episode_rewards_stats['min'].append(min(episode_rewards[-SHOW:]))
			episode_rewards_stats['rewards'].append(episode_reward)
			episode_rewards_stats['timesteps'].append(i)

		print(f"Episode reward : {episode_reward}")
		print(f"Lasted for {i} timesteps")
		print("\n")

	env.close()

	# save q table
	with open(f"FINAL/stage1/pkl.pickle", "wb") as f:
	    pickle.dump(q_table, f)

	# plot progress of the agent
	fig1 = plt.figure()
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['avg'], label="Average Rewards")
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['max'], label="Max rewards")
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['min'], label="Min rewards")
	plt.rcParams["figure.figsize"] = (11,10)
	plt.legend(loc=4)
	# plt.savefig('FINAL/stage1/plot_e25000_stage1_stats.png')

	fig2 = plt.figure()
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['rewards'], label="Reward")
	plt.title('Rewards per Episode')
	plt.rcParams["figure.figsize"] = (11,10)
	# plt.savefig('FINAL/stage1/plot_e25000_stage1_rewards.png')
	
	fig3 = plt.figure()
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['timesteps'], label = "Timesteps")
	plt.title('Tmesteps per episode')
	plt.rcParams["figure.figsize"] = (11,10)
	# plt.savefig('FINAL/stage1/timesteps')

	fig4 = plt.figure()
	plt.plot(episodes, episode_rewards, label="All Episode Rewards")
	plt.title('Rewards per Episode')
	plt.rcParams["figure.figsize"] = (11,10)
	# plt.savefig('FINAL/stage1/rewards.png')

	fig5 = plt.figure()
	plt.plot(episode_rewards_stats['ep'], episode_rewards_stats['avg'], label="Average Rewards")
	plt.title('Average Rewards per Episode')
	plt.rcParams["figure.figsize"] = (11,10)
	# plt.savefig('FINAL/stage1/avgrewards.png')

	return episode_rewards

if __name__ == '__main__':
	LEARNING_RATE = 0.1
	GAMMA = 0.95
	EPSILON = 0.9
	EPISODES = 10
	EPSILON_DECAY = 0.99

	q_table = create_Q_table()

	# for training the agent
	# all_rewards = train(EPISODES, LEARNING_RATE, EPSILON, EPSILON_DECAY, GAMMA, q_table)

	# for testing the agent
	test_agent(q_table)