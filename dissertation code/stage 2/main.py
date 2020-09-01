import gym
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 

env = gym.make('DebrisAvoidance-v1')

step_150 = 150
load_q_table = 0

# loads or creates q table
def create_Q_table_satellite():
	q_table = None
	# if q_table is None:
	# 	q_table = {}
	# 	for i in range(0, 5):
	# 		for j in range(0, 5):
	# 			for l in range(0, 5):
	# 				q_table[(i, j, l)] = [np.random.uniform(-5, 0) for z in range(3)]

	load_q_table = 1
	with open('qtable_satellite.pickle', "rb") as f:
	    q_table = pickle.load(f)
	return q_table

# loads or creates q table
def create_Q_table_debris():
	q_table = None
	# if q_table is None:
	# 	q_table = {}
	# 	for i in range(0, 5):
	# 		for j in range(0, 5):
	# 				q_table[(i, j)] = [np.random.uniform(-5, 0) for z in range(3)]

	with open('qtable_debris.pickle', "rb") as f:
		q_table = pickle.load(f)
	return q_table


def get_tuple(state):
    return tuple(state) 

# for testing the environment
def test(q_table_satellite, q_table_debris):
	state = env.reset()
	state_satellite = get_tuple(state[0])
	state_debris = get_tuple(state[1])
	for i in range(1000):
		action_satellite = np.argmax(q_table_satellite[state_satellite])
		action_debris = np.argmax(q_table_debris[state_debris])

		actions = (action_satellite, action_debris)
		observation, reward, done, info = env.step(action)
		observation_satellite = get_tuple(observation[0])
		observation_debris = get_tuple(observation[1])	
		env.render()

		if done: 
			break
		
		state = observation


# train the agents
def train(EPISODES, LEARNING_RATE, EPSILON, GAMMA, q_table_satellite, q_table_debris):
	episode_rewards_satellite = []
	episode_rewards_debris = []
	episode_rewards_stats_satellite = {'ep': [], 'avg': [], 'max': [], 'min': [], 'rewards': [], 'timesteps':[]}
	episode_rewards_stats_debris = {'ep': [], 'avg': [], 'max': [], 'min': [], 'rewards': []}

	epsilons = []
	timesteps = []

	for episode in range(EPISODES):
		epsilons.append(EPSILON)

		state = env.reset()
		state_satellite = get_tuple(state[0])
		state_debris = get_tuple(state[1])

		done = False
		print(f"Running episode {episode}")
		print(f"EPSILON {EPSILON}\n")

		episode_reward_satellite = 0
		episode_reward_debris = 0

		for i in range(1500):
			if np.random.uniform() > EPSILON:
				action_satellite = np.argmax(q_table_satellite[state_satellite])
				action_debris = np.argmax(q_table_debris[state_debris])
			else:
				action_satellite = np.random.randint(0, 3)
				action_debris = np.random.randint(0, 3)

			actions = (action_satellite, action_debris)
			observation, reward, done, info_satellite = env.step(actions)
			
			observation_satellite = get_tuple(observation[0])
			observation_debris = get_tuple(observation[1])

			reward_satellite = reward[0]
			reward_debris = reward[1]

			env.render()
			
			new_action_satellite = np.argmax(q_table_satellite[observation_satellite])
			new_action_debris = np.argmax(q_table_debris[observation_debris])

			episode_reward_satellite += reward_satellite
			episode_reward_debris += reward_debris

			if not done:	
				max_future_q_satellite = np.max(q_table_satellite[observation_satellite])
				q_table_satellite[state_satellite][action_satellite] = q_table_satellite[state_satellite][action_satellite] + LEARNING_RATE*(reward_satellite + GAMMA * q_table_satellite[observation_satellite][new_action_satellite] - q_table_satellite[state_satellite][action_satellite])

				max_future_q_debris = np.max(q_table_debris[observation_debris])				
				q_table_debris[state_debris][action_debris]  = q_table_debris[state_debris][action_debris] + LEARNING_RATE * (reward_debris + GAMMA * q_table_debris[observation_debris][new_action_debris] - q_table_debris[state_debris][action_debris])
			else:
				break
			
			state_satellite = observation_satellite
			state_debris = observation_debris

		EPSILON *= EPSILON_DECAY
		episode_rewards_satellite.append(episode_reward_satellite)
		episode_rewards_debris.append(episode_reward_debris)

		if not episode % step_150:
			average_reward_satellite = sum(episode_rewards_satellite[-step_150:])/step_150
			episode_rewards_stats_satellite['ep'].append(episode)
			episode_rewards_stats_satellite['avg'].append(average_reward_satellite)
			episode_rewards_stats_satellite['max'].append(max(episode_rewards_satellite[-step_150:]))
			episode_rewards_stats_satellite['min'].append(min(episode_rewards_satellite[-step_150:]))
			episode_rewards_stats_satellite['rewards'].append(episode_reward_satellite)
			episode_rewards_stats_satellite['timesteps'].append(i)

			average_reward_debris = sum(episode_rewards_debris[-step_150:])/step_150
			episode_rewards_stats_debris['ep'].append(episode)
			episode_rewards_stats_debris['avg'].append(average_reward_debris)
			episode_rewards_stats_debris['max'].append(max(episode_rewards_debris[-step_150:]))
			episode_rewards_stats_debris['min'].append(min(episode_rewards_debris[-step_150:]))
			episode_rewards_stats_debris['rewards'].append(episode_reward_debris)

		print(f"Episode reward satellite: {episode_reward_satellite}")
		print(f"Episode reward debris: {episode_reward_debris}")
		print(f"Lasted for {i} timesteps")
		print("\n")

	env.close()

	with open(f"{location}qtable_satellite.pickle", "wb") as f:
	    pickle.dump(q_table_satellite, f)

	with open(f"{location}qtable_debris.pickle", "wb") as f:
	    pickle.dump(q_table_debris, f)

    # plot statistics
	fig1 = plt.figure()
	plt.plot(episode_rewards_stats_satellite['ep'], episode_rewards_stats_satellite['avg'], label="Average Rewards")
	plt.plot(episode_rewards_stats_satellite['ep'], episode_rewards_stats_satellite['max'], label="Max rewards")
	plt.plot(episode_rewards_stats_satellite['ep'], episode_rewards_stats_satellite['min'], label="Min rewards")
	plt.legend(loc=4)
	plt.grid(True)
	# plt.savefig(location + 'stats_satellite.png')

	fig2 = plt.figure()
	plt.plot(episode_rewards_stats_debris['ep'], episode_rewards_stats_debris['avg'], label="Average Rewards")
	plt.plot(episode_rewards_stats_debris['ep'], episode_rewards_stats_debris['max'], label="Max rewards")
	plt.plot(episode_rewards_stats_debris['ep'], episode_rewards_stats_debris['min'], label="Min rewards")
	plt.legend(loc=1)
	plt.grid(True)
	# plt.savefig(location + 'stats_debris.png')

	fig3 = plt.figure()
	plt.plot(episode_rewards_stats_satellite['ep'], episode_rewards_stats_satellite['rewards'])
	plt.title('Satellite : Rewards per Episode')
	plt.grid(True)
	# plt.savefig(location +'rewards_satellite.png')
	
	fig4 = plt.figure()
	plt.plot(episode_rewards_stats_debris['ep'], episode_rewards_stats_debris['rewards'])
	plt.title('Obstacles : Rewards per Episode')
	plt.grid(True)
	# plt.savefig(location + 'rewards_debris.png')

	fig5 = plt.figure()
	plt.plot(episode_rewards_stats_satellite['ep'], episode_rewards_stats_satellite['timesteps'])
	plt.title('Timesteps per episode')
	plt.grid(True)
	# plt.savefig(location + 'timesteps.png')

	return

if __name__ == '__main__':
	LEARNING_RATE = 0.05
	GAMMA = 0.95
	EPSILON_DECAY = 0.999
	EPSILON = 0.9
	EPISODES = 10000
	q_table_satellite = create_Q_table_satellite()
	q_table_debris = create_Q_table_debris()

	train(EPISODES, LEARNING_RATE, EPSILON, GAMMA, q_table_satellite, q_table_debris)
	test(q_table_satellite, q_table_debris)