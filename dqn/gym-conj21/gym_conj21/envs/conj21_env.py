import networkx as nx
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class Conj21Env(gym.Env):

	def __init__(self):
		self.N = 19
		self.A = np.zeros((self.N,self.N), dtype=np.float32)
		self.current_index = [0,1]
		self.index_mat = np.zeros((self.N,self.N), dtype=np.float32)
		self.index_mat[0,1] = 1
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(0, 1, shape=(2,self.N,self.N), dtype=np.float32)

	def step(self, action):
		self.A[self.current_index[0],self.current_index[1]] = action
		self.A[self.current_index[1],self.current_index[0]] = action
		self.index_mat[self.current_index[0],self.current_index[1]] = 0
		episode_over = False
		reward = 0
		if self.current_index[1] < self.N-1:
			self.current_index[1] += 1
		elif self.current_index[0] < self.N-2:
			self.current_index[0] += 1
			self.current_index[1] = self.current_index[0] + 1
		else:
			episode_over = True
			reward = self.reward_calc()
		
		
		self.index_mat[self.current_index[0],self.current_index[1]] = 1
		
		
		
		A = np.expand_dims(self.A, axis=0)
		index_mat = np.expand_dims(self.index_mat, axis=0) 
		
		obs = np.concatenate((A, index_mat), axis=0)
		return obs, reward, episode_over, {}
		
	def reward_calc(self):
		#Example reward function, for Conjecture 2.1
		#Given a graph, it minimizes lambda_1 + mu.
		#Takes about a day to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
		#Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.
		
		#Construct the graph 
		G= nx.Graph()
		G.add_nodes_from(list(range(self.N)))
		for i in range(self.N):
			for j in range(i+1,self.N):
				if self.A[i,j] == 1:
					G.add_edge(i,j)
		
		#G is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
		if not (nx.is_connected(G)):
			return -np.inf
			
		#Calculate the eigenvalues of G
		evals = np.linalg.eigvalsh(self.A)
		evalsRealAbs = np.zeros(len(evals))
		for i in range(len(evals)):
			evalsRealAbs[i] = abs(evals[i])
		
		#Calculate the matching number of G
		maxMatch = nx.max_weight_matching(G)
		lambda1 = max(evalsRealAbs)
		mu = len(maxMatch)
			
		#Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
		#We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
		myScore = np.sqrt(self.N-1) + 1 - lambda1 - mu
		if myScore > 0:
			#You have found a counterexample. Do something with it.
			print(state)
			nx.draw_kamada_kawai(G)
			plt.show()
			exit()
			
		return myScore
		

	def reset(self):
		self.A = np.zeros((self.N,self.N), dtype=float)
		self.current_index = [0,1]
		self.index_mat = np.zeros((self.N,self.N), dtype=float)
		self.index_mat[0,1] = 1
		A = np.expand_dims(self.A, axis=0)
		index_mat = np.expand_dims(self.index_mat, axis=0) 
		
		obs = np.concatenate((A, index_mat), axis=0)
		
		return obs
		
		
	def render(self, mode='human'):
		pass
	def close(self):
		pass
