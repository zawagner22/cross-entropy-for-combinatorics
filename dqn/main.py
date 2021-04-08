"""
Code from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
"""



import argparse, os
import numpy as np
import agents as Agents
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
					description='Deep Q Learning: From Paper to Code')
	# the hyphen makes the argument optional
	parser.add_argument('-n_games', type=int, default=100000,
						help='Number of games to play')
	parser.add_argument('-lr', type=float, default=0.003,
						help='Learning rate for optimizer')
	parser.add_argument('-eps_min', type=float, default=0.001,
			help='Minimum value for epsilon in epsilon-greedy action selection')
	parser.add_argument('-gamma', type=float, default=0.99,
									help='Discount factor for update equation.')
	parser.add_argument('-eps_dec', type=float, default=0.99999,
						help='Linear factor for decreasing epsilon')
	parser.add_argument('-eps', type=float, default=1.0,
		help='Starting value for epsilon in epsilon-greedy action selection')
	parser.add_argument('-max_mem', type=int, default=80000, #~13Gb
								help='Maximum size for memory replay buffer')
	parser.add_argument('-bs', type=int, default=32,
							help='Batch size for replay memory sampling')
	parser.add_argument('-replace', type=int, default=1000,
						help='interval for replacing target network')
	parser.add_argument('-env', type=str, default='gym_conj21:conj21-v0',
							help='Atari environment.\nPongNoFrameskip-v4\n \
								  BreakoutNoFrameskip-v4\n \
								  SpaceInvadersNoFrameskip-v4\n \
								  EnduroNoFrameskip-v4\n \
								  AtlantisNoFrameskip-v4\n \
								  gym_c4free:c4free-v0\n \
								  gym_poshen:poshen-v0')
	parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
	parser.add_argument('-load_checkpoint', type=bool, default=False,
						help='load model checkpoint')
	parser.add_argument('-path', type=str, default='models/',
						help='path for model saving/loading')
	parser.add_argument('-algo', type=str, default='DuelingDDQNAgent',
					help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
	
	args = parser.parse_args()

	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	env = make_env(env_name='gym_conj21:conj21-v0')
	best_score = -np.inf
	best_avg_score = -np.inf
	agent_ = getattr(Agents, args.algo)
	agent = agent_(gamma=args.gamma,
				  epsilon=args.eps,
				  lr=args.lr,
				  input_dims=env.observation_space.shape,
				  n_actions=env.action_space.n,
				  mem_size=args.max_mem,
				  eps_min=args.eps_min,
				  batch_size=args.bs,
				  replace=args.replace,
				  eps_dec=args.eps_dec,
				  chkpt_dir=args.path,
				  algo=args.algo,
				  env_name=args.env)
	

	if args.load_checkpoint:
		agent.load_models()

	fname = args.algo + '_' + args.env.replace(':','-') + '_alpha' + str(args.lr) +'_' \
			+ str(args.n_games) + 'games'+ str(args.eps_dec) + 'epsdec' 
	figure_file = 'plots/' + fname + '.png'
	scores_file = fname + '_scores.npy'

	scores, eps_history = [], []
	n_steps = 0
	steps_array = []
	for i in range(args.n_games):
		done = False
		observation = env.reset()
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward

			#if not args.load_checkpoint:
			agent.store_transition(observation, action,
								 reward, observation_, int(done))
			agent.learn()
			observation = observation_
			n_steps += 1
		scores.append(score)
		steps_array.append(n_steps)

		avg_score = np.mean(scores[-100:])
		print('episode: ', i,'score: %.2f' % score,
			 ' average score %.1f' % avg_score, 'best average %.2f' % best_avg_score, 'best score %.2f' % best_score, 
			'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

		if avg_score > best_avg_score:
			#if not args.load_checkpoint:
			agent.save_models()
			best_avg_score = avg_score
		if score > best_score:
			best_score = score
			print(observation[0])
			with open('best_species/best_species_'+str(len(observation[0]))+'.txt', 'w') as f:
				for i in range(len(observation[0])):
					for j in range(len(observation[0])):
						f.write(str(int(observation[0][i][j]))+" ")
					f.write("\n")
				f.write("\n"+str(score))
		
		eps_history.append(agent.epsilon)
		
		if i%1000==0:
			plot_learning_curve(steps_array, scores, eps_history, figure_file)

	x = [i+1 for i in range(len(scores))]
	
	#np.save(scores_file, np.array(scores))