# Tejas Jha
# EECS 498: Reinforcement Learning - Homework 2

import numpy as np
import gym
import copy
import mytaxi
import math
import mdp.mdp as mdp
import matplotlib.pyplot as plt

###################  Setup in Part (a) ####################
# Part (a): Policy Evaluation
def evaluate_policy(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    return mdp.policy_eval(trans_mat, V_init, policy, theta, gamma, inplace)

# Global Variables for default actions taken
given_policy = np.load('policy.npy')
# Gather environment details and stored policy
env = gym.make('Taxi-v3').unwrapped
trans_mat = env.P
V_init = np.zeros(len(trans_mat))
actions = [0,1,2,3,4,5]
###########################################################

# Part (a): Evaluate value function of given policy in policy.npy
true_value_fn = evaluate_policy(trans_mat, V_init, given_policy, theta=0.01, gamma=1)

# Helper function - Generate the steps in an episode for an environment and given policy
# Returns: (T, 3) numpy array for length T with elements corresponding to each time step for 
# states, actions, and Rewards
def generate_episode(env, policy):
    states_visited = list()
    actions_taken = list()
    rewards_received = list()

    # edge case implementation error possible (currently do not check if initial state is final state)
    states_visited.append(env.reset())
    actions_taken.append(np.random.choice(actions,p=policy[states_visited[-1]]))
    next_state, reward, done, info = env.step(actions_taken[-1])

    while not done:
        rewards_received.append(reward)
        states_visited.append(next_state)
        actions_taken.append(np.random.choice(actions,p=policy[states_visited[-1]]))
        next_state, reward, done, info = env.step(actions_taken[-1])

    rewards_received.append(reward)

    return states_visited, actions_taken, rewards_received
        



# Part (b): Implementation for first-visit Monte Carlo Prediction for estimating state-value 
#           functions
#           Returns: rms w.r.t baseline at end of each episode (rms), final value function (V)
def mc_prediction(env, policy=given_policy, baseline=true_value_fn, gamma=1, episodes=50000):
    np.random.seed(3)
    env.seed(5)
    rms = np.zeros(episodes)
    V = np.random.rand(env.nS)
    returns = [[] for _ in range(env.nS)]

    # Loop over each episode run
    for i_episode in range(episodes):
        # Generate an episode following policy
        states_visited, actions_taken, rewards_received = generate_episode(env, policy)
        G = 0
        # Loop over each step of the episode
        for step in range(len(states_visited)-1, -1, -1):
            G = gamma*G + rewards_received[step]
            if states_visited.index(states_visited[step]) == step:
                returns[states_visited[step]].append(G)
                V[states_visited[step]] = sum(returns[states_visited[step]]) / float(len(returns[states_visited[step]]))
        rms_value = math.sqrt(sum((V - baseline)**2)/float(len(V)))
        rms[i_episode] = rms_value

        # To keep track of progress in loop
        if i_episode % 10000 == 0:
            print("Completed episode: " + str(i_episode))

    return rms, V

if __name__ == '__main__':
    
    # Part (b): Utilization of first-visit Monte Carlo Prediction to plot rms vs episodes and 
    #           scatter plot of estimated value function (red x) verses the true value function 
    #           (blue empty o)
    rms, V = mc_prediction(env)
    episodes = np.arange(len(rms))
    states = np.arange(env.nS)

    fig = plt.figure(figsize=(40,20))
    txt = 'Figure 1: Plot for exploring taxi-v3'
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    # Generate plots for Part(b)
    plt.subplot(241)
    plt.plot(episodes,rms, 'r')
    plt.xlabel("Episodes")
    plt.ylabel("RMS")
    plt.title("MC prediction")

    plt.subplot(242)
    plt.plot(states,V, 'rx')
    plt.plot(states, true_value_fn, 'bo')
    plt.xlabel("V(s)")
    plt.ylabel("RMS")
    plt.title("MC prediction")



    plt.show()