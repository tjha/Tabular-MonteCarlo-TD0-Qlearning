# Tejas Jha
# EECS 498: Reinforcement Learning - Homework 2

import numpy as np
import gym
import copy
import mytaxi
import mdp.mdp as mdp

# Global Variables for default actions taken (initialized here to be assigned in __main__)
given_policy = 0
true_value_fn = 0

# Part (a): Policy Evaluation
def evaluate_policy(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    return mdp.policy_eval(trans_mat, V_init, policy, theta, gamma, inplace)

# Part (b): Implementation for first-visit Monte Carlo Prediction for estimating state-value 
#           functions 
def mc_prediction(env, policy=given_policy, baseline=true_value_fn, gamma=1, episodes=50000):
    np.random.seed(3)
    env.seed(5)
    print("Hello")

if __name__ == '__main__':
    # Gather environment details and stored policy
    env = gym.make('Taxi-v3').unwrapped
    given_policy = np.load('policy.npy')
    trans_mat = env.P
    V_init = np.zeros(len(trans_mat))

    # Part (a): Evaluate value function of given policy in policy.npy
    true_value_fn = evaluate_policy(trans_mat, V_init, given_policy, theta=0.01, gamma=1)

    # Part (b): Utilization of first-visit Monte Carlo Prediction to plot rms vs episodes and 
    #           scatter plot of estimated value function (red x) verses the true value function 
    #           (blue empty o)
    mc_prediction(env)