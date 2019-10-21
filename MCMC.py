from __future__ import division
import numpy as np
import lib_distrib
import random
import time


# AUTHOR: Dmitrii Gudin, University of Notre Dame, dgudin@nd.edu


# DESCRIPTION:
# ***********************************************
# The module provides a Markov chain Monte-Carlo tool for estimating fit function parameters for multivariate multidimensional functions.
# ***********************************************


# NOTES:
# ***********************************************
# See the notes in lib_distrib.py, explaining the details of implementation of multidimensionality. They explain what shape a supplied function must have and provide terminology convention. 
# ***********************************************


# Calculates the likelihood for the hypothesized function with regards to the data. Maximum likelihood corresponds to the least squares for Gaussian distribution.
# -----
# [list of [float vector]] guess_f: list of values of the hypothesized function over a set of data coordinate points.
# [list of [float vector]] data_f: list of data function values over the set of the same coordinate points.
# -----
# Output:
#  [float] The likelihood value.
# -----
# Example of usage:
#     >>> import MCMC as m
#     >>> x = [1, 2, 3, 4, 5]
#     >>> data_f = [2, 4, 6, 8, 10]
#     >>> def guess_func (t):
#     ...     return 1.9*np.array(t) + 0.3
#     >>> guess_f = guess_func(x)
#     >>> print m.L (guess_f, data_f)
#     -0.1
#     >>> x = [(1,11), (2,22), (3,33), (4,44), (5,55)]
#     >>> data_f = [(2,2,1), (4,4,2), (6,6,3), (8,8,4), (10,10,5)]
#     >>> guess_f = guess_func(x)
#     >>> print m.L (guess_f, data_f)
#     -21043.4
def L (guess_f, data_f):
    guess_f = np.array([guess_f]).flatten()
    data_f = np.array([data_f]).flatten()
    return -sum([np.linalg.norm(g_f - d_f)**2 for g_f, d_f in zip(guess_f, data_f)])


# Performs MCMC approximation of parameters of the supplied fit function. Works in three sequential modes:
# 1. "Coarse" mode: large range of parameter change per step. The change is calculated from a gaussian distribution with the specified STD.
# 2. "Fine" mode: the range of parameter change per step decays over time.
# 3. "Final" mode: the final range of parameter change per step from "Fine" mode is used.
# -----
# [float vector function] func: fit function name (should be defined in advance). The function should be of the type func (x, p) as explained in lib_distrib.py.
# NOTE: if you want only some of the parameters varied and others fixed, then redefine the function before supplying to the algorithm, like so:
#     def func_new (x, p_var):
#         return func (x, (p[0]=1, p[1]=5, p_var[2:]*))
# [float vector] p_init: initial guess for the function parameters. 
# [list of [float vector]] data_x: data coordinate vectors.
# [list of [float vector]] data_f: data function value vectors.
# [float vector] dp: vector of STDs for parameter changes in the "Coarse" mode.
# [float vector] dp_decay: vector of parameter change decay factors, specifying by what fraction each of the dp elements decreases per step in the "Fine" mode. 
# [int] N_coarse: Number of steps in the "Coarse" mode.
# [int] N_fine: Number of steps in the "Fine" mode.   
# [int] N_final: Number of steps in the "Fine" mode.  
# [int] N_logging: How often (in terms of number of steps) to print the log messages. Non-positive means no logging.
# -----
# Output:
# [list of [list]] output_list: list of lists (one for each algorithm step) consisting of the following elements (in order):
#     - [int] output_step_num: step number.
#     - [tuple] output_p: list of parameter values guessed.
#     - [float] output_L: value of the likelihood function

def MCMC (func, p_init, data_x, data_f, dp, dp_decay, N_coarse=2000, N_fine=500, N_final=2000, N_logging=0):
    p_init = np.array([p_init]).flatten()
    data_x = [np.array([d]).flatten() for d in data_x]
    data_f = [np.array([d]).flatten() for d in data_f]
    dp = np.array([dp]).flatten()
    dp_decay = np.array([dp_decay]).flatten()
    # Start time count for logging.
    time_begin = time.time()
    # Calculate the initial likelihood value and initialize the output list.
    L_old = L_guess = L([func(x,p_init) for x in data_x], data_f)
    p_old = p_guess = p_init
    if len(p_init)==1:
        output_list = [[0, p_old[0], L_old]]
    else:
        output_list = [[0, p_old, L_old]]

    # "Coarse" mode.
    for i in range (1, N_coarse+1):
        # Guess the parameters by Gaussian displacement of the previous set.
        dp_gauss_param = (np.array([[0,dp_i] for dp_i in dp]).flatten())
        p_guess = p_old + lib_distrib.draw_rand (lib_distrib.Gauss, dp_gauss_param, -50*np.array(dp), 50*np.array(dp), [0]*len(dp), lib_distrib.Gauss([0]*len(dp),dp_gauss_param))
        # Calculate the new likelihood.
        L_guess = L([func(x,p_guess) for x in data_x], data_f)
        # Perform Metropolis-Hastings check to decide whether to accept the new parameter values.
        random.seed()
        if L_guess > L_old or random.uniform(0,1) < np.exp(L_guess-L_old):
            p_old = p_guess
            L_old = L_guess
        # Record the data.
        if len(p_old)==1:
            output_list.append([i, p_old[0], L_old])
        else:
            output_list.append([i, p_old, L_old])
        # Log message.
	if N_logging>0: 
            if i%N_logging==0:
	        print str(int(time.time()-time_begin)), "s: Coarse mode,", str(i), "out of", str(N_coarse), "steps done."

    # "Fine" mode.
    for i in range (N_coarse+1, N_coarse+N_fine+1):
        # Calculate the new parameter change.
        dp = np.multiply(dp,1-dp_decay)
        # Guess the parameters by Gaussian displacement of the previous set.
        dp_gauss_param = (np.array([[0,dp_i] for dp_i in dp]).flatten())
        p_guess = p_old + lib_distrib.draw_rand (lib_distrib.Gauss, dp_gauss_param, -50*np.array(dp), 50*np.array(dp), [0]*len(dp), lib_distrib.Gauss([0]*len(dp),dp_gauss_param))
        # Calculate the new likelihood.
        L_guess = L([func(x,p_guess) for x in data_x], data_f)
        # Perform Metropolis-Hastings check to decide whether to accept the new parameter values.
        random.seed()
        if L_guess > L_old or random.uniform(0,1) < np.exp(L_guess-L_old):
            p_old = p_guess
            L_old = L_guess
        # Record the data.
        if len(p_old)==1:
            output_list.append([i, p_old[0], L_old])
        else:
            output_list.append([i, p_old, L_old])
	# Log message.
	if N_logging>0: 
            if (i-N_coarse)%N_logging==0:
	        print str(int(time.time()-time_begin)), "s: Fine mode,", str(i-N_coarse), "out of", str(N_fine), "steps done."

    # "Final" mode.
    for i in range (N_coarse+N_fine+1, N_coarse+N_fine+N_final+1):
        # Guess the parameters by Gaussian displacement of the previous set.
        dp_gauss_param = (np.array([[0,dp_i] for dp_i in dp]).flatten())
        p_guess = p_old + lib_distrib.draw_rand (lib_distrib.Gauss, dp_gauss_param, -50*np.array(dp), 50*np.array(dp), [0]*len(dp), lib_distrib.Gauss([0]*len(dp),dp_gauss_param))
        # Calculate the new likelihood.
        L_guess = L([func(x,p_guess) for x in data_x], data_f)
        # Perform Metropolis-Hastings check to decide whether to accept the new parameter values.
        random.seed()
        if L_guess > L_old or random.uniform(0,1) < np.exp(L_guess-L_old):
            p_old = p_guess
            L_old = L_guess
        # Record the data.
        if len(p_old)==1:
            output_list.append([i, p_old[0], L_old])
        else:
            output_list.append([i, p_old, L_old])
	# Log message.
	if N_logging>0: 
            if (i-N_coarse-N_fine)%N_logging==0:
	        print str(int(time.time()-time_begin)), "s: Final mode,", str(i-N_coarse-N_fine), "out of", str(N_final), "steps done."

    # Return the result as list of lists.
    return output_list
         
