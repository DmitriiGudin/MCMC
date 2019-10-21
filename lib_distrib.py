from __future__ import division
import numpy as np
import random


# AUTHOR: Dmitrii Gudin, University of Notre Dame, dgudin@nd.edu


# DESCRIPTION:
# ***********************************************
# The module provides a set of tools for quick and easy generation of distributions characterized by probability density function. It supports multidimensionality, including functions with multiple arguments, multiple parameters and multi-dimensional function values. A typical multidimensional function will look like this:
#     def func (x, p):
#         x, p = np.array([x]).flatten(), np.array([p]).flatten()
#         {some code calculating the function value f}
#         return f
# , where x is the input coordinate vector (single number, list, tuple or numpy array), p is the set of parameters (single number, list, tuple or numpy array), and f is the function value (single number, list, tuple or numpy array). The exact type does not matter, as in the function body all inputs are first converted into numpy arrays, so they become iterable.
# ***********************************************


# Multidimensional Gaussian density distribution function.
# -----
# [float vector] x: coordinate.
# [float vector] p: list of parameters in the following order: (mu_1, sigma_1, mu_2, sigma_2, ..., mu_n, sigma_n) for coordinate elements x_1, x_2, ..., x_n.
# -----
# Output:
# [float] G: the Gaussian function value.
# -----
# Examples of usage:
#     >>> import lib_distrib as l
#     >>> l.Gauss (2, (0, 1))
#     0.053990966513188063
#     >>> l.Gauss ((1, 3), (0, 1, 2, 5))
#     0.018924176795831745
def Gauss (x, p):
    x, p = np.array([x]).flatten(), np.array([p]).flatten()
    # Define the 1-dimensional Gaussian function.
    def Gauss_1d (x_1d, mu_1d, sigma_1d):
        return 1/sigma_1d/np.sqrt(2*np.pi)*np.exp(-(x_1d-mu_1d)**2/2/sigma_1d**2)
    # Multiply all 1-dimensional Gaussian values to obtain the multidimensional Gaussian value.
    G = np.prod([Gauss_1d(x_1d, mu_1d, sigma_1d) for x_1d, mu_1d, sigma_1d in zip (x, p[::2], p[1::2])])
    # Return the result as a number.
    return G


# Generates a point for a given distribution and returns its coordinate.
# -----
# [float vector function] func: density distribution function name. The function should be of the type func (x, p) as explained above.
# [float vector] p: list of function parameters.
# [float vector] left_lim: list of minimum coordinate components of the box for Monte-Carlo sampling.
# [float vector] right_lim: list of maximum coordinate components of the box for Monte-Carlo sampling.
# [float vector] low_lim: list of minimum function value components of the box for Monte-Carlo sampling.
# [float vector] up_lim: list of maximum function value components of the box for Monte-Carlo sampling.
# -----
# Output:
# [numpy array] or [float] x: the point coordinate vector.
# -----
# Examples of usage:
#     >>> import lib_distrib as l
#     >>> l.draw_rand (l.Gauss, (0,3), -20, 20, 0, (l.Gauss(0,(0,3))))
#     3.0757019662666885
#     >>> p = [0,3,2,5]
#     >>> l.draw_rand (l.Gauss, p, (-50,-50), (50,50), (0,0), (l.Gauss(0,p[:2]), l.Gauss(0,p[2:])))
#     array([ 4.78994773, -3.22910379])
def draw_rand (func, p, left_lim, right_lim, low_lim, up_lim):
    p = np.array([p]).flatten()
    left_lim = np.array([left_lim]).flatten()
    right_lim = np.array([right_lim]).flatten()
    low_lim = np.array([low_lim]).flatten()
    up_lim = np.array([up_lim]).flatten()
    random.seed()
    # Do a loop of rolls, until one satisfies the Monte-Carlo conditions.
    while(True):
        # Generate a random coordinate vector.
        Roll = np.array([random.uniform(l_l,r_l) for l_l, r_l in zip(left_lim, right_lim)]).flatten()
        # Generate a random function-space vector.
        Roll_value = np.array([random.uniform(l_l, u_l) for l_l, u_l in zip(low_lim, up_lim)]).flatten()
        # The roll is only accepted if all the elements of the function-space vector are less than those of the corresponding function values with the coordinate vector as an argument.
        for i in range (0, len(Roll_value)):
            if Roll_value[i]>np.array([func(Roll,p)]).flatten()[i]:
                break
            if len(Roll)==1: return Roll[0]
            return Roll


# Generates a list of coordinates for a given distribution.
# ----
# [float vector function] func: density distribution function name. The function should be of the type func (x, p) as explained above.
# [float vector] p: list of function parameters.
# [float vector] left_lim: list of minimum coordinate components of the box for Monte-Carlo sampling.
# [float vector] right_lim: list of maximum coordinate components of the box for Monte-Carlo sampling.
# [float vector] low_lim: list of minimum function value components of the box for Monte-Carlo sampling.
# [float vector] up_lim: list of maximum function value components of the box for Monte-Carlo sampling.
# [int] N: number of points to generate.
# -----
# Output:
# [list of [numpy array]] output_list: list of the generated coordinate vectors.
# ----- 
# Example of usage:
#     >>> import lib_distrib as l
#     >>> p = [0, 3, 1, 5]
#     >>> l.gen_distrib (l.Gauss, (0,3), -20, 20, 0, (l.Gauss(0,(0,3))),7)
#     [5.8502994077335586, 1.6415874772056007, -8.2615979908369717, 0.42729648568183265, 4.0388538740797451, -0.71863002513151386, 0.37606182298244661]
#     >>> l.gen_distrib (l.Gauss, p, (-50,-50), (50,50), (0,0), (l.Gauss(0,p[:2]), l.Gauss(0,p[2:])), 5)
#     [array([-2.22807008,  1.20392057]), array([-5.68009949,  5.41191758]), array([-0.29557321, -8.68410193]), array([-2.94120285,  9.88142477]), array([-0.46809802, -6.34584931])]
def gen_distrib (func, p, left_lim, right_lim, low_lim, up_lim, N=1000):
    p = np.array([p]).flatten()
    left_lim = np.array([left_lim]).flatten()
    right_lim = np.array([right_lim]).flatten()
    low_lim = np.array([low_lim]).flatten()
    up_lim = np.array([up_lim]).flatten()
    output_list = []
    for i in range(0, N):
        # Use the point generation function N times and form a list out of results.
        output_list.append(draw_rand(func, p, left_lim, right_lim, low_lim, up_lim))
    return output_list



