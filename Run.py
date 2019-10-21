from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import lib_distrib
import MCMC
import Data_part_2


# General parameters.
N_logging = 25
dL_critical = 0.0000001 # Used when calculating acceptance rate; change of likelihood below this is considered null.
grid_density = 100 # How many cells per 1 dimension to use when plotting cross-distribution for 2 parameters.
cumul_points = 15000 # How many points the cumulative distribution graph to calculate for (essentially its resolution).


# Parameters for part 1.
N_points_1 = 200 # Number of generated data points.
x_min_1, x_max_1 = -20, 20 # The range to generate data points in.
A_1, B_1 = 7, -21 # y = Ax + B
dx_1, dy_1 = 1, 8 # Amount of noise.
dx_factor_1, dy_factor_1 = 20, 20 # The noise will be generated from -dxy_factor_1*dxy_1 to dxy_factor_1*dxy_1.
dp_1 = (0.01, 0.01)
dp_decay_1 = (0.0002, 0.0002)
N_coarse_1 = 15000
N_fine_1 = 15000
N_final_1 = 15000


# Parameters for part 2.
dp_2 = (0.4, 0.4, 0.1)
dp_decay_2 = (0.001, 0.001, 0.001)
N_coarse_2 = 1500
N_fine_2 = 1500
N_final_2 = 1500


# Linear function from part 1.
def linear (x, p):
    return p[0]*x + p[1]


# Sinusoidal+linear function from part 2.
def sinusoidal (x, p):
    return p[0]*np.sin(p[1]*x) + p[2]*x


# Plot the noisy linear data generated in part 1.
def plot_initial_data_1 (data_x, data_y):
    plt.clf()
    plt.title("Linear data with noise", size=24)
    plt.xlabel('x', size=24)
    plt.ylabel('y', size=24)
    plt.tick_params(labelsize=18)
    recs = [mpatches.Rectangle((0,0),1,1,fc='blue'), mpatches.Rectangle((0,0),1,1,fc='red')]
    legend_labels = ["Noisy data", "Original line"]
    plt.legend(recs, legend_labels, loc=4, fontsize=18)
    plt.scatter(data_x, data_y, color='blue', s=25, marker='o')
    plt.xlim(min(data_x), max(data_x))
    plt.ylim(min(data_y), max(data_y))
    plt.plot([x_min_1, x_max_1], [A_1*x_min_1+B_1, A_1*x_max_1+B_1], color='red', linewidth=3, linestyle='--')
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig('InitialData_1.png', dpi=100)
    plt.close()


# Plot the data for part 2.
def plot_initial_data_2 (data_x, data_y):
    plt.clf()
    plt.title("Sinusoidal+linear data", size=24)
    plt.xlabel('x', size=24)
    plt.ylabel('y', size=24)
    plt.tick_params(labelsize=18)
    plt.scatter(data_x, data_y, color='blue', s=25, marker='o')
    plt.xlim(min(data_x), max(data_x))
    plt.ylim(min(data_y), max(data_y))
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig('InitialData_2.png', dpi=100)
    plt.close()


# Calculates acceptance rates for three modes. The arguments are arrays of the likelihood values for each mode.
def calc_acceptance_rates (L_coarse, L_fine, L_final):
    L_list = [L_coarse, L_fine, L_final]
    val_list = [[],[],[]] # Lists containing 1 if L changed and 0 if not for each step.
    for i in range(0,3):
        L_old = L_list[i][0]
        for L_new in L_list[i]:
            if abs(L_new-L_old)<dL_critical:
                val_list[i].append(0)
            else:
                val_list[i].append(1)
        val_list[i]=val_list[i][1:] # Remove the first element (1) as redundant, since the count starts with the 1st step, not 0th.
    return np.array([np.mean(val_list[0]), np.mean(val_list[1]), np.mean(val_list[2])])


# Plots the posterior distribution for all individual and dual combinations of the supplied list of arrays of parameter values.
# par_names is the list of the names of the parameters to put the correct labels on the plots.
# part: for this project, either "1" or "2"
def plot_distr (data_list, par_names, part):
    data_list = np.array(data_list)
    # Plot individual distributions.
    for data, par_name in zip(data_list, par_names):
        plt.clf() 
        if part==1:
            plt.title("y = A*x + B", size=24)
        else:
            plt.title("y = A*sin(B*x) + C*x", size=24)
        plt.xlabel(par_name, size=24)
        plt.ylabel("Occurrence", size=24)
        plt.tick_params(labelsize=18)
        plt.hist(data, bins=50, color='black', fill=False, linewidth=2, histtype='step', density=True)
        plt.gcf().set_size_inches(25.6, 14.4)
        plt.gcf().savefig('Parameter_'+str(par_name)+'_'+str(part)+'.png')
        plt.close()
    # Plot dual distributions.
    for i in range (0, len(par_names)):
        for j in range (i+1, len(par_names)):
            plt.clf()
            if part==1:
                plt.title("y = A*x + B", size=24)
            else:
                plt.title("y = A*sin(B*x) + C*x", size=24)
            plt.xlabel(par_names[i], size=24)
            plt.ylabel(par_names[j], size=24)
            plt.tick_params(labelsize=18)
            extent=[min(data_list[i]), max(data_list[i]), min(data_list[j]), max(data_list[j])]
            x_grid, y_grid = np.meshgrid(np.linspace(extent[0],extent[1],grid_density+1), np.linspace(extent[2],extent[3],grid_density+1))
            # griddata ignores duplicates, so they need to be summed up.
            data_temp_x, data_temp_y, data_temp_N = [], [], []
            for x, y in zip (data_list[i], data_list[j]):
                if x in data_temp_x:
                    data_temp_N[data_temp_x.index(x)]+=1
                else:
                    data_temp_x.append(x)
                    data_temp_y.append(y)
                    data_temp_N.append(1)
            data_temp_x, data_temp_y, data_temp_N = np.array(data_temp_x), np.array(data_temp_y), np.array(data_temp_N)
            # Now, use griddata.
            grid_data = griddata((data_temp_x, data_temp_y), data_temp_N, (x_grid, y_grid), method='cubic', fill_value=0)
            grid_data[grid_data<0] = 0
            plt.imshow(grid_data, origin='lower', cmap=plt.get_cmap('hot'), extent=extent, aspect='auto', interpolation='bilinear')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label('Frequency', size=24)
            plt.gcf().set_size_inches(25.6, 14.4)
            plt.gcf().savefig('Parameters_'+str(par_names[i])+'_'+str(par_names[j])+'_'+str(part)+'.png')
            plt.close()
            

# Calculates cumulative distribution for the parameters, plots it and calculates confidence intervals.
# data_list, par_names, part - same as in plot_distr.
# conf_int - set of confidence values, for example (0.68, 0.95, 0.997).
def plot_cumul(data_list, par_names, part, conf_int):
    for data, par_name in zip (data_list, par_names):
        x_points = np.linspace(min(data), max(data), cumul_points+1)
        cumul_data = list(np.array([sum(i <= num for i in data) for num in x_points])/len(data))
        # Find the index for closest point to the 50% mark.
        opt_y = min(cumul_data, key=lambda x:abs(x-0.5)) 
        opt_i = cumul_data.index(opt_y)
        opt_x = x_points[opt_i]
        # Perform the confidence interval search.
        conf_int_x = []
        for i in range (0, len(conf_int)):
            # Find the closest points to the confidence values and their indeces.
            bot_y = min(cumul_data, key=lambda x:abs(x-(1-conf_int[i])/2))
            top_y = min(cumul_data, key=lambda x:abs(x-(1+conf_int[i])/2))
            bot_i, top_i = cumul_data.index(bot_y), cumul_data.index(top_y)   
            # Calculate the parameter values at these points and the exact confidence interval.
            left_x = x_points[bot_i]
            right_x = x_points[top_i]
            conf = top_y-bot_y
            conf_int_x.append([left_x, right_x, bot_y, top_y, conf])
        # Plot the distribution and the intervals.
        plt.clf()
        if part==1:
            plt.title("y = A*x + B - cumulative distribution", size=24)
        else:
            plt.title("y = A*sin(B*x) + C*x - cumulative distribution", size=24)
        plt.xlabel(par_name, size=24)
        plt.ylabel("Cumulative fraction", size=24)
        plt.xlim(min(x_points), max(x_points))
        plt.ylim(0, 1)
        plt.tick_params(labelsize=18)
        plt.plot(x_points, cumul_data, color='blue', linewidth=3)
        plt.plot([opt_x, opt_x], [0, 1], color='red', linewidth=2)
        for i in range (0, len(conf_int)):
            plt.plot([conf_int_x[i][0], conf_int_x[i][0]], [0, 1], color='black', linewidth=1, linestyle='--')
            plt.plot([conf_int_x[i][1], conf_int_x[i][1]], [0, 1], color='black', linewidth=1, linestyle='--')
            plt.plot([x_points[0], x_points[-1]], [conf_int_x[i][2], conf_int_x[i][2]], color='black', linewidth=1, linestyle='--')
            plt.plot([x_points[0], x_points[-1]], [conf_int_x[i][3], conf_int_x[i][3]], color='black', linewidth=1, linestyle='--')
        plt.gcf().set_size_inches(25.6, 14.4)
        plt.gcf().savefig('Cumulative_'+str(par_name)+'_'+str(part)+'.png')
        plt.close()
        # Print confidence interval data.
        for i in range (0, len(conf_int)):
            print "Parameter", par_name, ", confidence level ", conf_int_x[i][4]*100, "%. ", par_name, " = [", conf_int_x[i][0], ", ", conf_int_x[i][1], "]."


# Plots the data set and the function predicted by the MCMC procedure.
def plot_part_2_result (data_x, data_y, func, p):
    plt.clf()
    plt.title("Sinusoidal+linear fit", size=24)
    plt.xlabel('x', size=24)
    plt.ylabel('y', size=24)
    plt.tick_params(labelsize=18)
    plt.scatter(data_x, data_y, color='blue', s=25, marker='o')
    plt.xlim(min(data_x), max(data_x))
    plt.ylim(min(data_y), max(data_y))
    # Generate the fit line.
    x_grid = np.linspace(min(data_x), max(data_x), 5000+1)
    y = func (x_grid, p)
    # Plot the fit line.
    plt.plot(x_grid, y, color='red', linewidth=3)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig('Result_2.png', dpi=100)
    plt.close()


def PART_1():
    # Generate clean data.
    data_x_1 = np.random.uniform(x_min_1, x_max_1, N_points_1)
    data_y_1 = A_1*data_x_1 + B_1
    # Add noise.
    data_x_1 += lib_distrib.gen_distrib (lib_distrib.Gauss, (0, dx_1), -dx_factor_1*dx_1, dx_factor_1*dx_1, 0, lib_distrib.Gauss(0, (0, dx_1)), N_points_1)
    data_y_1 += lib_distrib.gen_distrib (lib_distrib.Gauss, (0, dy_1), -dy_factor_1*dy_1, dy_factor_1*dy_1, 0, lib_distrib.Gauss(0, (0, dy_1)), N_points_1)
    # Plot the generated data.
    plot_initial_data_1 (data_x_1, data_y_1)
    # Perform the MCMC procedure. 
    data_1 = MCMC.MCMC (linear, (0,0), data_x_1, data_y_1, dp_1, dp_decay_1, N_coarse_1, N_fine_1, N_final_1, N_logging)
    # Retrieve the relevant data (the last N_final_1 batches):
    data_final_1 = data_1[N_coarse_1+N_fine_1+1:N_coarse_1+N_fine_1+N_final_1+1]
    # Calculate and output the optimal A, B values:
    A_1_opt, B_1_opt = np.median([d[1][0] for d in data_final_1]), np.median([d[1][1] for d in data_final_1])
    print "Optimal linear parameter values: A = ", A_1_opt, ", B = ", B_1_opt
    # Calculate the acceptance rates for all 3 modes. Print them out.
    acc_rates = calc_acceptance_rates ([d[2] for d in data_1[1:N_coarse_1+1]], [d[2] for d in data_1[N_coarse_1+1:N_coarse_1+N_fine_1+1]], [d[2] for d in data_final_1])
    print "Acceptance rate for the Coarse mode: ", round(acc_rates[0]*100, 2), "%."
    print "Acceptance rate for the Fine mode: ", round(acc_rates[1]*100, 2), "%."
    print "Acceptance rate for the Final mode: ", round(acc_rates[2]*100, 2), "%."
    # Plot the posterior distribution for A, B and both. Only for the Final mode.
    plot_distr ([[d[1][0] for d in data_final_1], [d[1][1] for d in data_final_1]], ['A','B'], 1)
    # Plot the cumulative sum graph and calculate 68% and 95% confidence intervals.
    plot_cumul ([[d[1][0] for d in data_final_1],[d[1][1] for d in data_final_1]], ['A','B'], 1, [0.68, 0.95])


def PART_2():
    # Retrieve the data set.
    data_x_2 = Data_part_2.x
    data_y_2 = Data_part_2.y
    # Plot the retrieved data.
    plot_initial_data_2 (data_x_2, data_y_2)
    # Perform the MCMC procedure.
    data_2 = MCMC.MCMC (sinusoidal, (0,0,0), data_x_2, data_y_2, dp_2, dp_decay_2, N_coarse_2, N_fine_2, N_final_2, N_logging)
    # Retrieve the relevant data (the last N_final_1 batches):
    data_final_2 = data_2[N_coarse_2+N_fine_2+1:N_coarse_2+N_fine_2+N_final_2+1]
    # Calculate and output the optimal A, B, C values:
    A_2_opt, B_2_opt, C_2_opt = np.median([d[1][0] for d in data_final_2]), np.median([d[1][1] for d in data_final_2]), np.median([d[1][2] for d in data_final_2])
    print "Optimal sinusoidal-linear parameter values: A = ", A_2_opt, ", B = ", B_2_opt, ", C = ", C_2_opt
    # Calculate the acceptance rates for all 3 modes. Print them out.
    acc_rates = calc_acceptance_rates ([d[2] for d in data_2[1:N_coarse_2+1]], [d[2] for d in data_2[N_coarse_2+1:N_coarse_2+N_fine_2+1]], [d[2] for d in data_final_2])
    print "Acceptance rate for the Coarse mode: ", round(acc_rates[0]*100, 2), "%."
    print "Acceptance rate for the Fine mode: ", round(acc_rates[1]*100, 2), "%."
    print "Acceptance rate for the Final mode: ", round(acc_rates[2]*100, 2), "%."
    # Plot the posterior distribution for A, B and both. Only for the Final mode.
    plot_distr ([[d[1][0] for d in data_final_2], [d[1][1] for d in data_final_2], [d[1][2] for d in data_final_2]], ['A','B','C'], 2)
    # Plot the cumulative sum graph and calculate 68% and 95% confidence intervals.
    plot_cumul ([[d[1][0] for d in data_final_2], [d[1][1] for d in data_final_2], [d[1][2] for d in data_final_2]], ['A','B','C'], 2, [0.68, 0.95])
    # Plot the predicted function over the initial dataset.
    plot_part_2_result (data_x_2, data_y_2, sinusoidal, (A_2_opt, B_2_opt, C_2_opt))


if __name__ == '__main__':
    PART_1()
    PART_2()
