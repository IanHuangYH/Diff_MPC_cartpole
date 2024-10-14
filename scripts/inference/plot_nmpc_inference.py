import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# modify
NUM_STATE = 5
CONTROL_STEP = 50
WEIGHT_GUIDANC = 0.01 # non-conditioning weight

RESULT_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving/'
RESULT_NAME = 'nmpc_batch_4096_random112500_logminmax'
RESULT_FOLDER = os.path.join(RESULT_SAVED_PATH, RESULT_NAME)
# model_list = [10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
model_list = [40000]



num_modelread = len(model_list)
# diffusion data prepare
diffusion_u_filename = 'u_diffusion.npy'
diffusion_x_filename = 'x_diffusion.npy'
diffusion_j_filename = 'j_diffusion.npy'


x_diff_all = np.zeros((num_modelread, CONTROL_STEP, NUM_STATE))
u_diff_all = np.zeros((num_modelread, CONTROL_STEP))
j_diff_all = np.zeros((num_modelread, CONTROL_STEP))

for i in range(num_modelread):
    result_path = os.path.join(RESULT_FOLDER, str(model_list[i]))
    x_path = os.path.join(result_path, diffusion_x_filename)
    u_path  = os.path.join(result_path, diffusion_u_filename)
    j_path = os.path.join(result_path, diffusion_j_filename)
    x_diff_all[i,:,:] = np.load(x_path)
    u_diff_all[i,:] = np.load(u_path)
    j_diff_all[i,:] = np.load(j_path)



# mpc data prepare
ini_guess_num = 2
idx_ini_guess = [0,1]
x_mpc_all = np.zeros((ini_guess_num, CONTROL_STEP, NUM_STATE))
u_mpc_all = np.zeros((ini_guess_num, CONTROL_STEP))
j_mpc_all = np.zeros((ini_guess_num, CONTROL_STEP))
for i in range(ini_guess_num):
    iniguess_name = 'iniguess_' + str(idx_ini_guess[i]) + '_'
    mpc_x_filename = iniguess_name + 'x_mpc.npy'
    mpc_u_filename = iniguess_name + 'u_mpc.npy'
    mpc_j_filename = iniguess_name + 'j_mpc.npy'
    mpc_x_path = os.path.join(RESULT_FOLDER, mpc_x_filename)
    mpc_u_path = os.path.join(RESULT_FOLDER, mpc_u_filename)
    mpc_j_path = os.path.join(RESULT_FOLDER, mpc_j_filename)
    x_mpc_all[i,:,:] = np.load(mpc_x_path)
    u_mpc_all[i,:] = np.load(mpc_u_path)
    j_mpc_all[i,:] = np.load(mpc_j_path)




# other draw prepare
step = np.linspace(0,CONTROL_STEP-1, CONTROL_STEP)
PLOT_NUM = 6


plt.figure(figsize=(10, 8))
for i in range(2):
    x_track_mpc = x_mpc_all[i,:,:]
    u_track_mpc = u_mpc_all[i,:]
    j_track_mpc = j_mpc_all[i,:]
    
    plt.subplot(PLOT_NUM, 1, 1)
    plt.plot(step, x_track_mpc[:,0])
    plt.ylabel('Position (m)')
    plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(PLOT_NUM, 1, 2)
    plt.plot(step, x_track_mpc[:,1])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(PLOT_NUM, 1, 3)
    plt.plot(step, x_track_mpc[:,2])
    plt.ylabel('Angle (rad)')
    plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(PLOT_NUM, 1, 4)
    plt.plot(step, x_track_mpc[:,3])
    plt.ylabel('Ag Velocity (rad/s)')
    plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(PLOT_NUM, 1, 5)
    plt.plot(step, u_track_mpc)
    plt.ylabel('Ctl Input (N)')
    plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(PLOT_NUM, 1, 6)
    plt.plot(step, j_track_mpc)
    plt.ylabel('cost value')
    plt.xlabel('Control Step')
    plt.grid()

legend_list = ['MPC_pos','MPC_neg']
for i in range(num_modelread):
    legend_name = 'diff_'+str(model_list[i])
    legend_list.append(legend_name)
    x_track_d = x_diff_all[i,:,:]
    u_track_d = u_diff_all[i,:]
    j_track_d = j_diff_all[i,:]

    plt.subplot(PLOT_NUM, 1, 1)
    plt.plot(step, x_track_d[:,0])

    plt.subplot(PLOT_NUM, 1, 2)
    plt.plot(step, x_track_d[:,1])

    plt.subplot(PLOT_NUM, 1, 3)
    plt.plot(step, x_track_d[:,2])

    plt.subplot(PLOT_NUM, 1, 4)
    plt.plot(step, x_track_d[:,3])

    plt.subplot(PLOT_NUM, 1, 5)
    plt.plot(step, u_track_d)

    plt.subplot(PLOT_NUM, 1, 6)
    plt.plot(step, j_track_d)
    
for i in range(1,PLOT_NUM+1):
    plt.subplot(PLOT_NUM, 1, i)
    plt.legend(legend_list) 

figure_name = 'compare_checkpt_w_' + str(WEIGHT_GUIDANC) + '_'+ RESULT_NAME + '.png'
figure_path = os.path.join(RESULT_FOLDER, figure_name)
plt.savefig(figure_path)
print("save")

