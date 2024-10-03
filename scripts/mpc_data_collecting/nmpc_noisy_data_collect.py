import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt
import time

############### Seetings ######################
# Attention: this py file can only set the initial range of position and theta, initial x_dot and theta_dot are always 0

# data saving folder
SAVE_PATH = "/MPC_DynamicSys/sharedVol/train_data/nmpc/normal"

# control steps
CONTROL_STEPS = 80

# data range
NUM_INITIAL_X = 5
# POSITION_INITIAL_RANGE = np.linspace(-0.5,0.5,NUM_INITIAL_X) 

POSITION_INITIAL_RANGE = np.linspace(-0.5,-0.05555556,NUM_INITIAL_X) 


NUM_INIYIAL_THETA = 20
THETA_INITIAL_RANGE = np.linspace(3*np.pi/4,5*np.pi/4,NUM_INIYIAL_THETA) 

# number of noisy data for each state
NUM_NOISY_DATA =20
NOISE_MEAN = 0
NOISE_SD = 0.15
CONTROLSTEP_X_NUMNOISY = CONTROL_STEPS*NUM_NOISY_DATA

HOR = 64 # mpc prediction horizon


# initial guess
INITIAL_GUESS_NUM = 2
initial_guess_x = [5, 0]
initial_guess_u = [1000, -10000]

# save data round to 4 digit
ROUND_DIG = 6

IDX_X_INI = 0
IDX_THETA_INI = 1
IDX_THETA = 2
IDX_THETA_RED = 4

# trainind data files name
filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INIYIAL_THETA)+'_noise_'+str(NUM_NOISY_DATA)+'_step_'+str(CONTROL_STEPS)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx

np.random.seed(42)

############### Dynamics Define ######################
# dynamic parameter
M_CART = 2.0
M_POLE = 1.0
M_TOTAL = M_CART + M_POLE
L_POLE = 1.0
MPLP = M_POLE*L_POLE
G = 9.81
MPG = M_POLE*G
MTG = M_TOTAL*G
MTLP = M_TOTAL*G
PI_2 = 2*np.pi
PI_UNDER_2 = 2/np.pi
PI_UNDER_1 = 1/np.pi

def EulerForwardCartpole_virtual_Casadi(dynamic_update_virtual_Casadi, dt, x,u) -> ca.vertcat:
    xdot = dynamic_update_virtual_Casadi(x,u)
    return x + xdot * dt

def dynamic_update_virtual_Casadi(x, u) -> ca.vertcat:
    # Return the derivative of the state
    # u is 1x1 array, covert to scalar by u[0] 
        
    return ca.vertcat(
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u[0] 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        x[3],        # thetadot
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u[0]
          )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    )

def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi


def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, ts: float, opts_setting ):
    # casadi_Opti
    optimizer_normal = ca.Opti()
    
    ##### normal mpc #####  
    # x and u mpc prediction along N
    X_pre = optimizer_normal.variable(num_state, horizon + 1) 
    U_pre = optimizer_normal.variable(1, horizon)
    # set intial guess
    optimizer_normal.set_initial(X_pre, initial_guess_x)
    optimizer_normal.set_initial(U_pre, initial_guess_u)

    optimizer_normal.subject_to(X_pre[:, 0] == x0)  # starting state

    # cost 
    cost = 0

    # initial cost
    cost += Q_cost[0,0]*X_pre[0, 0]**2 + Q_cost[1,1]*X_pre[1, 0]**2 + Q_cost[2,2]*X_pre[2, 0]**2 + Q_cost[3,3]*X_pre[3, 0]**2 + Q_cost[4,4]*X_pre[4, 0]**2

    # state cost
    for k in range(0,HOR-1):
        x_next = system_update(system_dynamic,ts,X_pre[:, k],U_pre[:, k])
        optimizer_normal.subject_to(X_pre[:, k + 1] == x_next)
        cost += Q_cost[0,0]*X_pre[0, k+1]**2 + Q_cost[1,1]*X_pre[1, k+1]**2 + Q_cost[2,2]*X_pre[2, k+1]**2 + Q_cost[3,3]*X_pre[3, k+1]**2 + Q_cost[4,4]*X_pre[4, k+1]**2 + R_cost * U_pre[:, k]**2

    # terminal cost
    x_terminal = system_update(system_dynamic,ts,X_pre[:, horizon-1],U_pre[:, horizon-1])
    optimizer_normal.subject_to(X_pre[:, horizon] == x_terminal)
    cost += P[0,0]*X_pre[0, HOR]**2 + P[1,1]*X_pre[1, HOR]**2 + P[2,2]*X_pre[2, HOR]**2 + P[3,3]*X_pre[3, HOR]**2 + P[4,4]*X_pre[4, HOR]**2 + R_cost * U_pre[:, HOR-1]**2

    optimizer_normal.minimize(cost)
    optimizer_normal.solver('ipopt',opts_setting)
    sol = optimizer_normal.solve()
    X_sol = sol.value(X_pre)
    U_sol = sol.value(U_pre)
    Cost_sol = sol.value(cost)
    return X_sol, U_sol, Cost_sol


############# MPC #####################

# mpc parameters
NUM_STATE = 5
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
R = 0.001
P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])


TS = 0.01

# Define the initial states range
rng_x = POSITION_INITIAL_RANGE 
rng_theta = THETA_INITIAL_RANGE 
rng0 = []
for idx_noisy in rng_x:
    for n in rng_theta:
        rng0.append([idx_noisy,n])
rng0 = np.array(rng0)
num_datagroup = len(rng0)
print(f'rng0 -- {rng0.shape}')

# ##### data collecting loop #####

# data set for each turn
x_track = np.zeros((NUM_STATE, (CONTROL_STEPS+1)))
u_track = np.zeros((1, CONTROL_STEPS))

# data (x,u) collecting (saved in PT file)
x_all_tensor = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*(CONTROL_STEPS),NUM_STATE) # x0: 64000*5
u_all_tensor = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*(CONTROL_STEPS),HOR,1) # u: 64000*40*1
J_all_tensor = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*(CONTROL_STEPS)) # J: 64000*5

# all noisy data
x_all_noisy = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*CONTROLSTEP_X_NUMNOISY, NUM_STATE) # 1280000*5
u_all_noisy = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*CONTROLSTEP_X_NUMNOISY, HOR, 1) # 1280000*40*1
J_all_noisy = torch.zeros(INITIAL_GUESS_NUM*num_datagroup*CONTROLSTEP_X_NUMNOISY) # J: 64000*5

noisey_x_array = np.zeros((NUM_NOISY_DATA, NUM_STATE))    # NUM_NOISY_DATA x NUM_STATE

# Opt
opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

start_time = time.time()

for idx_ini_guess in range(0, INITIAL_GUESS_NUM): 
    for turn in range(0,num_datagroup):
        x_ini_guess = initial_guess_x[idx_ini_guess]
        u_ini_guess = initial_guess_u[idx_ini_guess]
        idx_group_of_control_step = idx_ini_guess*num_datagroup+turn
        #initial states
        x_0 = rng0[turn,IDX_X_INI]
        theta_0 = rng0[turn,IDX_THETA_INI]
        theta_red_0 = ThetaToRedTheta(theta_0)
        x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
        print(f'x0-- {x0}')
        x_track[:,0] = x0

        ################ generate noise data for initial state ##########################################################
        # noisy at x0
        for n in range(0,NUM_NOISY_DATA):
            noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = (1,2))
            noisy_state = x0 + [noise[0,0], 0, noise[0,1],0,0]
            noisy_state[IDX_THETA_RED] = ThetaToRedTheta(noisy_state[IDX_THETA])
            noisey_x_array [n,:] = noisy_state 

        # save the initail noisy x group
        x_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY:idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY+NUM_NOISY_DATA,:] = torch.tensor(noisey_x_array)

        # main mpc loop
        for idx_control_step in range(0, CONTROL_STEPS):

            ################################## calculating u for noisy data based on x generated by last round ##################################
            u_for_noisy_x = np.zeros((NUM_NOISY_DATA, HOR))

            for idx_noisy in range(0,len(noisey_x_array)):
                X_noise_sol, U_noisy_sol, Cost_noise_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, noisey_x_array[idx_noisy,:], x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, TS, opts_setting)
                
                for v in range(0,HOR):
                    u_for_noisy_x[idx_noisy,v] = U_noisy_sol[v]

            # save noisy u 
            u_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA : idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA + NUM_NOISY_DATA,:,0] = torch.tensor(u_for_noisy_x)
            J_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA : idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA + NUM_NOISY_DATA] = torch.tensor(Cost_noise_sol)

            ################################################# normal mpc loop to update state #################################################
            
            X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, TS, opts_setting)
            
            # select the first updated states as new starting state ans save in the x_track
            x0 = X_sol[:,1]
           
            x_track[:,idx_control_step+1] = x0
            u_track[:,idx_control_step] = U_sol[0]
            print('-----------------------------------------normal result--------------------------------------------------------')
            print(f'initial_guess, turn, control_step -- {idx_ini_guess, turn, idx_control_step}')
            print(f'u_sol-- {u_track[:,idx_control_step]}')
            print(f'x0_new-- {x_track[:,idx_control_step+1]}')
            print(f'cost-- {Cost_sol}')
            # save control inputs in tensor
            u_reshape = U_sol.reshape(1,HOR)
            u_tensor = torch.tensor(u_reshape)
            u_all_tensor[idx_group_of_control_step*(CONTROL_STEPS)+idx_control_step,:,0] = u_tensor
            J_all_tensor[idx_group_of_control_step*(CONTROL_STEPS)+idx_control_step] = torch.tensor(Cost_sol)

            ################################## generate noise state in next step  ##################################
            noisey_x_array = np.zeros((NUM_NOISY_DATA, NUM_STATE))

            for n in range(0,NUM_NOISY_DATA):
                noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = x0.shape[0])
                noisy_state = x0 + noise
                noisy_state[IDX_THETA_RED] = ThetaToRedTheta(noisy_state[IDX_THETA])
                noisey_x_array [n,:] = noisy_state 

                
            # save the initail noisy x group (except the last x0)
            if idx_control_step != CONTROL_STEPS - 1:
                x_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + (idx_control_step+1)*NUM_NOISY_DATA : idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + (idx_control_step+1)*NUM_NOISY_DATA + NUM_NOISY_DATA,:] = torch.tensor(noisey_x_array)


        # save states in tensor
        x_save = x_track[:,:-1]
        x_reshape = np.transpose(x_save)
        x_tensor = torch.tensor(x_reshape)
        x_all_tensor[idx_group_of_control_step*(CONTROL_STEPS):idx_group_of_control_step*(CONTROL_STEPS)+(CONTROL_STEPS),:] = x_tensor
        
        #save data each group into folder seperately
        x_normal_tensor_single_Group_singel_guess = x_tensor
        u_normal_tensor_single_Group_singel_guess = u_all_tensor[idx_group_of_control_step*(CONTROL_STEPS):(idx_group_of_control_step+1)*(CONTROL_STEPS),:,:]
        J_normal_tensor_single_Group_singel_guess = J_all_tensor[idx_group_of_control_step*(CONTROL_STEPS):(idx_group_of_control_step+1)*(CONTROL_STEPS)]
        
        x_noise_tensor_single_Group_singel_guess = x_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY:(idx_group_of_control_step+1)*CONTROLSTEP_X_NUMNOISY,:]
        u_noise_tensor_single_Group_singel_guess = u_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY:(idx_group_of_control_step+1)*CONTROLSTEP_X_NUMNOISY,:,:]
        J_noise_tensor_single_Group_singel_guess = J_all_noisy[idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY:(idx_group_of_control_step+1)*CONTROLSTEP_X_NUMNOISY]
        
        u_data_group = torch.cat((u_normal_tensor_single_Group_singel_guess, u_noise_tensor_single_Group_singel_guess), dim=0)
        x_data_group = torch.cat((x_normal_tensor_single_Group_singel_guess, x_noise_tensor_single_Group_singel_guess), dim=0)
        J_data_group = torch.cat((J_normal_tensor_single_Group_singel_guess, J_noise_tensor_single_Group_singel_guess), dim=0)
        
        GroupFileName = 'guess_' + str(idx_ini_guess) + '_ini_' + str(turn) + '_'
        UGroupFileName = GroupFileName + U_DATA_NAME
        XGroupFileName = GroupFileName + X0_CONDITION_DATA_NAME
        JGroupFileName = GroupFileName + J_DATA_NAME
        
        torch.save(u_data_group, os.path.join(SAVE_PATH, UGroupFileName))
        torch.save(x_data_group, os.path.join(SAVE_PATH, XGroupFileName))
        torch.save(J_data_group, os.path.join(SAVE_PATH, JGroupFileName))


# show the first saved u and x0
print(f'first_u -- {u_all_tensor[0,:,0]}')
print(f'first_x0 -- {x_all_tensor[0,:]}')

##### data combing #####
# u combine u_normal + u_noisy
u_training_data = torch.cat((u_all_tensor, u_all_noisy), dim=0)
print(f'u_training_data -- {u_training_data.size()}')

# x0 combine x_normal + x_noisy
x0_conditioning_data = torch.cat((x_all_tensor, x_all_noisy), dim=0)
print(f'x0_conditioning_data -- {x0_conditioning_data.size()}')

# J combine j_normal + j_noisy
J_training_data = torch.cat((J_all_tensor, J_all_noisy), dim=0)

# data saving
torch.save(u_training_data, os.path.join(SAVE_PATH, U_DATA_NAME))
torch.save(x0_conditioning_data, os.path.join(SAVE_PATH, X0_CONDITION_DATA_NAME))
torch.save(J_training_data, os.path.join(SAVE_PATH, J_DATA_NAME))

end_time = time.time()

duration = end_time - start_time
print(f"Time taken for generating data: {duration} seconds")