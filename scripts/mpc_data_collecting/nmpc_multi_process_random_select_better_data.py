import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Manager, Array
import multiprocessing

############### Seetings ######################
# Attention: this py file can only set the initial range of position and theta, initial x_dot and theta_dot are always 0

# modify
SAVE_PATH =  "/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/NN/"
CONTROL_STEPS = 50
NUM_INITIAL_X = 5
NUM_INIYIAL_THETA = 6
NUM_NOISY_DATA =15
B_RANDOMINIGUESS_FOR_NOISE = True
INITIAL_GUESS_DECAY = 1
FIND_GLOBAL_NUM = 5

B_IFNOR_CAND_FINISHED = 1
PATH_X_CAN_NOR = os.path.join(SAVE_PATH, 'x_cand_nor.pt')
PATH_U_CAN_NOR = os.path.join(SAVE_PATH, 'u_cand_nor.pt')
PATH_J_CAN_NOR = os.path.join(SAVE_PATH, 'j_cand_nor.pt')


B_IFNOR_SELECT_FINISHED = 1
PATH_X_BEST_NOR = os.path.join(SAVE_PATH, 'x_best_nor.pt')
PATH_U_BEST_NOR = os.path.join(SAVE_PATH, 'u_best_nor.pt')
PATH_J_BEST_NOR = os.path.join(SAVE_PATH, 'j_best_nor.pt')

B_IFNOISE_CAND_FINISHED = 1
PATH_X_CAN_NOISE = os.path.join(SAVE_PATH, 'x_cand_noise.pt')
PATH_U_CAN_NOISE = os.path.join(SAVE_PATH, 'u_cand_noise.pt')
PATH_J_CAN_NOISE = os.path.join(SAVE_PATH, 'j_cand_noise.pt')

B_IFNOISE_SELECT_FINISHED = 0
PATH_X_BEST_NOISE = os.path.join(SAVE_PATH, 'x_best_noise.pt')
PATH_U_BEST_NOISE = os.path.join(SAVE_PATH, 'u_best_noise.pt')
PATH_J_BEST_NOISE = os.path.join(SAVE_PATH, 'j_best_noise.pt')


# data (x,u) collecting (saved in PT file)
NUM_GROUP_INI_GUESS = FIND_GLOBAL_NUM*NUM_INITIAL_X*NUM_INIYIAL_THETA

SIZE_FINALNOR_CANDIDATE_DATA = NUM_GROUP_INI_GUESS*(CONTROL_STEPS)
SIZE_FINALNOR_DATA_SELECT = int(SIZE_FINALNOR_CANDIDATE_DATA/FIND_GLOBAL_NUM)
SIZE_FINALNOISE_CANDIDATE_DATA = NUM_INITIAL_X*NUM_INIYIAL_THETA*(CONTROL_STEPS)*NUM_NOISY_DATA
SIZE_NOISE_SELECT_EACH_STATE = int(NUM_NOISY_DATA/FIND_GLOBAL_NUM)
SIZE_FINALNOISE_DATA_SELECT = SIZE_FINALNOR_DATA_SELECT * SIZE_NOISE_SELECT_EACH_STATE



# Random range for initialguess
MAX_U_INIGUESS = 4500
MIN_U_INIGUESS = -4500

# data range
POS_MIN = -3
POS_MAX = 3
POSITION_INITIAL_RANGE = np.linspace(POS_MIN,POS_MAX,NUM_INITIAL_X) 

THETA_MIN = 1.8
THETA_MAX = 4.4
THETA_INITIAL_RANGE = np.linspace(THETA_MIN,THETA_MAX,NUM_INIYIAL_THETA) 

# number of noisy data for each state

NOISE_MEAN = 0
NOISE_SD = 0.15
CONTROLSTEP_X_NUMNOISY = CONTROL_STEPS*NUM_NOISY_DATA

HOR = 64 # mpc prediction horizon

SAVE_EACH_GROUP = 0


# initial guess
INITIAL_GUESS_NUM = 1

IDX_X_INI = 0
IDX_THETA_INI = 1
IDX_THETA = 2
IDX_THETA_RED = 4

# trainind data files name
filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INIYIAL_THETA)+'_noise_'+str(SIZE_NOISE_SELECT_EACH_STATE)+'_step_'+str(CONTROL_STEPS)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx

#np.random.seed(42)

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
    
def EulerForwardCartpole_virtual(dt, x,u) -> np.array:
    xdot = np.array([
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        x[3],        # thetadot
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u
          )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    ])
    return x + xdot * dt

def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi


def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, P_cost:np.array, ts: float, opts_setting, nMaxGuess:int = 3 ):
    retries = 0
    
    while retries < nMaxGuess:
        try:
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
            for k in range(0,horizon-1):
                x_next = system_update(system_dynamic,ts,X_pre[:, k],U_pre[:, k])
                optimizer_normal.subject_to(X_pre[:, k + 1] == x_next)
                cost += Q_cost[0,0]*X_pre[0, k+1]**2 + Q_cost[1,1]*X_pre[1, k+1]**2 + Q_cost[2,2]*X_pre[2, k+1]**2 + Q_cost[3,3]*X_pre[3, k+1]**2 + Q_cost[4,4]*X_pre[4, k+1]**2 + R_cost * U_pre[:, k]**2

            # terminal cost
            x_terminal = system_update(system_dynamic,ts,X_pre[:, horizon-1],U_pre[:, horizon-1])
            optimizer_normal.subject_to(X_pre[:, horizon] == x_terminal)
            cost += P_cost[0,0]*X_pre[0, horizon]**2 + P_cost[1,1]*X_pre[1, horizon]**2 + P_cost[2,2]*X_pre[2, horizon]**2 + P_cost[3,3]*X_pre[3, horizon]**2 + P_cost[4,4]*X_pre[4, horizon]**2 + R_cost * U_pre[:, horizon-1]**2

            optimizer_normal.minimize(cost)
            optimizer_normal.solver('ipopt',opts_setting)
            sol = optimizer_normal.solve()
            X_sol = sol.value(X_pre)
            U_sol = sol.value(U_pre)
            Cost_sol = sol.value(cost)
            break
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            if retries<1:
                if x0[2] <= np.pi: 
                    initial_guess_x = -5
                    initial_guess_u = -1000
                else:
                    initial_guess_x = 5
                    initial_guess_u = 1000
            else:
                initial_guess_u, initial_guess_x = GenerateRandomInitialGuess(MIN_U_INIGUESS, MAX_U_INIGUESS)
            print("retries:",retries,", modified initial guess as u", initial_guess_u)
            
            retries += 1
            
        if retries >= nMaxGuess:
            print("MPC solve error, cannot find solution")
    return X_sol, U_sol, Cost_sol


# Opt
opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}



###################multi-process######################################################################################
def MPC_NormalData_Process(x0, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_result_normal, j_result_normal, x_result_normal, idx_control_step=0) -> float:
    u_for_normal_x = np.zeros(HOR)
    
    X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, P, TS, opts_setting)
    u_for_normal_x = U_sol.reshape(HOR,1)
    j_for_normal_x = np.array(Cost_sol)
    # save normal x,u,j data in 0th step 
    idx_0step_normal_data = idx_group_of_control_step*(CONTROL_STEPS) + idx_control_step
    
    
    # shared memory with manager list
    x_result_normal[idx_0step_normal_data] = x0.tolist()
    u_result_normal[idx_0step_normal_data] = u_for_normal_x.tolist()
    j_result_normal[idx_0step_normal_data] = j_for_normal_x.tolist()
    
    # print normal
    print('-----------------------------------------normal result--------------------------------------------------------')
    print(f'(idx_ini_guess*num_datagroup+turn, control step) -- {idx_group_of_control_step, idx_control_step}')
    
    return U_sol[0], U_sol, X_sol

def GenerateNoiseCandForEachStep( nGuess, x0_normal, u0_normal, idx_group_of_control_step, u_noise_candidate, j_noise_candidate, x_noise_candidate, idx_control_step):
    
    nDataNumForOneStep = NUM_NOISY_DATA
    x_noise = np.zeros((nDataNumForOneStep, NUM_STATE))
    u_noise = np.zeros((nDataNumForOneStep, HOR, 1))
    j_noise = np.zeros((nDataNumForOneStep))
    for idx_Noiseth in range( SIZE_NOISE_SELECT_EACH_STATE ):
        if (idx_control_step == 0):
            noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = (1,2))
            noisy_state = x0_normal.numpy() + [noise[0,0], 0, noise[0,1],0,0]
        else:
            noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = NUM_STATE)
            noisy_state = x0_normal.numpy() + noise
            
        noisy_state[IDX_THETA_RED] = ThetaToRedTheta(noisy_state[IDX_THETA])
        
        for idx_iniGuess_forOneNoise in range(0,nGuess):
            # solve MPC with different ini guess for 1 noise    
            U_range = INITIAL_GUESS_DECAY*np.abs(u0_normal)
            u_ini_guess, x_ini_guess = GenerateRandomInitialGuess(-U_range, U_range)
            X_noise_sol, U_noisy_sol, Cost_noise_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, noisy_state, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, P, TS, opts_setting)
            
            # save data to specified idx
            nIdxSaveNoisyInOneStep = idx_Noiseth * nGuess + idx_iniGuess_forOneNoise
            x_noise[nIdxSaveNoisyInOneStep,:] = noisy_state
            u_noise[nIdxSaveNoisyInOneStep,:,:] = U_noisy_sol.reshape(1,HOR,1)
            j_noise[nIdxSaveNoisyInOneStep] = Cost_noise_sol


    # save noise x,u,j data in 0th step 
    idx_start_0step_nosie_data = idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA
    idx_end_0step_nosie_data = idx_start_0step_nosie_data + NUM_NOISY_DATA
    
    # shared memory with manager list
    x_noise_candidate[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = x_noise.tolist()
    u_noise_candidate[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = u_noise.tolist()
    j_noise_candidate[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = j_noise.tolist()


def RollOutMPCForSingleGroupANdIniGuess_Normal(x_ini_guess: float, u_ini_guess:float,idx_group_of_control_step:int,x0_state:np.array, 
                                               x_nor_candidate:torch.tensor, u_nor_candidate:torch.tensor, j_nor_candidate:torch.tensor):

    ################ generate data for 0th step ##########################################################
    try:
        # normal at x0
        u0, u_sol, x_sol = MPC_NormalData_Process(x0_state, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_nor_candidate, j_nor_candidate, x_nor_candidate)
        
        # update initial guess
        u_ini_guess, x_ini_guess =  concat_iniguess_from_MPCSol(u_sol, x_sol)
        
        ############################################## generate data for control step loop ##############################################
        # main mpc loop
        for idx_control_step in range(1, CONTROL_STEPS):
            #system dynamic update x 
            x0_next = EulerForwardCartpole_virtual(TS,x0_state,u0)
            
            ################################################# normal mpc loop to update state #################################################
            u0, u_sol, x_sol = MPC_NormalData_Process(x0_next, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_nor_candidate, j_nor_candidate, x_nor_candidate,idx_control_step)
            
            # update
            x0_state = x0_next
            u_ini_guess, x_ini_guess =  concat_iniguess_from_MPCSol(u_sol, x_sol)
    
    except Exception as e:
        print("wrong! dataset will fail!")
        print("fail group:", idx_group_of_control_step)
        print(f"Error: {e}")

def GenerateRandomInitialGuess(min_random, max_random):
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
    if u_ini_guess >=0:
        x_ini_guess = 5
    else:
        x_ini_guess = -5
    return u_ini_guess, x_ini_guess

def concat_iniguess_from_MPCSol(u_sol, x_sol):
    u_ini_guess = np.concatenate((u_sol[1:], u_sol[-1].reshape(1)), axis=0)
    x_ini_guess = np.concatenate((x_sol[:,1:], x_sol[:,-1].reshape(NUM_STATE,1)), axis=1)
    return u_ini_guess, x_ini_guess

def FindBestRollOutFromNormalCandidate(x_candidate:list, u_candidate:list, j_candidate:list, nGroup, nIniGuess, nControlStep):
    nGuess_ControlStep = nIniGuess * nControlStep
    Final_x = torch.zeros((SIZE_FINALNOR_DATA_SELECT, NUM_STATE))
    Final_u = torch.zeros((SIZE_FINALNOR_DATA_SELECT, HOR, 1))
    Final_j = torch.zeros((SIZE_FINALNOR_DATA_SELECT,))

    for i in range(nGroup):
        nCurStartIdx = i*nGuess_ControlStep
        nCurEndIdx = nCurStartIdx + (nControlStep-1)
        nBestIdx = nCurEndIdx
        for j in range(0,nIniGuess-1):
            if( j_candidate[nCurEndIdx+(j+1)*nControlStep] < j_candidate[nBestIdx] ):
                nBestIdx = nCurEndIdx+(j+1)*nControlStep
        nBestStartIdx = nBestIdx - nControlStep + 1
        
        nSaveStartIdx = i*nControlStep
        nSaveEndIdx = nSaveStartIdx + nControlStep
        Final_x[nSaveStartIdx:nSaveEndIdx,:] = torch.from_numpy(np.array(x_candidate[nBestStartIdx:nBestIdx+1]))
        Final_u[nSaveStartIdx:nSaveEndIdx,:,:] = torch.from_numpy(np.array(u_candidate[nBestStartIdx:nBestIdx+1]))
        Final_j[nSaveStartIdx:nSaveEndIdx] = torch.from_numpy(np.array(j_candidate[nBestStartIdx:nBestIdx+1]))
        
    # save
    torch.save(Final_x, PATH_X_BEST_NOR)
    torch.save(Final_u, PATH_U_BEST_NOR)
    torch.save(Final_j, PATH_J_BEST_NOR)
        
    return Final_u, Final_x, Final_j

def FindBestRollOutFromNoisyCandidate(x_candidate:list, u_candidate:list, j_candidate:list, nGroup, nIniGuess, nControlStep, nNumNoiseCandidate):
    nGuess_ControlStep = nNumNoiseCandidate * nControlStep
    nNumNoiseForEachStep = int(nNumNoiseCandidate/nIniGuess)
    Final_x = torch.zeros((SIZE_FINALNOISE_DATA_SELECT, NUM_STATE))
    Final_u = torch.zeros((SIZE_FINALNOISE_DATA_SELECT, HOR, 1))
    Final_j = torch.zeros((SIZE_FINALNOISE_DATA_SELECT,))
    
    for idx_group in range(nGroup):
        for idx_controlstep in range(nControlStep):
            for idx_Noise in range(nNumNoiseForEachStep):
                nCurIdx = idx_group * nGuess_ControlStep + idx_controlstep * nNumNoiseCandidate + idx_Noise * nIniGuess
                nBestIdx = nCurIdx
                for idx_NoiseCandidate in range(0,nIniGuess-1):
                    if( j_candidate[nCurIdx+idx_NoiseCandidate + 1] < j_candidate[nBestIdx] ):
                        nBestIdx = nCurIdx+idx_NoiseCandidate + 1
                nIdx_Final = idx_group * nControlStep * nNumNoiseForEachStep + idx_controlstep * nNumNoiseForEachStep + idx_Noise 
                Final_x[nIdx_Final,:] = torch.from_numpy(np.array(x_candidate[nBestIdx]))
                Final_u[nIdx_Final,:,:] = torch.from_numpy(np.array(u_candidate[nBestIdx]))
                Final_j[nIdx_Final] = torch.from_numpy(np.array(j_candidate[nBestIdx]))
                
    # save
    torch.save(Final_x, PATH_X_BEST_NOISE)
    torch.save(Final_u, PATH_U_BEST_NOISE)
    torch.save(Final_j, PATH_J_BEST_NOISE)
                
    return Final_u, Final_x, Final_j

    

######################################################################################################################
# ##### data collecting loop #####

x_nor_candidate_shape = (SIZE_FINALNOR_CANDIDATE_DATA,NUM_STATE)
u_nor_candidate_shape = (SIZE_FINALNOR_CANDIDATE_DATA,HOR,1)
j_nor_candidate_shape = (SIZE_FINALNOR_CANDIDATE_DATA)
x_normal_shape = (SIZE_FINALNOR_DATA_SELECT,NUM_STATE)
u_normal_shape = (SIZE_FINALNOR_DATA_SELECT,HOR,1)
j_normal_shape = (SIZE_FINALNOR_DATA_SELECT)
x_noise_candidate_shape = (SIZE_FINALNOISE_CANDIDATE_DATA,NUM_STATE)
u_noise_candidate_shape = (SIZE_FINALNOISE_CANDIDATE_DATA,HOR,1)
j_noise_candidate_shape = (SIZE_FINALNOISE_CANDIDATE_DATA)
x_noise_shape = (SIZE_FINALNOISE_DATA_SELECT,NUM_STATE)
u_noise_shape = (SIZE_FINALNOISE_DATA_SELECT,HOR,1)
j_noise_shape = (SIZE_FINALNOISE_DATA_SELECT)


def main():
    MAX_CORE_CPU = 24
    start_time = time.time()
    print("data save path = ",SAVE_PATH,"\n")
    print("start")
    with Manager() as manager:
        if B_IFNOR_CAND_FINISHED == 0:
            print("start prepare shared memory \n")
            # prepare normal data argument
            x_nor_candidate_shared_memory = manager.list([[0.0] * x_nor_candidate_shape[1]] * x_nor_candidate_shape[0])
            u_nor_candidate_shared_memory = manager.list([[[0.0] for _ in range(u_nor_candidate_shape[1])] for _ in range(u_nor_candidate_shape[0])])
            j_nor_candidate_shared_memory = manager.list([0.0] * j_nor_candidate_shape)
            
            argument_ForNormalCandidate = []
            for turn in range(num_datagroup):
                for idx_candidate_iniguess in range(FIND_GLOBAL_NUM):
                    # initial guess
                    u_ini_guess, x_ini_guess = GenerateRandomInitialGuess(MIN_U_INIGUESS, MAX_U_INIGUESS)
                    idx_group_of_control_step = turn * FIND_GLOBAL_NUM + idx_candidate_iniguess
                
                    #initial states
                    x_0 = rng0[turn,IDX_X_INI]
                    theta_0 = rng0[turn,IDX_THETA_INI]
                    theta_red_0 = ThetaToRedTheta(theta_0)
                    x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
                    
                    argument_ForNormalCandidate.append((x_ini_guess, u_ini_guess, idx_group_of_control_step, x0, 
                                                x_nor_candidate_shared_memory, u_nor_candidate_shared_memory, j_nor_candidate_shared_memory))
                    
            # generate normal data candidate
            print("start generate data \n")
            with Pool(processes=MAX_CORE_CPU) as pool:
                pool.starmap(RollOutMPCForSingleGroupANdIniGuess_Normal, argument_ForNormalCandidate)
                
            # save normal candidate
            x_cand_nor = torch.from_numpy(np.array(x_nor_candidate_shared_memory))
            u_cand_nor = torch.from_numpy(np.array(u_nor_candidate_shared_memory))
            j_cand_nor = torch.from_numpy(np.array(j_nor_candidate_shared_memory))
            torch.save(x_cand_nor, PATH_X_CAN_NOR)
            torch.save(u_cand_nor, PATH_U_CAN_NOR)
            torch.save(j_cand_nor, PATH_J_CAN_NOR)
            
            
        ######################################## step 2, select best data from normal #####################
        print("step 1 finished, get normal candidate data") 
        if (B_IFNOR_CAND_FINISHED == 1 and B_IFNOR_SELECT_FINISHED == 0):
            x_cand_nor = torch.load(PATH_X_CAN_NOR)
            x_nor_candidate_shared_memory = x_cand_nor.tolist()
            u_cand_nor = torch.load(PATH_U_CAN_NOR)
            u_nor_candidate_shared_memory = u_cand_nor.tolist()
            j_cand_nor = torch.load(PATH_J_CAN_NOR)
            j_nor_candidate_shared_memory = j_cand_nor.tolist()
        # find best normal data
        if B_IFNOR_SELECT_FINISHED == 0:
            u_nor_best, x_nor_best, j_nor_best = FindBestRollOutFromNormalCandidate(x_nor_candidate_shared_memory, u_nor_candidate_shared_memory, j_nor_candidate_shared_memory, 
                                                                            num_datagroup, FIND_GLOBAL_NUM, CONTROL_STEPS)
        
        ####################################### step 3, generate candidate noisy data #########################
        print("step 2 finished, get normal select data") 
        if B_IFNOR_SELECT_FINISHED == 1:
            x_nor_best = torch.load(PATH_X_BEST_NOR)
            u_nor_best = torch.load(PATH_U_BEST_NOR)
            j_nor_best = torch.load(PATH_J_BEST_NOR)
        
        if B_IFNOISE_CAND_FINISHED == 0:
            # prepare noisy data argument
            x_noise_shared_memory = manager.list([[0.0] * x_noise_candidate_shape[1]] * x_noise_candidate_shape[0])
            u_noise_shared_memory = manager.list([[[0.0] for _ in range(u_noise_candidate_shape[1])] for _ in range(u_noise_candidate_shape[0])])
            j_noise_shared_memory = manager.list([0.0] * j_noise_candidate_shape)
            
            argument_ForNoisyCandidate = []
            for turn in range(num_datagroup):
                for idx_controlstep in range(CONTROL_STEPS):
                    nIdxSinglePointFromNormal = turn*CONTROL_STEPS + idx_controlstep
                    x0_normal = x_nor_best[nIdxSinglePointFromNormal,:]
                    u0_normal = u_nor_best[nIdxSinglePointFromNormal,0,0]
                    idx_group_of_control_step = turn
                    
                    argument_ForNoisyCandidate.append((FIND_GLOBAL_NUM, x0_normal, u0_normal, idx_group_of_control_step, u_noise_shared_memory, j_noise_shared_memory, x_noise_shared_memory, idx_controlstep))
            
            # generate noisy data candidate
            with Pool(processes=MAX_CORE_CPU) as pool:
                pool.starmap(GenerateNoiseCandForEachStep, argument_ForNoisyCandidate)
            
            # save noise candidate
            x_cand_noise = torch.from_numpy(np.array(x_noise_shared_memory))
            u_cand_noise = torch.from_numpy(np.array(u_noise_shared_memory))
            j_cand_noise = torch.from_numpy(np.array(j_noise_shared_memory))
            torch.save(x_cand_noise, PATH_X_CAN_NOISE)
            torch.save(u_cand_noise, PATH_U_CAN_NOISE)
            torch.save(j_cand_noise, PATH_J_CAN_NOISE)
            
        ############################################### step 4. select best noise data ############################################################
        print("step 3 finished, get noise candidate data") 
        if (B_IFNOISE_CAND_FINISHED == 1 and B_IFNOISE_SELECT_FINISHED == 0):
            x_cand_noise = torch.load(PATH_X_CAN_NOISE)
            x_noise_shared_memory = x_cand_noise.tolist()
            u_cand_noise = torch.load(PATH_U_CAN_NOISE)
            u_noise_shared_memory = u_cand_noise.tolist()
            j_cand_noise = torch.load(PATH_J_CAN_NOISE)
            j_noise_shared_memory = j_cand_noise.tolist()
        # find best noisy data
        u_noise_best, x_noise_bset, j_noise_best = FindBestRollOutFromNoisyCandidate(x_noise_shared_memory, u_noise_shared_memory, j_noise_shared_memory, 
                                                                                num_datagroup, FIND_GLOBAL_NUM, CONTROL_STEPS, NUM_NOISY_DATA)
                
        if (B_IFNOISE_SELECT_FINISHED==1):
            x_noise_bset = torch.load(PATH_X_BEST_NOISE)
            u_noise_best = torch.load(PATH_U_BEST_NOISE)
            j_noise_best = torch.load(PATH_J_BEST_NOISE)

        # show the normal
        print(f'normal u -- {u_nor_best[0,:,0]}')
        print(f'normal x -- {x_nor_best[0,:]}')

        ##### data combing #####
        # u combine u_normal + u_noisy
        u_training_data = torch.cat((u_nor_best, u_noise_best), dim=0)
        print(f'u_training_data -- {u_training_data.size()}')

        # x0 combine x_normal + x_noisy
        x0_conditioning_data = torch.cat((x_nor_best, x_noise_bset), dim=0)
        print(f'x0_conditioning_data -- {x0_conditioning_data.size()}')

        # J combine j_normal + j_noisy
        J_training_data = torch.cat((j_nor_best, j_noise_best), dim=0)

        # data saving
        torch.save(u_training_data, os.path.join(SAVE_PATH, U_DATA_NAME))
        torch.save(x0_conditioning_data, os.path.join(SAVE_PATH, X0_CONDITION_DATA_NAME))
        torch.save(J_training_data, os.path.join(SAVE_PATH, J_DATA_NAME))

    end_time = time.time()

    duration = end_time - start_time
    print(f"Time taken for generating data: {duration} seconds")
    print("generate data finish!")




if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()