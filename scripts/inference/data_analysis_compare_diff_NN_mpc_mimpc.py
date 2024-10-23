from abc import ABC, abstractmethod
import torch.nn as nn
import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS, diffusion_model_base
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.datasets.nmpc_cart_pole_data import NMPC_Dataset
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params

from multiprocessing import Pool
import multiprocessing

# modify
DATASET = 'NMPC_UJ_Dataset'
J_NORMALIZER = 'GaussianNormalizer' #GaussianNormalizer, LimitsNormalizer, LogMinMaxNormalizer, LogZScoreNormalizer, OnlyLogNormalizer
UX_NORMALIZER = 'GaussianNormalizer' 
MODEL_FOLDER = 'nmpc_batch_256_random6000_zscore_xu_logzscore_j_selectdata_float64'
DATA_LOAD_FOLER = 'NN'

B_PEDICT_J_BYMODEL = 0
# path
RESULT_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving/'
MODEL_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+MODEL_FOLDER
DATA_LOAD_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/'+DATA_LOAD_FOLER

OPT_SETTING = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

INITILA_X = 5 #10
INITIAL_THETA = 6 #15
CONTROL_STEP = 50
NOISE_NUM = 3 #15
HOR = 64

NUM_FIND_GLOBAL = 5
# fix_random_seed(40)

# Data Name Setting
filename_idx = '_ini_'+str(INITILA_X)+'x'+str(INITIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
X0_CONDITION_DATA_NAME = 'x0' + filename_idx
U_DATA_FILENAME = 'u' + filename_idx
J_DATA_FILENAME = 'j' + filename_idx


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

# Sample parameters
TS = 0.01
X0_RANGE = np.array([-0.5, 0.5])
THETA0_RANGE = np.array([3*np.pi/4, 5*np.pi/4])

NUM_STATE = 5

DEBUG = 1

WEIGHT_GUIDANC = 0.01 # non-conditioning weight
X0_IDX = 150 # range:[0,199] 20*20 data 



RESULT_NAME = MODEL_FOLDER
RESULT_FOLDER = os.path.join(RESULT_SAVED_PATH, RESULT_NAME)


class ControllerManager:
    def __init__(self, controller: MPCBasedController):
        # Initially set the strategy
        self._strategy = controller
        
    def set_strategy(self, controller: MPCBasedController):
        # Dynamically change the strategy
        self._strategy = controller

    def do_something(self, data):
        # Delegate behavior to the strategy
        self._strategy.execute(data)

class MPCBasedController(ABC):
    @abstractmethod
    def SolveUandGetCost(self, ):
        pass


class AMPCNet_Inference(nn.Module):
    def __init__(self, input_size, output_size):
        super(AMPCNet_Inference, self).__init__()
        # Define the hidden layers and output layer
        self.hidden1 = nn.Linear(input_size, 2)  # First hidden layer with 2 neurons
        self.hidden2 = nn.Linear(2, 50)          # Second hidden layer with 50 neurons
        self.hidden3 = nn.Linear(50, 50)         # Third hidden layer with 50 neurons
        self.output = nn.Linear(50, output_size) # Output layer

    def forward(self, x):
        # print(x.dtype)  # Check the data type of the input tensor
        # print(self.hidden1.weight.dtype)  # Check the data type of the Linear layer's weights
        # Forward pass through the network with the specified activations
        x = x.to(torch.float32) 
        x = torch.tanh(self.hidden1(x))          # Tanh activation for first hidden layer
        x = torch.tanh(self.hidden2(x))          # Tanh activation for second hidden layer
        x = torch.tanh(self.hidden3(x))          # Tanh activation for third hidden layer
        x = self.output(x)                       # Linear activation (no activation function) for the output layer

        # reshape the output
        x = x.view(1, 8, 1) # 512(batch size)*8*1

        return x
    
    


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

def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, P_cost:np.array, ts: float, opts_setting ):
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
    return X_sol, U_sol, Cost_sol

def calMPCCost(Q,R,P,u_hor:torch.tensor,x0:np.array, ModelUpdate_func, dt) -> float:
    # u 1x64x1, x0: ,state
    num_state = x0.shape[0]
    num_u = u_hor.size(0)
    num_hor = u_hor.size(1)
    cost = 0
    
    # initial cost
    for i in range(num_state):
        cost = cost + Q[i][i] * x0[i] ** 2
    
    for i in range(num_u):
        cost = cost + R[i][i] * u_hor[i][0][0] ** 2
        
    x_cur = x0
    u_cur = u_hor[0][0][0]
    IdxLastU = num_hor-1
    # stage cost
    for i in range(1,IdxLastU):
        xnext = ModelUpdate_func(dt, x_cur, u_cur)
        unext = u_hor[:,i,0]
        for j in range(1,num_state):
            cost = cost + Q[j][j] * xnext[j] ** 2
        for j in range(num_u):
            cost = cost + R[j][j] * unext ** 2
        # update
        u_cur = unext
        x_cur = xnext
        
        
    #final cost
    for i in range(num_state):
        cost = cost + P[i][i] * xnext[i] ** 2
            
            
    return cost

def PickBestDiffResult( candidate_list:list ) -> torch.tensor:
    # candidate_list [[cost1, u_hor1],[cost2, u_hor2],...]
    num_candidate = len(candidate_list)
    best_idx = 0
    for i in range(num_candidate-1):
        if( candidate_list[i+1][0] < candidate_list[best_idx][0] ):
            best_idx = i+1
    return candidate_list[best_idx][0], candidate_list[best_idx][1]

def runMPC(x0_test_red:np.array, result_dir, Q, R, P, Hor, initial_guess_x, initial_guess_u):
    # ##### MPC setting #####
    opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    idx_pos_neg = 0
    # data save
    x_track_mpc = np.zeros((CONTROL_STEP+1, NUM_STATE))
    u_track_mpc = np.zeros((CONTROL_STEP))
    j_track_mpc = np.zeros((CONTROL_STEP))
    x_track_mpc[0,:] = x0_test_red

    #save the initial states
    print(f'MPC start------ x0-- {x0_test_red}')
    x_cur = x0_test_red
    ############# control loop ##################
    for i in range(0, CONTROL_STEP):
        X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x_cur, initial_guess_x, initial_guess_u, NUM_STATE, Hor, Q, R, P, TS, opts_setting)
        u_first = U_sol[0]
        x_next = EulerForwardCartpole_virtual(TS, x_cur, u_first)

        # update
        x_cur = x_next
        x_track_mpc[i+1,:] = x_next
        u_track_mpc[i] = u_first
        j_track_mpc[i] = Cost_sol
        
        print("step:",i,'\n')
        print("xnext=",x_next,'\n')
        print("control input=",u_first,'\n')

    x_track_mpc = x_track_mpc[0:-1,:] # make it dimension same with u, j
    
    os.makedirs(result_dir, exist_ok=True)

    
    if initial_guess_u >= 0:
        filename_iniguess = '0'
    else:
        filename_iniguess = '1'
    filename_final_iniguess = 'iniguess_'+filename_iniguess + '_'
    
    mpc_u = filename_final_iniguess + 'u_mpc.npy'
    mpc_u_path = os.path.join(result_dir, mpc_u)
    np.save(mpc_u_path, u_track_mpc)
    
    mpc_x = filename_final_iniguess + 'x_mpc.npy'
    mpc_x_path = os.path.join(result_dir, mpc_x)
    np.save(mpc_x_path, x_track_mpc)
    
    mpc_j = filename_final_iniguess + 'j_mpc.npy'
    mpc_j_path = os.path.join(result_dir, mpc_j)
    np.save(mpc_j_path, j_track_mpc)


def GenerateRandomInitialGuess(min_random, max_random):
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
    if u_ini_guess >=0:
        x_ini_guess = 5
    else:
        x_ini_guess = -5
    return u_ini_guess, x_ini_guess
    
@single_experiment_yaml
def InferenceBy_one_method_single_IniState(
    #########################################################################################
    # Common
    nIdxIniState: int,
    
    Diff_result: list,
    
    NN_result: list,
    
    MPC_result: list,
    
    MPC_multiguess_result: list,
    
    nMethodSelect: int,
    
    x0_red: np.array,

    x0_clean: np.array,
    
    Q: np.array,
    
    R: np.array,
    
    P:np.array,
    
    Hor: int,

    device: str,
    
    # Diffusion
    Diff_model: diffusion_model_base.GaussianDiffusionModel,
    
    Diff_dataset: NMPC_Dataset,
    
    # NN
    NN_model: AMPCNet_Inference,
    
    NN_dataset: NMPC_Dataset,
    
    # MPC
    initial_Uguess_range: np.array
):
    x_cur = x0_red
    x_cur_clean = x0_clean
    # control loop
    for i in range(CONTROL_STEP):
        if(nMethodSelect == 1): #Diff
            FindGlobal_list = []
            for j in range(NUM_FIND_GLOBAL):
                tensor = torch.randn(1, Hor, 1)  # Create a 1x64x1 tensor
                FindGlobal_list.append([0, tensor])
            x0_tensor = torch.tensor(x_cur_clean).to(device)
            x0_strd = Diff_dataset.normalize_condition(x0_tensor)
            
            for j in range(NUM_FIND_GLOBAL):
                with torch.no_grad():
                    with TimerCUDA() as timer_model_sampling:
                        u_normalized_iters = Diff_model.run_CFG(
                            x0_strd, None, WEIGHT_GUIDANC,
                            n_samples=1, horizon=Hor,
                            return_chain=True,
                            sample_fn=ddpm_cart_pole_sample_fn,
                            n_diffusion_steps_without_noise=25,
                        )
                print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')

                ########
                u_iters = Diff_dataset.unnormalize_states(u_normalized_iters[:,:,0:Hor,:])
                u_final_iter_candidate = u_iters[-1].cpu()
                             
                FindGlobal_list[j][0] = calMPCCost(Q,R,P,u_final_iter_candidate,x_cur, EulerForwardCartpole_virtual, TS)
                FindGlobal_list[j][1] = u_final_iter_candidate
                    
            Cost, u_best_cand = PickBestDiffResult(FindGlobal_list)
            Diff_result[nIdxIniState][i] = Cost
            u_first = u_best_cand[0,0,0]
        elif(nMethodSelect == 2): #NN
            x0_tensor = torch.tensor(x_cur_clean).to(device)
            x0_strd = NN_dataset.normalize_condition(x0_tensor)
            with torch.no_grad():
                with TimerCUDA() as t_NN_sampling:
                    u_normalized = NN_model(x0_strd)
                    inputs_final = NN_dataset.unnormalize_states(u_normalized)
            inputs_final = inputs_final[0,:,0].cpu()
            Cost = calMPCCost(Q,R,P,inputs_final,x_cur, EulerForwardCartpole_virtual, TS)
            NN_result[nIdxIniState][i] = Cost
            u_first = inputs_final[0]
                    
        elif(nMethodSelect == 3): #MPC
            u_ini_guess, x_ini_guess = GenerateRandomInitialGuess(initial_Uguess_range[0], initial_Uguess_range[1])
            X_sol, U_sol, Cost = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x_cur, x_ini_guess, u_ini_guess, NUM_STATE, Hor, Q, R, P, TS, OPT_SETTING)
            MPC_result[nIdxIniState][i] = Cost
            u_first = U_sol[0]
        elif(nMethodSelect == 4): #MPC multiguess
            FindGlobal_list = []
            for j in range(NUM_FIND_GLOBAL):
                array = np.zeros(1, Hor)  # Create a 1x64x1 tensor
                FindGlobal_list.append([0, array])
            for j in range(NUM_FIND_GLOBAL):
                u_ini_guess, x_ini_guess = GenerateRandomInitialGuess(initial_Uguess_range[0], initial_Uguess_range[1])
                X_sol, U_sol, Cost = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x_cur, x_ini_guess, u_ini_guess, NUM_STATE, Hor, Q, R, P, TS, OPT_SETTING)
                FindGlobal_list[j][0] = Cost
                FindGlobal_list[j][1] = U_sol
            Cost, u_best_cand = PickBestDiffResult(FindGlobal_list)
            MPC_multiguess_result[nIdxIniState][i] = Cost
            u_first = u_best_cand[0]
        else: #notihing
            print("error")
        
        x_next = EulerForwardCartpole_virtual(TS, x_cur, u_first)

        # update
        x_cur = x_next
        x_cur_clean = torch.tensor( [[x_next[0], x_next[1], x_next[4], x_next[3]]] )
    
    

def main():
    arg_list = []
    # model_list = [10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
    model_list = [40000]
    num_modelread = len(model_list)
    
    MAX_CORE_CPU = 1
    
    #initial state
    x_0_test = -0.47
    theta_0_test = 1.85
    thetared_0_test = ThetaToRedTheta(theta_0_test)
    x0_test_red = np.array([x_0_test , 0, theta_0_test, 0, thetared_0_test])
    x0_test_clean = np.array([[x_0_test , 0, thetared_0_test, 0]])

    # MPC parameters
    Q_REDUNDANT = 1000.0
    P_REDUNDANT = 1000.0
    Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
    R = np.diag([0.001])
    P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])
    HORIZON = 63 # mpc horizon
    initial_guess_x_grp = [5, 0]
    initial_guess_u_grp = [1000, -10000]
    idx_pos = 0
    idx_neg = 1
    
    
    for i in range(num_modelread):
        model_path = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+str(MODEL_FOLDER)+'/'+ str(model_list[i])
        result_path = os.path.join(RESULT_FOLDER, str(model_list[i]))
        arg_list.append((experiment, {'model_dir': model_path, 'result_dir': result_path, 
                                      'x0_test_red':x0_test_red, 'x0_test_clean':x0_test_clean, 'Q':Q, 'R':R, 'P':P,
                                      'results_dir':'logs', 'seed': 30, 'load_model_dir':MODEL_SAVED_PATH, 
                                      'train_data_load_path':DATA_LOAD_PATH,
                                      'u_filename': U_DATA_FILENAME, 'j_filename': J_DATA_FILENAME, 'x0_filename': X0_CONDITION_DATA_NAME,
                                      'device': 'cuda'}))
        
    with Pool(processes=MAX_CORE_CPU) as pool:
        pool.starmap(run_experiment, arg_list)
        
    # run MPC
    runMPC(x0_test_red, RESULT_FOLDER, Q, R, P, HORIZON, initial_guess_x_grp[idx_pos], initial_guess_u_grp[idx_pos])
    runMPC(x0_test_red, RESULT_FOLDER, Q, R, P, HORIZON, initial_guess_x_grp[idx_neg], initial_guess_u_grp[idx_neg])
        
    print("save data finsih")

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
