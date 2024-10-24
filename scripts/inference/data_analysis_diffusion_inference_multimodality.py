from abc import ABC, abstractmethod
import torch.nn as nn
import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch

from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS, diffusion_model_base
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params


import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# diffusion modify
DIFF_MODEL_FOLDER = 'nmpc_batch_4096_random112500_zscroe_xu_decay1'
DIFF_MODEL_CHECKPOINT = 8000
DIFF_DATA_LOAD_FOLER = 'Random_also_noisedata_decayguess1_112500'
DIFF_DATASET_CLASS = 'NMPC_Dataset'
DIFF_INITILA_X = 10 #10
DIFF_INITIAL_THETA = 15 #15
DIFF_NOISE_NUM = 15 #15
DIFF_J_NORMALIZER = 'GaussianNormalizer' #GaussianNormalizer, LimitsNormalizer, LogMinMaxNormalizer, LogZScoreNormalizer, OnlyLogNormalizer
DIFF_UX_NORMALIZER = 'GaussianNormalizer' 

# MPC_modify
MPC_U_RANGE = np.array([-4500.0,4500])
OPT_SETTING = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

# Common modify
CONTROL_STEP = 50
HOR = 64
NUM_FIND_MULTIMODALITY = 20

DEVICE = 'cuda'
NUM_CONTROLLER = 1
B_ISSAVE = 0
N_GPU = torch.cuda.device_count()
N_DATA_SCALE = 15
NUM_MONTECARLO = N_GPU * N_DATA_SCALE

# monte carlo initial state sample range
INITIAL_X_RANGE = np.array([-3,3])
INITIAL_THETA_RANGE = np.array([1.8,4.4])

# result filename
RESULT_SAVED_PATH = os.path.join('/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving',DIFF_MODEL_FOLDER,str(DIFF_MODEL_CHECKPOINT))
U_SOL_ALL = 'analysis_diff_infe_all_u.npy'
NUM_MULTI_MODALITY = 'analysis_diff_num_multimodality.npy'
# path
DIFF_MODEL_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+DIFF_MODEL_FOLDER+'/'+str(DIFF_MODEL_CHECKPOINT)
DIFF_DATA_LOAD_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/'+DIFF_DATA_LOAD_FOLER

# idx
NUM_ONE_METHODGRP = CONTROL_STEP * NUM_MONTECARLO

# fix_random_seed(40)
# Data Name Setting
Diff_filename_idx = '_ini_'+str(DIFF_INITILA_X)+'x'+str(DIFF_INITIAL_THETA)+'_noise_'+str(DIFF_NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
DIFF_X0_CONDITION_DATA_NAME = 'x0' + Diff_filename_idx
DIFF_U_DATA_FILENAME = 'u' + Diff_filename_idx
DIFF_J_DATA_FILENAME = 'j' + Diff_filename_idx

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
NUM_STATE = 5
WEIGHT_GUIDANC = 0.01 # non-conditioning weight


class MPCBasedController(ABC):
    @abstractmethod
    def SolveUThenGetCost(self:np.array, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int):
        # x_cur_red, x_cur_clean are 1D array with 5 and 4 seperately
        pass
    
    def SolveMultiModalityUAndRecord(self:np.array, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int, 
                                     nIdxMethod: int, nIdxIniStateGrp: int, nIdxContrlStep:int, u_buffer:list):
        pass
    
    def PrepareModelForDDP(self, device, rank):
        pass

class DiffusionController(MPCBasedController):
    def __init__(self, device: str, 
                 model_dir: str, train_data_load_path: str,
                 j_filename, u_filename, x0_filename,
                 diffusionDatasetClass: str):
        device = get_torch_device(device)
        tensor_args = {'device': device, 'dtype': torch.float32}
        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
        args.pop('j_filename', None)
        args.pop('u_filename', None)
        args.pop('x0_filename', None)
        args.pop('train_data_load_path', None)

        #################################################################
        # Load dataset
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class=diffusionDatasetClass,
            j_normalizer=DIFF_J_NORMALIZER,
            train_data_load_path = train_data_load_path,
            j_filename=j_filename,
            u_filename = u_filename,
            x0_filename = x0_filename,
            ux_normalizer = DIFF_UX_NORMALIZER,
            **args,
            tensor_args=tensor_args
        )
        self.m_dataset = train_subset.dataset
        
        ####################################################################
        # Load model
        # diffusion setting
        # Load prior model
        diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
        state_dim=self.m_dataset.state_dim,
        n_support_points=self.m_dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
        )
        diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=ConditionedTemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
        )
        # 'ema_model_current_state_dict.pth'
        diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
        map_location=tensor_args['device'])
        )
        diffusion_model.eval()
        model = diffusion_model
        self.m_model = torch.compile(model)
        self.m_model.eval()
        
        self.m_device = device
        
    def SolveUThenGetCost(self, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int ):
        FindGlobal_list = []
        for j in range(NUM_FIND_MULTIMODALITY):
            tensor = torch.randn(1, Hor, 1)  # Create a 1x64x1 tensor
            FindGlobal_list.append([0, tensor])
        x0_tensor = torch.tensor(x_cur_clean).to(self.m_device)
        x0_strd = self.m_dataset.normalize_condition(x0_tensor)
        
        starttime = time.time()
        for j in range(NUM_FIND_MULTIMODALITY):
            with torch.no_grad():
                u_normalized_iters = self.m_model.run_CFG(
                    x0_strd, None, WEIGHT_GUIDANC,
                    n_samples=1, horizon=Hor,
                    return_chain=True,
                    sample_fn=ddpm_cart_pole_sample_fn,
                    n_diffusion_steps_without_noise=5,
                )
                u_iters = self.m_dataset.unnormalize_states(u_normalized_iters[:,:,0:Hor,:])
                u_final_iter_candidate = u_iters[-1].cpu()
                            
            FindGlobal_list[j][0] = calMPCCost(Q,R,P,u_final_iter_candidate, x_cur_red, EulerForwardCartpole_virtual, TS)
            FindGlobal_list[j][1] = u_final_iter_candidate
        endtime = time.time()
                
        Cost, u_best_cand = PickBestDiffResult(FindGlobal_list)
        u_first = u_best_cand[0,0,0]
        deltaTime = endtime - starttime
        return u_first, Cost.item(), deltaTime
    
    def SolveMultiModalityUAndRecord(self:np.array, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int, 
                                     nIdxMethod: int, nIdxIniStateGrp: int, nIdxContrlStep:int, u_buffer:list):
        Q_tensor = torch.tensor(Q, device=self.m_device)
        R_tensor = torch.tensor(R, device=self.m_device)
        P_tensor = torch.tensor(P, device=self.m_device)
        
        FindGlobal_list = torch.zeros(NUM_FIND_MULTIMODALITY, 2, device=self.m_device)
        
        x0_clean_tensor = torch.tensor(x_cur_clean).to(self.m_device)
        x0_red_tensor = torch.tensor(x_cur_red).to(self.m_device)
        x0_strd = self.m_dataset.normalize_condition(x0_clean_tensor)
        x0_red_tensor_extend = x0_red_tensor.unsqueeze(0).repeat(NUM_FIND_MULTIMODALITY, 1) 
        
        with torch.no_grad():
            u_normalized_iters = self.m_model.run_CFG(
                x0_strd, None, WEIGHT_GUIDANC,
                n_samples=NUM_FIND_MULTIMODALITY, horizon=Hor,
                return_chain=True,
                sample_fn=ddpm_cart_pole_sample_fn,
                n_diffusion_steps_without_noise=5,
            )
            u_iters = self.m_dataset.unnormalize_states(u_normalized_iters[:,:,0:Hor,:]) # iter x n_sample x Hor x 1
            u_final_iter_candidate = u_iters[-1] # n_sample x Hor x 1
            
            
            FindGlobal_list[:,0] = calMPCCost_tensor(Q_tensor,R_tensor,P_tensor,u_final_iter_candidate, x0_red_tensor_extend, EulerForwardCartpole_virtual_tensor, TS, self.m_device)
            FindGlobal_list[:,1] = u_final_iter_candidate[:,0,0]
        
        Cost, u_best_cand = PickBestDiffResult(FindGlobal_list)
        u_first = u_best_cand.item()
        
        for j in range(NUM_FIND_MULTIMODALITY):
            u_buffer[nIdxMethod][nIdxIniStateGrp][j][nIdxContrlStep] = FindGlobal_list[j,0].item()

        return u_first
    
    def PrepareModelForDDP(self, device, rank):
        self.m_model.to(device)
        self.m_model = DDP(self.m_model, device_ids=[rank])
        self.m_model.eval()
    
class MPCMultiGuessController(MPCBasedController):
    def __init__(self, initial_Uguess_range: np.array):
        self.minGuessU = initial_Uguess_range[0]
        self.maxGuessU = initial_Uguess_range[1]
        
    def SolveUThenGetCost(self, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int):
        FindGlobal_list = []
        for j in range(NUM_FIND_MULTIMODALITY):
            array = np.zeros((1, Hor))  # Create a 1x64x1 tensor
            FindGlobal_list.append([0, array])
            
        starttime = time.time()
        for j in range(NUM_FIND_MULTIMODALITY):
            u_ini_guess, x_ini_guess = GenerateRandomInitialGuess(self.minGuessU, self.maxGuessU)
            X_sol, U_sol, Cost = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x_cur_red, x_ini_guess, u_ini_guess, NUM_STATE, Hor, Q, R, P, TS, OPT_SETTING)
            FindGlobal_list[j][0] = Cost
            FindGlobal_list[j][1] = U_sol
        endtime = time.time()
        Cost, u_best_cand = PickBestDiffResult(FindGlobal_list)
        
        u_first = u_best_cand[0]
        deltaTime = endtime - starttime
        return u_first, Cost, deltaTime  
    
class ControllerManager:
    def __init__(self):
        # Initially set the strategy
        self.m_StrategyList : list = []
        self.m_Strategy : MPCBasedController = None
        self.m_NumStrategy : int = 0
    
    def add_stategy(self, Contoller: MPCBasedController):
        self.m_StrategyList.append(Contoller)
        self.m_NumStrategy +=1
        
    def set_strategy(self, nIdxController: int):
        if nIdxController > self.m_NumStrategy:
            print("error")
            return
        self.m_Strategy = self.m_StrategyList[nIdxController]
        

    def ComputeInputAndCost(self, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int ):
        # Delegate behavior to the strategy
        u, cost, time = self.m_Strategy.SolveUThenGetCost(x_cur_red, x_cur_clean, Q, R, P, Hor)
        return u, cost, time
    
    def SolveMultiModalityUAndRecord(self, x_cur_red:np.array, x_cur_clean:np.array, Q:np.array, R:np.array, P:np.array, Hor:int, 
                                     nIdxMethod: int, nIdxIniStateGrp: int, nIdxContrlStep:int, u_buffer:list ):
        return self.m_Strategy.SolveMultiModalityUAndRecord(x_cur_red, x_cur_clean, Q, R, P, Hor, nIdxMethod, nIdxIniStateGrp, nIdxContrlStep, u_buffer)
    
    def SolveMultiModalityUAndRecordForAllIniState(self, x_cur_red:torch.tensor, x_cur_clean:torch.tensor, Q:torch.tensor, R:torch.tensor, P:torch.tensor, Hor:int, 
                                     nIdxMethod: int, nIdxContrlStep:int, u_buffer:list ):
        return 
    def PrepareModelForDDP(self, device, rank):
        self.m_Strategy.PrepareModelForDDP(device, rank)
    
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

def GenerateRandomInitialGuess(min_random, max_random):
    u_ini_guess = np.random.uniform(min_random, max_random, 1)[0]
    if u_ini_guess >=0:
        x_ini_guess = 5
    else:
        x_ini_guess = -5
    return u_ini_guess, x_ini_guess
    
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

def EulerForwardCartpole_virtual_tensor(dt, x:torch.tensor,u:torch.tensor, device:torch.device) -> torch.tensor:
# x: n_sample x 5 x 1, u: n_sample
    xdot = torch.stack([
        x[:,1],            # xdot 
        ( MPLP * -torch.sin(x[:,2]) * x[:,3]**2 
          +MPG * torch.sin(x[:,2]) * torch.cos(x[:,2])
          + u 
          )/(M_TOTAL - M_POLE*torch.cos(x[:,2]))**2, # xddot

        x[:,3],        # thetadot
        ( -MPLP * torch.sin(x[:,2]) * torch.cos(x[:,2]) * x[:,3]**2
          -MTG * torch.sin(x[:,2])
          -(torch.cos(x[:,2])*u)
          )/(MTLP - MPLP*torch.cos(x[:,2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[:,2]-torch.pi) * x[:,3]   # theta_stat_dot],dim=1, device=device)
    ], dim=1)
    

    return x + xdot * dt

def ThetaToRedTheta(theta):
    return (theta-torch.pi)**2/-torch.pi + torch.pi

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

def calMPCCost_tensor(Q:torch.tensor,R:torch.tensor,P:torch.tensor,u_hor:torch.tensor,x0:torch.tensor, ModelUpdate_func, dt, device) -> float:
    # u nsamplex64x1, x0: nsample x state, compute all the sample at the same time 
    num_state = x0.shape[1]
    num_u = 1
    num_hor = u_hor.size(1)
    cost = torch.zeros(u_hor.shape[0], device=device)
    
    # initial cost
    for i in range(num_state):
        cost = cost + Q[i][i] * x0[:,i] ** 2
    
    for i in range(num_u):
        cost = cost + R[i][i] * u_hor[:,0,0] ** 2
        
    x_cur = x0
    u_cur = u_hor[:,0,0]
    IdxLastU = num_hor-1
    # stage cost
    for i in range(1,IdxLastU):
        xnext = ModelUpdate_func(dt, x_cur, u_cur, device)
        unext = u_hor[:,i,0]
        for j in range(1,num_state):
            cost = cost + Q[j][j] * xnext[:,j] ** 2
        for j in range(num_u):
            cost = cost + R[j][j] * unext ** 2
        # update
        u_cur = unext
        x_cur = xnext
        
        
    #final cost
    for i in range(num_state):
        cost = cost + P[i][i] * xnext[:,i] ** 2
            
            
    return cost

def PickBestDiffResult( candidate_list:torch.tensor ) -> torch.tensor:
    # candidate_list [[cost1, u_hor1],[cost2, u_hor2],...]
    num_candidate = len(candidate_list)
    best_idx = 0
    for i in range(num_candidate-1):
        if( candidate_list[i+1][0] < candidate_list[best_idx][0] ):
            best_idx = i+1
    return candidate_list[best_idx][0], candidate_list[best_idx][1]

def InferenceBy_one_method_single_IniState(nIdxIniState: int, nMethodSelect: int, u_all_result_buffer: list,
                                           x0_red: np.array, x0_clean: np.array,
                                           Q: np.array, R: np.array, P:np.array, Hor: int,
                                           contoller: ControllerManager
):
    x_cur_red = x0_red
    x_cur_clean = x0_clean
    
    # control loop
    for i in range(CONTROL_STEP):
        print("method:",nMethodSelect,"Initial_state_th:",nIdxIniState, "control_step:",i,"\n")
        
        u_first = contoller.SolveMultiModalityUAndRecord(x_cur_red, x_cur_clean, Q, R, P, Hor, 
                                                         nMethodSelect, nIdxIniState, i, u_all_result_buffer)
        x_next = EulerForwardCartpole_virtual(TS, x_cur_red, u_first)
        
        # update
        x_cur_red = x_next
        x_cur_clean = torch.tensor( [x_next[0], x_next[1], x_next[4], x_next[3]] )
        
    print("method:",nMethodSelect,"Initial_state_th:",nIdxIniState, "finish! \n")
    
def InferenceBy_one_method_All_IniState(   nMethodSelect: int, u_all_result_buffer: list,
                                           X0_All_red: np.array, X0_All_clean: np.array,
                                           Q: np.array, R: np.array, P:np.array, Hor: int,
                                           contoller: ControllerManager
):
# X0_All_red: MonteCarlo x 5, X0_All_clean: MonteCarlo x 4
    x_cur_red = X0_All_red
    x_cur_clean = X0_All_clean
    
    # control loop
    for i in range(CONTROL_STEP):
        print("method:",nMethodSelect, "control_step:",i,"\n")
        
        u_first = contoller.SolveMultiModalityUAndRecord(x_cur_red, x_cur_clean, Q, R, P, Hor, 
                                                         nMethodSelect, nIdxIniState, i, u_all_result_buffer)
        x_next = EulerForwardCartpole_virtual(TS, x_cur_red, u_first)
        
        # update
        x_cur_red = x_next
        x_cur_clean = torch.tensor( [x_next[0], x_next[1], x_next[4], x_next[3]] )
        
    print("method:",nMethodSelect,"Initial_state_th:",nIdxIniState, "finish! \n")
    
# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
# Destroy the distributed environment
def cleanup():
    dist.destroy_process_group()

    
def RunTaskForEachGPU(rank,
                        nIdxIniState: int, nMethodSelect: int, u_all_result_buffer: list,
                        x0_red: np.array, x0_clean: np.array,
                        Q: np.array, R: np.array, P:np.array, Hor: int,
                        contoller: ControllerManager, NumGPU:int):
    setup(rank, NumGPU)
    # Set the device for the current process
    device = torch.device(f"cuda:{rank}")
    
    # move it to the current device
    contoller.PrepareModelForDDP(device, rank)
    
    # run inference and dynamic
    InferenceBy_one_method_single_IniState(nIdxIniState, nMethodSelect, u_all_result_buffer,
                                           x0_red, x0_clean,
                                           Q, R, P, Hor,
                                           contoller)
    
    cleanup()
    

def TestSingleInitialStateForEachMethod(Diff_ctrl,Q, R, P, Hor, U_All_SharedMemory):
    
    CtrlManager = ControllerManager()
    CtrlManager.add_stategy(Diff_ctrl)
    
    x0 = np.random.uniform(INITIAL_X_RANGE[0], INITIAL_X_RANGE[1])
    theta0 = np.random.uniform(INITIAL_THETA_RANGE[0], INITIAL_THETA_RANGE[1])
    theta_red_0 = ThetaToRedTheta(theta0)
    X0_clean = np.array([x0, 0.0, theta_red_0, 0.0])
    X0_red = np.array([x0, 0.0, theta0, 0.0, theta_red_0])

    for i in range(NUM_CONTROLLER):
        CtrlManager.set_strategy(i)
        InferenceBy_one_method_single_IniState(0, i, U_All_SharedMemory,
                                X0_red, X0_clean, 
                                Q, R, P, Hor,
                                CtrlManager)

def TestAllInitialStateForOneMethod(nSelectMethod, ctrl,Q, R, P, Hor, U_All_SharedMemory):
    CtrlManager = ControllerManager()
    CtrlManager.add_stategy(ctrl)
    CtrlManager.set_strategy(0)
    
    X0_all_clean = torch.zeros(NUM_MONTECARLO,4)
    X0_all_red = torch.zeros(NUM_MONTECARLO,5)
    for i in range(NUM_MONTECARLO):
        x0 = torch.random.uniform(INITIAL_X_RANGE[0], INITIAL_X_RANGE[1])
        theta0 = torch.random.uniform(INITIAL_THETA_RANGE[0], INITIAL_THETA_RANGE[1])
        theta_red_0 = ThetaToRedTheta(theta0)
        X0_clean = torch.tensor([x0, 0.0, theta_red_0, 0.0])
        X0_red = torch.tensor([x0, 0.0, theta0, 0.0, theta_red_0])
        X0_all_clean[i,:] = X0_clean
        X0_all_red[i,:] = X0_red
        
    InferenceBy_one_method_All_IniState(nSelectMethod, U_All_SharedMemory,
                                X0_all_red, X0_all_clean, 
                                Q, R, P, Hor,
                                CtrlManager)


def main():
    
    u_all_filepath = os.path.join(RESULT_SAVED_PATH,U_SOL_ALL)
    num_multimodality_filepath = os.path.join(RESULT_SAVED_PATH,NUM_MULTI_MODALITY)
    if B_ISSAVE == 0:
        MAX_CORE_CPU = 2
        # MPC parameters
        Q_REDUNDANT = 1000.0
        P_REDUNDANT = 1000.0
        Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
        R = np.diag([0.001])
        P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])
        
        # initialize controller
        Diff_ctrl = DiffusionController(DEVICE, DIFF_MODEL_PATH, DIFF_DATA_LOAD_PATH, DIFF_J_DATA_FILENAME, DIFF_U_DATA_FILENAME, DIFF_X0_CONDITION_DATA_NAME, DIFF_DATASET_CLASS) 
        MPCMulti_ctrl = MPCMultiGuessController(MPC_U_RANGE)
        U_All_SharedMemory = list([[[[0.0 for _ in range(CONTROL_STEP)] for _ in range(NUM_FIND_MULTIMODALITY)] for _ in range(NUM_MONTECARLO)] for _ in range(NUM_CONTROLLER)])
        TestSingleInitialStateForEachMethod(Diff_ctrl, Q, R, P, HOR, U_All_SharedMemory)
        
        with mp.Manager() as manager:
            # method x group x test multimodality x control step
            U_All_SharedMemory = manager.list([ manager.list([ manager.list([ manager.list([0.0 for _ in range(CONTROL_STEP)]) for _ in range(NUM_FIND_MULTIMODALITY)]) for _ in range(NUM_MONTECARLO)]) for _ in range(NUM_CONTROLLER)])
            Num_Multimodaity_SharedMemory = manager.list([[[0.0 for _ in range(CONTROL_STEP)] for _ in range(NUM_MONTECARLO)] for _ in range(NUM_CONTROLLER)])
            
            # test
            
            
            ArgList = []
            
            CtrlManager = ControllerManager()
            CtrlManager.add_stategy(Diff_ctrl)
            CtrlManager.set_strategy(0)
            for i in range(N_GPU):
                for j in range(int(NUM_MONTECARLO/N_GPU)):
                    ArgList.append()
            
            for i in range(NUM_CONTROLLER):
                CtrlManager = ControllerManager()
                CtrlManager.add_stategy(Diff_ctrl)
                CtrlManager.set_strategy(i)
                for j in range(0,NUM_MONTECARLO):
                    x0 = np.random.uniform(INITIAL_X_RANGE[0], INITIAL_X_RANGE[1])
                    theta0 = np.random.uniform(INITIAL_THETA_RANGE[0], INITIAL_THETA_RANGE[1])
                    theta_red_0 = ThetaToRedTheta(theta0)
                    X0_clean = np.array([x0, 0.0, theta_red_0, 0.0])
                    X0_red = np.array([x0, 0.0, theta0, 0.0, theta_red_0])
                    ArgList.append((j, i, U_All_SharedMemory,
                                    X0_red, X0_clean, 
                                    Q, R, P, HOR,
                                    CtrlManager, N_GPU))
            mp.spawn(RunTaskForEachGPU, ArgList, nprocs=N_GPU, join=True)
            
            
            U_all = np.array(U_All_SharedMemory)
            np.save(u_all_filepath, U_all)
            u_test = np.load(u_all_filepath)
            print("test finish")
            
    
    # if B_ISSAVE == 1:
    #     j_all = np.load(j_all_filepath)
    #     j_mean = np.load(j_mean_filepath)
    #     j_std = np.load(j_std_filepath)
    #     time_all = np.load(time_all_filepath)
    #     time_mean = np.load(time_mean_filepath)
    #     time_std = np.load(time_std_filepath)
    #     print("j_all.shape:",j_all.shape)
    #     print("j_mean.shape:",j_mean.shape)
    #     print("j_std.shape:",j_std.shape)
    #     print("time_all.shape:",time_all.shape)
    #     print("time_mean.shape:",time_mean.shape)
    #     print("time_std.shape:",time_std.shape)
        
        
    

if __name__ == '__main__':
    # multiprocessing.set_start_method("spawn")
    main()