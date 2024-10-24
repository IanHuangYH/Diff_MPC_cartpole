from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params

from multiprocessing import Pool
import multiprocessing

# modify
DATASET = 'NMPC_Dataset'
J_NORMALIZER = 'GaussianNormalizer' #GaussianNormalizer, LimitsNormalizer, LogMinMaxNormalizer, LogZScoreNormalizer, OnlyLogNormalizer
UX_NORMALIZER = 'LimitsNormalizer'
MODEL_FOLDER = 'nmpc_batch_4096_random112500_limit_xu_randominiguess_noisedata_decayguess1'
RESULT_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving/'
MODEL_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+MODEL_FOLDER
DATA_LOAD_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/Random_also_noisedata_decayguess1_112500'

INITILA_X = 10
INITIAL_THETA = 15
CONTROL_STEP = 50
NOISE_NUM = 15
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


allow_ops_in_compiled_graph()




@single_experiment_yaml
def experiment(
    #########################################################################################
    # Model id
    x0_test_red: np.array,
    
    x0_test_clean: np.array,
    
    Q: np.array,
    
    R: np.array,
    
    P:np.array,
    
    model_dir: str,
    
    result_dir: str = RESULT_FOLDER,
    
    train_data_load_path: str = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/',
    
    u_filename: str = None,
    
    j_filename: str = None,
    
    x0_filename: str = None,

    planner_alg: str = 'mpd',

    n_samples: int = 1,

    n_diffusion_steps_without_noise: int = 5,

    ##############################################################
    device: str = 'cuda',

    ##############################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    load_model_dir: str = None
    ##############################################################
    # **kwargs
):
    ##############################################################
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ###############################################################
    print(f'##########################################################################################################')
    print(f'Algorithm -- {planner_alg}')
    
    if planner_alg == 'mpd':
        pass
    else:
        raise NotImplementedError

    ################################################################
    

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    
    args.pop('j_filename', None)
    args.pop('u_filename', None)
    args.pop('x0_filename', None)
    args.pop('train_data_load_path', None)

    #################################################################
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class=DATASET,
        j_normalizer=J_NORMALIZER,
        train_data_load_path = train_data_load_path,
        j_filename=j_filename,
        u_filename = u_filename,
        x0_filename = x0_filename,
        ux_normalizer = UX_NORMALIZER,
        **args,
        tensor_args=tensor_args

    )
    dataset = train_subset.dataset
    print(f'dataset -- {len(dataset)}')

    n_support_points = dataset.n_support_points
    Num_Hor_u = n_support_points - 1 # last one is j
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    ############################################################################
    # sampling loop
    x_track_d = np.zeros((CONTROL_STEP+1, NUM_STATE))
    u_track_d = np.zeros((CONTROL_STEP))
    j_track_d = np.zeros((CONTROL_STEP))
    x_track_d[0,:] = x0_test_red
    
    print('initial_state=(', x_track_d[:,0],')')
    
    FindGlobal_list = []
    for i in range(NUM_FIND_GLOBAL):
        tensor = torch.randn(1, n_support_points, 1)  # Create a 1x64x1 tensor
        FindGlobal_list.append([0, tensor])
        
    x_cur = x0_test_red
    
    # diffusion setting
    # Load prior model
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
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

    model = torch.compile(model)
    
    for i in range(0, CONTROL_STEP):
        
        x0_test_clean = torch.tensor(x0_test_clean).to(device) # load data to cuda

        hard_conds = None
        context = dataset.normalize_condition(x0_test_clean)

        #########################################################################
        # Sample u with classifier-free-guidance (CFG) diffusion model
        for j in range(NUM_FIND_GLOBAL):
            with TimerCUDA() as timer_model_sampling:
                u_normalized_iters = model.run_CFG(
                    context, hard_conds, WEIGHT_GUIDANC,
                    n_samples=n_samples, horizon=n_support_points,
                    return_chain=True,
                    sample_fn=ddpm_cart_pole_sample_fn,
                    n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
                )
            print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')

            ########
            u_iters = dataset.unnormalize_states(u_normalized_iters[:,:,0:Num_Hor_u,:])
            u_final_iter_candidate = u_iters[-1]
            u_final_iter_candidate = u_final_iter_candidate.cpu()
            
            # cal cost
            FindGlobal_list[j][0] = calMPCCost(Q,R,P,u_final_iter_candidate,x_cur, EulerForwardCartpole_virtual, TS)
            FindGlobal_list[j][1] = u_final_iter_candidate
                
        
        j_predict, u_best_cand = PickBestDiffResult(FindGlobal_list)
        j_track_d[i] = j_predict
        
            

        print(f'\n--------------------------------------\n')
        u_best_cand = u_best_cand.cpu()
        applied_input = u_best_cand[0,0,0]
        print("step:",i,'\n')
        print(f'applied_input -- {applied_input}')
        
        # save the control input from diffusion sampling
        u_track_d[i] = applied_input

        # update cart pole state
        x_next = EulerForwardCartpole_virtual(TS, x_cur, applied_input)
        print(f'x_next-- {x_next}')
        
        # save the new state
        x_track_d[i+1,:] = x_next
        
        # update
        x_cur = x_next
        x0_test_clean = torch.tensor( [x_next[0], x_next[1], x_next[4], x_next[3]] )

 #-------------------------- Sampling finished --------------------------------

    x_track_d = x_track_d[0:-1,:] # make it dimension same with u, j
    ########################## Diffusion & MPC Control Inputs Results Saving ################################

    os.makedirs(result_dir, exist_ok=True)
    
    # save diffusion
    diffusion_u = 'u_diffusion.npy'
    diffusion_u_path = os.path.join(result_dir, diffusion_u)
    np.save(diffusion_u_path, u_track_d)
    
    diffusion_x = 'x_diffusion.npy'
    diffusion_x_path = os.path.join(result_dir, diffusion_x)
    np.save(diffusion_x_path, x_track_d)
    
    diffusion_j = 'j_diffusion.npy'
    diffusion_j_path = os.path.join(result_dir, diffusion_j)
    np.save(diffusion_j_path, j_track_d)

def main():
    arg_list = []
    # model_list = [10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
    model_list = [8000]
    num_modelread = len(model_list)
    
    MAX_CORE_CPU = 1
    
    #initial state
    x_0_test = -0.47
    theta_0_test = 2.5
    thetared_0_test = ThetaToRedTheta(theta_0_test)
    x0_test_red = np.array([x_0_test , 0, theta_0_test, 0, thetared_0_test])
    x0_test_clean = np.array([x_0_test , 0, thetared_0_test, 0])

    # MPC parameters
    Q_REDUNDANT = 1000.0
    P_REDUNDANT = 1000.0
    Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
    R = np.diag([0.001])
    P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])
    HORIZON = 64 # mpc horizon
    initial_guess_x_grp = [5, 0]
    initial_guess_u_grp = [1000, -10000]
    idx_pos = 0
    idx_neg = 1
    
    # prepare multi-task and run diffusion
    for i in range(num_modelread):
        model_path = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+str(MODEL_FOLDER)+'/'+ str(model_list[i])
        result_path = os.path.join(RESULT_FOLDER, str(model_list[i]))
        arg_list.append((experiment, {'model_dir': model_path, 'result_dir': result_path, 
                                      'x0_test_red':x0_test_red, 'x0_test_clean':x0_test_clean, 'Q':Q, 'R':R, 'P':P,
                                      'results_dir':'logs', 'seed': 30, 'load_model_dir':MODEL_SAVED_PATH, 
                                      'train_data_load_path':DATA_LOAD_PATH,
                                      'u_filename': U_DATA_FILENAME, 'j_filename': J_DATA_FILENAME, 'x0_filename': X0_CONDITION_DATA_NAME,}))
        
    with Pool(processes=MAX_CORE_CPU) as pool:
        pool.starmap(run_experiment, arg_list)
        
    # run MPC
    runMPC(x0_test_red, RESULT_FOLDER, Q, R, P, HORIZON, initial_guess_x_grp[idx_pos], initial_guess_u_grp[idx_pos])
    runMPC(x0_test_red, RESULT_FOLDER, Q, R, P, HORIZON, initial_guess_x_grp[idx_neg], initial_guess_u_grp[idx_neg])
        
    print("save data finsih")

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
