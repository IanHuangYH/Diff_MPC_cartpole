from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import casadi as ca
import control
import numpy as np
import os

import matplotlib.pyplot as plt
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

#mpc param
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
R = 0.001
P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])


TS = 0.01


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
        unext = u_hor[:][i][0]
        for j in range(1,num_state):
            cost = cost + Q[j][j] * xnext[j] ** 2
        for j in range(num_u):
            cost = cost + R[j][j] * unext ** 2
    #final cost
    
    for i in range(num_state):
        cost = cost + Q[i][i] * xnext[i] ** 2
    for i in range(num_u):
        cost = cost + R[i][i] * u_hor[i][IdxLastU][0] ** 2
            
            
    return cost

def PickBestDiffResult( candidate_list:list ) -> torch.tensor:
    # candidate_list [[cost1, u_hor1],[cost2, u_hor2],...]
    num_candidate = len(candidate_list)
    best_idx = 0
    for i in range(num_candidate-1):
        if( candidate_list[i+1][0] < candidate_list[i][0] ):
            best_idx = i+1
    return candidate_list[best_idx][1]

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = 'trained_models' 
MODEL_FOLDER = 'nmpc_1st_org_model' # choose a main model folder saved in the trained_models (eg. 420000 is the number of total training data, this folder contains all trained models based on the 420000 training data)
MODEL_ID = 230000 # number of training

MODEL_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/trained_models/'+str(MODEL_FOLDER)+'/'+ str(MODEL_ID) # the absolute path of the trained model

WEIGHT_GUIDANC = 0.1 # non-conditioning weight
X0_IDX = 150 # range:[0,199] 20*20 data 
CONTROL_STEP = 80 # control loop (steps)
HORIZON = 64 # mpc horizon
U_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving'

X0_RANGE = np.array([-0.5, 0.5])
THETA0_RANGE = np.array([3*np.pi/4, 5*np.pi/4])
DATASET = 'NMPC_Dataset'
NUM_STATE = 5

NUM_FIND_GLOBAL = 5

TS = 0.01

@single_experiment_yaml
def experiment(
    #########################################################################################
    # Model id
    model_id: str = MODEL_FOLDER, 

    planner_alg: str = 'mpd',

    n_samples: int = 1,

    n_diffusion_steps_without_noise: int = 5,

    ##############################################################
    device: str = 'cuda',

    ##############################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    ##############################################################
    # **kwargs
):
    ##############################################################

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ###############################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')
    
    if planner_alg == 'mpd':
        pass
    else:
        raise NotImplementedError

    ################################################################
    model_dir = MODEL_PATH 
    results_dir = os.path.join(model_dir, 'results_inference')
    
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    #################################################################
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class=DATASET,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    print(f'dataset -- {len(dataset)}')

    n_support_points = dataset.n_support_points
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    #################################################################
    # load initial starting state x0
    x_0_test = np.random.uniform( X0_RANGE[0], X0_RANGE[1] )
    theta_0_test = np.random.uniform( THETA0_RANGE[0], THETA0_RANGE[1] )
    thetared_0_test = ThetaToRedTheta(theta_0_test)


    #initial context
    x0_test_red = np.array([x_0_test , 0, theta_0_test, 0, thetared_0_test])
    x0_test_clean = np.array([[x_0_test , 0, thetared_0_test, 0]])

    ############################################################################
    # sampling loop
    x_track = np.zeros((NUM_STATE, CONTROL_STEP+1))
    u_track = np.zeros((CONTROL_STEP))
    u_horizon_track = np.zeros((HORIZON, CONTROL_STEP))

    x_track[:,0] = x0_test_red
    
    print('initial_state=(', x_track[:,0],')')
    
    FindGlobal_list = []
    for i in range(5):
        tensor = torch.randn(1, HORIZON, 1)  # Create a 1x64x1 tensor
        FindGlobal_list.append([0, tensor])
        
    

    for i in range(0, CONTROL_STEP):
        
        x0_test_clean = torch.tensor(x0_test_clean).to(device) # load data to cuda

        hard_conds = None
        context = dataset.normalize_condition(x0_test_clean)


        #########################################################################
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


        ########
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
            u_iters = dataset.unnormalize_states(u_normalized_iters)
            u_final_iter_candidate = u_iters[-1]
            FindGlobal_list[j][0] = calMPCCost(Q,R,P,u_final_iter_candidate,x0_test_red, EulerForwardCartpole_virtual, TS)
            FindGlobal_list[j][1] = u_final_iter_candidate
        
        u_best_cand = PickBestDiffResult(FindGlobal_list)
        
            

        print(f'\n--------------------------------------\n')
        
        horizon_inputs = np.zeros((1, HORIZON))
        u_best_cand = u_best_cand.cpu()
        horizon_inputs = u_best_cand.squeeze(-1).numpy()
        print(f'horizon_inputs -- {horizon_inputs}')
        applied_input = horizon_inputs[0][0] # retain 4 decimal places
        print(f'applied_input -- {applied_input}')

        # save the control input from diffusion sampling
        u_track[i] = applied_input
        u_horizon_track[:,i] = horizon_inputs

        # update cart pole state
        x_next = EulerForwardCartpole_virtual(TS, x0_test_red, applied_input)
        print(f'x_next-- {x_next}')
        
        # save the new state
        x_track[:,i+1] = x_next
        
        # update
        x0_test_red = x_next

        

    # print all x and u 
    print(f'x_track-- {x_track.T}')
    print(f'u_track-- {u_track}')
 #-------------------------- Sampling finished --------------------------------



 ########################## MPC #################################

    # simulation time
    # T = 3.3  # Total time (seconds) 6.5
    # dt = 0.1  # Time step (seconds)
    # t = np.arange(0, T, dt) # time intervals 65
    # print(t.shape)

    N = HORIZON # prediction horizon

    # mpc parameters
    Q = np.diag([10, 1, 10, 1]) 
    R = np.array([[1]])
    P = np.diag([100, 1, 100, 1])

    # Define the initial states range
    rng_x = np.linspace(-1,1,20) 
    rng_theta = np.linspace(-np.pi/4,np.pi/4,20) 
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
            rng0.append([m,n])
    rng0 = np.array(rng0)
    print(f'rng0 -- {rng0}')

    # ##### data collecting loop #####

    # data set for each turn
    x_mpc_track = np.zeros((4, num_loop+1))
    u_mpc_track = np.zeros((1, num_loop))
    u_mpc_horizon_track = np.zeros((num_loop, HORIZON))

    x_0 = rng0[test,0]
    x_0= round(x_0, 4)
    theta_0 = rng0[test,1]
    theta_0= round(theta_0, 4)

    #save the initial states
    x0_test_red = np.array([x_0, 0, theta_0, 0])  # Initial states
    print(f'x0-- {x0_test_red}')
    x_mpc_track[:,0] = x0_test_red

    ############# control loop ##################
    for i in range(0, num_loop):
        # casadi_Opti
        optimizer = ca.Opti()

        # x and u mpc prediction along N
        X_pre = optimizer.variable(4, N + 1) 
        print(X_pre)
        U_pre = optimizer.variable(1, N) 

        optimizer.subject_to(X_pre[:, 0] == x0_test_red)  # starting state

        # cost 
        cost = 0

        # initial cost
        cost += Q[0,0]*X_pre[0, 0]**2 + Q[1,1]*X_pre[1, 0]**2 + Q[2,2]*X_pre[2, 0]**2 + Q[3,3]*X_pre[3, 0]**2

        # state cost
        for k in range(0,N-1):
            x_next = cart_pole_dynamics(X_pre[:, k], U_pre[:, k])
            optimizer.subject_to(X_pre[:, k + 1] == x_next)
            cost += Q[0,0]*X_pre[0, k+1]**2 + Q[1,1]*X_pre[1, k+1]**2 + Q[2,2]*X_pre[2, k+1]**2 + Q[3,3]*X_pre[3, k+1]**2 + U_pre[:, k]**2

        # terminal cost
        x_terminal = cart_pole_dynamics(X_pre[:, N-1], U_pre[:, N-1])
        optimizer.subject_to(X_pre[:, N] == x_terminal)
        cost += P[0,0]*X_pre[0, N]**2 + P[1,1]*X_pre[1, N]**2 + P[2,2]*X_pre[2, N]**2 + P[3,3]*X_pre[3, N]**2 + U_pre[:, N-1]**2

        optimizer.minimize(cost)
        optimizer.solver('ipopt')
        sol = optimizer.solve()

        X_sol = sol.value(X_pre)
        # print(f'X_sol_shape -- {X_sol.shape}')
        U_sol = sol.value(U_pre)
        print(f'U_sol - {U_sol}')
        
        # select the first updated states as new starting state ans save in the x_track
        x0_test_red = X_sol[:,1]
        print(f'x0_new-- {x0_test_red}')
        x_mpc_track[:,i+1] = x0_test_red

        #save the first computed control input
        u_mpc_track[:,i] = U_sol[0]
        
        # save the computed control inputs along the mpc horizon
        u_mpc_horizon_track[i,:] = U_sol
        
    u_mpc_track = np.round(u_mpc_track,decimals=4)
    u_mpc_horizon_track = np.round(u_mpc_horizon_track,decimals=4)
    
    print(f'u_mpc_track -- {u_mpc_track}')
    print(f'u_mpc_horizon_track -- {u_mpc_horizon_track}')

    ########################## Diffusion & MPC Control Inputs Results Saving ################################

    results_folder = os.path.join(U_SAVED_PATH, 'model_'+ str(MODEL_ID), 'x0_'+ str(X0_IDX))
    os.makedirs(results_folder, exist_ok=True)
    
    # save the first u 
    diffusion_u = 'u_diffusion.npy'
    diffusion_u_path = os.path.join(results_folder, diffusion_u)
    np.save(diffusion_u_path, u_track)

    mpc_u = 'u_mpc.npy'
    mpc_u_path = os.path.join(results_folder, mpc_u)
    np.save(mpc_u_path, u_mpc_track)

    # save the u along horizon
    diffusion_u_horizon = 'u_horizon_diffusion.npy'
    diffusion_u_horizon_path = os.path.join(results_folder, diffusion_u_horizon)
    np.save(diffusion_u_horizon_path, u_horizon_track)

    mpc_u_horizon = 'u_horizon_mpc.npy'
    mpc_u_horizon_path = os.path.join(results_folder, mpc_u_horizon)
    np.save(mpc_u_horizon_path, u_mpc_horizon_track)

    ########################## plot ################################
    num_i = num_loop
    step = np.linspace(0,num_i+2,num_i+1)
    step_u = np.linspace(0,num_i+1,num_i)

    plt.figure(figsize=(10, 8))

    plt.subplot(5, 1, 1)
    plt.plot(step, x_track[0, :])
    plt.plot(step, x_mpc_track[0, :])
    plt.legend(['Diffusion Sampling', 'MPC']) 
    plt.ylabel('Position (m)')
    plt.grid()

    plt.subplot(5, 1, 2)
    plt.plot(step, x_track[1, :])
    plt.plot(step, x_mpc_track[1, :])
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(5, 1, 3)
    plt.plot(step, x_track[2, :])
    plt.plot(step, x_mpc_track[2, :])
    plt.ylabel('Angle (rad)')
    plt.grid()

    plt.subplot(5, 1, 4)
    plt.plot(step, x_track[3, :])
    plt.plot(step, x_mpc_track[3, :])
    plt.ylabel('Ag Velocity (rad/s)')
    plt.grid()

    plt.subplot(5, 1, 5)
    plt.plot(step_u, u_track.reshape(num_loop,))
    plt.plot(step_u, u_mpc_track.reshape(num_loop,))
    plt.ylabel('Ctl Input (N)')
    plt.xlabel('Control Step')
    plt.grid()
    # plt.show()
    # save figure 
    figure_name = 'w_' + str(WEIGHT_GUIDANC) + 'x0_' + str(X0_IDX) + 'steps_' + str(CONTROL_STEP) + '.png'
    figure_path = os.path.join(results_dir, figure_name)
    plt.savefig(figure_path)

    ######### Performance Check #########
    position_difference = np.sum(np.abs(x_track[0, :] - x_mpc_track[0, :]))
    print(f'position_difference - {position_difference}')

    velocity_difference = np.sum(np.abs(x_track[1, :] - x_mpc_track[1, :]))
    print(f'velocity_difference - {velocity_difference}')

    theta_difference = np.sum(np.abs(x_track[2, :] - x_mpc_track[2, :]))
    print(f'theta_difference - {theta_difference}')

    thetaVel_difference = np.sum(np.abs(x_track[3, :] - x_mpc_track[3, :]))
    print(f'thetaVel_difference - {thetaVel_difference}')

    u_difference = np.sum(np.abs(u_track.reshape(num_loop,) - u_mpc_track.reshape(num_loop,)))
    print(f'u_difference - {u_difference}')



if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
