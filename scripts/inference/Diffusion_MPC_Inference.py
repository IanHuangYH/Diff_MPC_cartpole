from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import casadi as ca
import control
import numpy as np
import os

#import einops
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

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../data_trained_models/'
MODEL_FOLDER = '2406400_training_data'

MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/data_trained_models/2406400_training_data/200000'
MODEL_ID = 200000
WEIGHT_GUIDANC = 0.01
X0_IDX = 95 # range:[0,99]
ITERATIONS = 50
HORIZON = 8
U_SAVED_PATH = '/root/cartpoleDiff/cartpole_inference_u_results'

# cart pole dynamics
def cart_pole_dynamics(x, u):
    A = np.array([
    [0, 1, 0, 0],
    [0, -0.1, 3, 0],
    [0, 0, 0, 1],
    [0, -0.5, 30, 0]
    ])

    B = np.array([
    [0],
    [2],
    [0],
    [5]
    ])

    C = np.eye(4)

    D = np.zeros((4,1))

    # state space equation
    sys_continuous = control.ss(A, B, C, D)

    # sampling time
    Ts = 0.1

    # convert to discrete time dynamics
    sys_discrete = control.c2d(sys_continuous, Ts, method='zoh')

    A_d = sys_discrete.A
    B_d = sys_discrete.B
    C_d = sys_discrete.C
    D_d = sys_discrete.D

    # States
    x_pos = x[0]
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    x_next = ca.vertcat(
        A_d[0,0]*x_pos + A_d[0,1]*x_dot + A_d[0,2]*theta + A_d[0,3]*theta_dot + B_d[0,0]*u,
        A_d[1,0]*x_pos + A_d[1,1]*x_dot + A_d[1,2]*theta + A_d[1,3]*theta_dot + B_d[1,0]*u,
        A_d[2,0]*x_pos + A_d[2,1]*x_dot + A_d[2,2]*theta + A_d[2,3]*theta_dot + B_d[2,0]*u,
        A_d[3,0]*x_pos + A_d[3,1]*x_dot + A_d[3,2]*theta + A_d[3,3]*theta_dot + B_d[3,0]*u,
    )
    return x_next


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
    run_prior_only = False
    run_prior_then_guidance = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_guide':
        run_prior_then_guidance = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
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
        dataset_class='InputsDataset',
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
    rng_x = np.linspace(-1,1,10) # 10 x_0 samples
    rng_theta = np.linspace(-np.pi/4,np.pi/4,10) # 10 theta_0 samples
    
    # all possible initial states combinations
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
           rng0.append([m,n])
    rng0 = np.array(rng0,dtype=float)

    # one initial state for test
    test = X0_IDX

    x_0 = rng0[test,0]
    x_0= round(x_0, 3)
    theta_0 = rng0[test,1]
    theta_0= round(theta_0, 3)


    #initial context
    x0 = np.array([[x_0 , 0, theta_0, 0]])  # np.array([[x_0 , 0, theta_0, 0]])  

    ############################################################################
    # sampling loop
    num_loop = ITERATIONS
    x_track = np.zeros((4, num_loop+1))
    u_track = np.zeros((1, num_loop))
    u_horizon_track = np.zeros((num_loop, HORIZON))

    x_track[:,0] = x0

    for i in range(0, num_loop):
        x0 = torch.tensor(x0).to(device) # load data to cuda

        hard_conds = None
        context = dataset.normalize_condition(x0)
        context_weight = WEIGHT_GUIDANC

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
        with TimerCUDA() as timer_model_sampling:
            inputs_normalized_iters = model.run_CFG(
                context, hard_conds, context_weight,
                n_samples=n_samples, horizon=n_support_points,
                return_chain=True,
                sample_fn=ddpm_cart_pole_sample_fn,
                n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            )
        print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
        # t_total = timer_model_sampling.elapsed

        ########
        inputs_iters = dataset.unnormalize_states(inputs_normalized_iters)

        inputs_final = inputs_iters[-1]
        print(f'control_inputs -- {inputs_final}')

        print(f'\n--------------------------------------\n')
        
        x0 = x0.cpu() # copy cuda tensor at first to cpu
        x0_array = np.squeeze(x0.numpy()) # matrix (1*4) to vector (4)
        horizon_inputs = np.zeros((1, HORIZON))
        inputs_final = inputs_final.cpu()
        for n in range(0,8):
            horizon_inputs[0,n] = round(inputs_final[0,n,0].item(),4)
        print(f'horizon_inputs -- {horizon_inputs}')
        applied_input = round(inputs_final[0,0,0].item(),4) # retain 4 decimal places
        print(f'applied_input -- {applied_input}')

        # save the control input from diffusion sampling
        u_track[:,i] = applied_input
        u_horizon_track[i,:] = horizon_inputs

        # update cart pole state
        x_next = cart_pole_dynamics(x0_array, applied_input)
        print(f'x_next-- {x_next}')
        x0 = np.array(x_next)
        x0 = x0.T # transpose matrix

        # save the new state
        x_track[:,i+1] = x0

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
    rng_x = np.linspace(-1,1,10) 
    rng_theta = np.linspace(-np.pi/4,np.pi/4,10) 
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
    x_0= round(x_0, 3)
    theta_0 = rng0[test,1]
    theta_0= round(theta_0, 3)

    #save the initial states
    x0 = np.array([x_0, 0, theta_0, 0])  # Initial states
    print(f'x0-- {x0}')
    x_mpc_track[:,0] = x0

    ############# control loop ##################
    for i in range(0, num_loop):
        # casadi_Opti
        optimizer = ca.Opti()

        # x and u mpc prediction along N
        X_pre = optimizer.variable(4, N + 1) 
        print(X_pre)
        U_pre = optimizer.variable(1, N) 

        optimizer.subject_to(X_pre[:, 0] == x0)  # starting state

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
        x0 = X_sol[:,1]
        print(f'x0_new-- {x0}')
        x_mpc_track[:,i+1] = x0

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
    figure_name = 'w_' + str(WEIGHT_GUIDANC) + 'x0_' + str(X0_IDX) + 'steps_' + str(ITERATIONS) + '.png'
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