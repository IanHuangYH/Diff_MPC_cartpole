from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.torch_utils import get_torch_device
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS
import os
import torch
import numpy as np

# modify
DATASET = 'NMPC_UJ_Dataset'
J_NORMALIZER = 'LogZScoreNormalizer' #GaussianNormalizer, LimitsNormalizer, LogMinMaxNormalizer, LogZScoreNormalizer, OnlyLogNormalizer, GaussianMinMaxNormalizer
UX_NORMALIZER = 'LimitsNormalizer'
MODEL_FOLDER = 'nmpc_batch_4096_random112500_float64_minmax_xu_logzscore_j_decay1_ulimit6000'
DATA_FOLDER = 'Random_also_noisedata_decayguess1_112500_ulimit6000'

DEBUG = 1
model_list = [16000]



# path
DATA_LOAD_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/'+DATA_FOLDER
RESULT_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/model_performance_saving/'
MODEL_SAVED_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+MODEL_FOLDER
MODEL_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/data_trained_models/'+str(MODEL_FOLDER)+'/'+ str(model_list[0])

INITILA_X = 10
INITIAL_THETA = 15
CONTROL_STEP = 50
NOISE_NUM = 15
HOR = 64

# MPC parameters
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
R = np.diag([0.001])
P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])
HORIZON = 63 # mpc horizon
TS = 0.01

NUM_FIND_GLOBAL = 5
# fix_random_seed(40)

WEIGHT_GUIDANC = 0.01 # non-conditioning weight

# Data Name Setting
filename_idx = '_ini_'+str(INITILA_X)+'x'+str(INITIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
X_DATA_FILENAME = 'x0' + filename_idx
U_DATA_FILENAME = 'u' + filename_idx
J_DATA_FILENAME = 'j' + filename_idx

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
    # stage cost
    for i in range(1,num_hor):
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

def SortCost(CostArray):
    pass

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

def get_sorted_ranks(CostList):
    # Get the sorted indices (sorted by value) and add 1 to make ranks start from 1
    sorted_indices = sorted(range(len(CostList)), key=lambda x: CostList[x])
    
    # Create a rank list, where rank[i] gives the rank of lst[i]
    ranks = [0] * len(CostList)
    for rank, index in enumerate(sorted_indices):
        ranks[index] = rank

    return ranks


##################################################### Load traindata (x,u -> j) + normalize data #######################################
print("---------------------------------load data start--------------------------------")
device = 'cuda'
device = get_torch_device(device)

tensor_args = {'device': device, 'dtype': torch.float64}
args = load_params_from_yaml(os.path.join(MODEL_PATH, "args.yaml"))
args.pop('j_filename', None)
args.pop('u_filename', None)
args.pop('x0_filename', None)
args.pop('train_data_load_path', None)
args.pop('batch_size',None)

train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
    dataset_class=DATASET,
    batch_size = 1,
    j_normalizer=J_NORMALIZER,
    train_data_load_path = DATA_LOAD_PATH,
    j_filename=J_DATA_FILENAME,
    u_filename = U_DATA_FILENAME,
    x0_filename = X_DATA_FILENAME,
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
print("---------------------------------load data finished--------------------------------")

######################################################## Load model ##############################################################33
# diffusion setting
# Load prior model
print("---------------------------------load model start--------------------------------")
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
    torch.load(os.path.join(MODEL_PATH, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
    map_location=tensor_args['device'])
)
diffusion_model.eval()
model = diffusion_model

model = torch.compile(model)
print("---------------------------------load model finished--------------------------------")

########################################### input normalize train data to diffusion model, output j ###################################
print("---------------------------------test prediction start--------------------------------")

MSE_j = torch.zeros(0,device=device)

x_data = torch.load(os.path.join(DATA_LOAD_PATH, X_DATA_FILENAME), map_location=device)
u_data = torch.load(os.path.join(DATA_LOAD_PATH, U_DATA_FILENAME), map_location=device)
j_data = torch.load(os.path.join(DATA_LOAD_PATH, J_DATA_FILENAME), map_location=device)


Data_Num = x_data.shape[0]
FindGlobal_list = []
for i in range(NUM_FIND_GLOBAL):
    tensor = torch.randn(1, n_support_points, 1)  # Create a 1x64x1 tensor
    FindGlobal_list.append([0, tensor])
    
for i in range(0,Data_Num):
    x_red = x_data[i,:]
    if DEBUG:
        x_red[0] = -0.47
        x_red[1] = 0
        x_red[2] = 3*np.pi/4
        x_red[3] = 0
    theta_red = ThetaToRedTheta(x_red[2])
    
        
    
    x_GT_unnor = torch.tensor([x_red[0], x_red[1], theta_red, x_red[3]], device=device)
    u_GT_unnor = u_data[i,:,0]
    j_GT_unnor = j_data[i]
    
    x_GT_nor = dataset.normalize_condition(x_GT_unnor)
    u_GT_nor = dataset.normalize_states(u_GT_unnor)
    j_GT_nor = dataset.normalize_cost(j_GT_unnor)
    
    J_GT_List = []
    J_predic_List = []
    for j in range(5):
        u_normalized_iters = model.run_CFG(
                        x_GT_nor, None, WEIGHT_GUIDANC,
                        n_samples=1, horizon=n_support_points,
                        return_chain=True,
                        sample_fn=ddpm_cart_pole_sample_fn,
                        n_diffusion_steps_without_noise=5,
                    )
        
        u_iters = dataset.unnormalize_states(u_normalized_iters[:,:,0:Num_Hor_u,:])
        u_pred_nor = u_normalized_iters[:,:,0:Num_Hor_u,:][-1]
        u_predict_unnor = u_iters[-1]
        
        j_predict_nor =u_normalized_iters[-1,0,-1,0]
        j_predict_unnor = dataset.unnormalize_cost(j_predict_nor)
        
        FindGlobal_list[j][0] = j_predict_unnor
        FindGlobal_list[j][1] = u_predict_unnor
        
        
        u_predict_unnor = u_predict_unnor.cpu()
        x_red = x_red.cpu()
        x0 = x_red.numpy()
        
        J_predic_List.append(j_predict_unnor)
        J_GT_List.append(calMPCCost(Q,R,P,u_predict_unnor,x0, EulerForwardCartpole_virtual, TS))
        
    RankOfGTCost = get_sorted_ranks(J_GT_List) 
    RankOfPredictCost = get_sorted_ranks(J_predic_List)

    error_nor = j_GT_nor - j_predict_nor
    error_unnor = j_GT_unnor - j_predict_unnor
    MSE_j = MSE_j + (error_nor) ** 2

MSE_j = MSE_j.sqrt()
print("---------------------------------test prediction end--------------------------------")

# predicted j - ground true j