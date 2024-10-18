import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HOR = 64

NUM_INITIAL_X = 10 #10
NUM_INIYIAL_THETA = 15 #15
NOISE_NUM = 15 #15
CONTROL_STEP = 50
INITIAL_GUESS_NUM = 1
PLOT_NORMAL = 1
PLOT_NOISE = 1

FILTER_MAX_U = 1000000

RESULT_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/Random_also_noisedata_decayguess1_112500_ulimit5000/'

INITIAL_STATE_GROUP = NUM_INIYIAL_THETA * NUM_INITIAL_X
IDX_THETA = 2
IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * CONTROL_STEP

IDX_NOISE_START = INITIAL_GUESS_NUM * INITIAL_STATE_GROUP * CONTROL_STEP 
IDX_START_GROUP = CONTROL_STEP*NOISE_NUM 
IDX_NOISE_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * NOISE_NUM * CONTROL_STEP


filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INIYIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx
FIG_NAME = 'DataCheck.png'


def load_tensor_from_file(file_path):
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    return tensor

def access_tensor_element(tensor, indices):
    # Navigate through the tensor using the given indices
    return tensor[indices]

if __name__ == "__main__":
    # Specify the path to your .pt file
    file_path = RESULT_PATH
    file_name_x = X0_CONDITION_DATA_NAME
    file_name_u = U_DATA_NAME
    file_name_j = J_DATA_NAME
    complete_file_path_x = file_path + file_name_x
    complete_file_path_u = file_path + file_name_u
    complete_file_path_j = file_path + file_name_j
    
    # Load the tensor
    tensor_x : torch.tensor = load_tensor_from_file(complete_file_path_x)
    tensor_u : torch.tensor = load_tensor_from_file(complete_file_path_u)
    tensor_j : torch.tensor = load_tensor_from_file(complete_file_path_j)
    
    # Print some information about the tensor
    print(f"Tensor x shape: {tensor_x.shape}")
    print(f"Tensor u shape: {tensor_u.shape}")
    print(f"Tensor j shape: {tensor_j.shape}")
    
    # Specify the index you want to navigate to
    step = list(range(CONTROL_STEP))  # Generates a sequence from 0 to CONTROL_STEP-1

    plt.figure(figsize=(20, 15))
    ##################################### x: guess = pos, plot all initila state group, control step, noise ##################
    
    
    ##################################### u: guess = pos, plot all initila state group, control step, noise ##################
    NumTooBig_noise_u = 0
    # # noise
    if PLOT_NOISE == 1:
        for j in range(INITIAL_STATE_GROUP):
            for k in range(NOISE_NUM):
                theta_noise_list_control_loop = []
                u_noise_list_control_loop = []
                j_noise_list_control_loop = []
                for m in range(CONTROL_STEP):
                    nIdx = IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m
                    theta_noise_list_control_loop = theta_noise_list_control_loop + [tensor_x[nIdx, IDX_THETA].tolist()]
                    u_noise_list_control_loop = u_noise_list_control_loop + [tensor_u[nIdx, 0,0].tolist()]
                    j_noise_list_control_loop = j_noise_list_control_loop + [tensor_j[nIdx].tolist()]
                    if tensor_u[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m, 0,0].abs() > FILTER_MAX_U:
                        print("too big! (idx_group,idx_noise_num,control_step)=(",j,k,m,")=",tensor_u[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m, 0,0])
                        NumTooBig_noise_u += 1
                if (max(abs(x) for x in u_noise_list_control_loop) < FILTER_MAX_U): # only plot meaningful u
                    plt.subplot(3, 1, 1)
                    plt.plot(step, theta_noise_list_control_loop, 'm--')
                    plt.subplot(3, 1, 2)
                    plt.plot(step, u_noise_list_control_loop, 'm--')
                    plt.subplot(3, 1, 3)
                    plt.plot(step, j_noise_list_control_loop, 'm--')
                else:
                    #print("(idx_group,idx_noise_num)=(",j,k,")","abs(u) too big=",max(u_noise_list_control_loop, key=abs))
                    pass
    print("NumTooBig_noise_u=",NumTooBig_noise_u)     
   
    # normal
    if PLOT_NORMAL == 1:
        for i in range(INITIAL_STATE_GROUP):
            nStartIdx = i*CONTROL_STEP
            nEndIdx = nStartIdx + CONTROL_STEP
            theta_list_control_loop = tensor_x[nStartIdx:nEndIdx, IDX_THETA].tolist()
            u_list_control_loop = tensor_u[nStartIdx:nEndIdx, 0, 0].tolist()
            j_list_control_loop = tensor_j[nStartIdx:nEndIdx].tolist()
            
            if (max(abs(x) for x in u_list_control_loop) < FILTER_MAX_U): # only plot meaningful u
                plt.subplot(3, 1, 1)
                plt.plot(step, theta_list_control_loop)
                plt.subplot(3, 1, 2)
                plt.plot(step, u_list_control_loop)
                plt.subplot(3, 1, 3)
                plt.plot(step, j_list_control_loop)
            else:
                #print("(idx_group,idx_noise_num)=(",j,k,")","abs(u) too big=",max(u_noise_list_control_loop, key=abs))
                pass
    
    
    plt.subplot(3, 1, 1)
    plt.ylabel('theta')
    plt.grid()
    plt.title('result for random initial guess')
    
    plt.subplot(3, 1, 2)
    plt.ylabel('u')
    plt.grid()
    plt.title('result for random initial guess')
    
    plt.subplot(3, 1, 3)
    plt.ylabel('j')
    plt.grid()
    plt.title('result for random initial guess')

    
    plt.show()
    print("over-----------------------------------")
    fig_save_path = RESULT_PATH + FIG_NAME
    plt.savefig(fig_save_path)
