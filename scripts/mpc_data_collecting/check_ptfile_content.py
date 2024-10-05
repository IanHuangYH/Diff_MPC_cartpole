import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

HOR = 64

NUM_INIYIAL_THETA = 2

NUM_INITIAL_X = 2

INITIAL_GUESS_NUM = 2
INITIAL_STATE_GROUP = NUM_INIYIAL_THETA * NUM_INITIAL_X
NOISE_NUM = 20
CONTROL_STEP = 80
IDX_THETA = 2
IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * CONTROL_STEP

IDX_NOISE_START = INITIAL_GUESS_NUM * INITIAL_STATE_GROUP * CONTROL_STEP 
IDX_START_GROUP = CONTROL_STEP*NOISE_NUM 
IDX_NOISE_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * NOISE_NUM * CONTROL_STEP


filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INIYIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx

def load_tensor_from_file(file_path):
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    return tensor

def access_tensor_element(tensor, indices):
    # Navigate through the tensor using the given indices
    return tensor[indices]

if __name__ == "__main__":
    # Specify the path to your .pt file
    file_path = '/MPC_DynamicSys/sharedVol/train_data/nmpc/multi_normal/'
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
    print(f"Tensor u shape: {tensor_j.shape}")
    
    # Specify the index you want to navigate to
    step = list(range(CONTROL_STEP))  # Generates a sequence from 0 to CONTROL_STEP-1

    plt.figure(figsize=(20, 15))
    ##################################### x: guess = pos, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 1)
    
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            theta_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                theta_noise_list_control_loop = theta_noise_list_control_loop + [tensor_x[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m, IDX_THETA].tolist()]
            plt.plot(step, theta_noise_list_control_loop, 'm--')
    # normal
    for i in range(INITIAL_STATE_GROUP):
        theta_list_control_loop = tensor_x[i*CONTROL_STEP:i*CONTROL_STEP + CONTROL_STEP, IDX_THETA].tolist()
        plt.plot(step, theta_list_control_loop)
    
    plt.ylabel('theta (rad)')
    plt.grid()
    plt.title('guess=pos')
    
    
    ##################################### u: guess = pos, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 2)
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            u_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                u_noise_list_control_loop = u_noise_list_control_loop + [tensor_u[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m, 0,0].tolist()]
        plt.plot(step, u_noise_list_control_loop, 'm--')     
   
    # normal
    for i in range(INITIAL_STATE_GROUP):
        u_list_control_loop = tensor_u[i*CONTROL_STEP:i*CONTROL_STEP + CONTROL_STEP, 0, 0].tolist()
        plt.plot(step, u_list_control_loop)
        
    
    
    plt.ylabel('u')
    plt.grid()
    plt.title('guess=pos')
     
     
    ##################################### J: guess = pos, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 3)
    
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            j_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                j_noise_list_control_loop = j_noise_list_control_loop + [tensor_j[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m].tolist()]
            plt.plot(step, j_noise_list_control_loop, 'm--')
    # normal
    for i in range(INITIAL_STATE_GROUP):
        j_list_control_loop = tensor_j[i*CONTROL_STEP:i*CONTROL_STEP + CONTROL_STEP].tolist()
        plt.plot(step, j_list_control_loop)
    
    plt.ylabel('J')
    plt.grid()
    plt.title('guess=pos')   
    
    ##################################### X: guess = neg, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 4)
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            theta_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                theta_noise_list_control_loop = theta_noise_list_control_loop + [tensor_x[IDX_NOISE_START + IDX_NOISE_START_AFTER_INITIALGUESS_GROUP + j * IDX_START_GROUP + k + NOISE_NUM*m, IDX_THETA].tolist()]
            plt.plot(step, theta_noise_list_control_loop, 'm--')
    # normal
    for i in range(INITIAL_STATE_GROUP):
        theta_list_control_loop = tensor_x[IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP:IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP+i*CONTROL_STEP + CONTROL_STEP, IDX_THETA].tolist()
        plt.plot(step, theta_list_control_loop)
    
    plt.ylabel('theta (rad)')
    plt.grid()
    plt.title('guess=neg')
    
    
    
    ##################################### U: guess = neg, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 5)
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            u_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                u_noise_list_control_loop = u_noise_list_control_loop + [tensor_u[IDX_NOISE_START + IDX_NOISE_START_AFTER_INITIALGUESS_GROUP + j * IDX_START_GROUP + k + NOISE_NUM*m, 0,0].tolist()]
        plt.plot(step, u_noise_list_control_loop, 'm--')  
    
    # normal
    for i in range(INITIAL_STATE_GROUP):
        u_list_control_loop = tensor_u[IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP:IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP + CONTROL_STEP, 0, 0].tolist()
        plt.plot(step, u_list_control_loop)
    
    plt.ylabel('u')
    plt.grid()
    plt.title('guess=neg')
    
    ##################################### J: guess = neg, plot all initila state group, control step, noise ##################
    plt.subplot(6, 1, 6)
    # noise
    for j in range(INITIAL_STATE_GROUP):
        for k in range(NOISE_NUM):
            j_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                j_noise_list_control_loop = j_noise_list_control_loop + [tensor_j[IDX_NOISE_START + IDX_NOISE_START_AFTER_INITIALGUESS_GROUP + j * IDX_START_GROUP + k + NOISE_NUM*m].tolist()]
            plt.plot(step, j_noise_list_control_loop, 'm--')
    # normal
    for i in range(INITIAL_STATE_GROUP):
        j_list_control_loop = tensor_j[IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP:IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP+i*CONTROL_STEP + CONTROL_STEP].tolist()
        plt.plot(step, j_list_control_loop)
    
    plt.ylabel('theta (rad)')
    plt.grid()
    plt.title('guess=neg')
    
    plt.show()
    print("over-----------------------------------")
    plt.savefig('output.png')
