import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

INITIAL_GUESS_NUM = 2
INITIAL_STATE_GROUP = 4
NOISE_NUM = 2
CONTROL_STEP = 80
IDX_THETA = 2
IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * CONTROL_STEP

IDX_NOISE_START = INITIAL_GUESS_NUM * INITIAL_STATE_GROUP * CONTROL_STEP 
IDX_START_GROUP = CONTROL_STEP*NOISE_NUM 
IDX_NOISE_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * NOISE_NUM * CONTROL_STEP

def load_tensor_from_file(file_path):
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    return tensor

def access_tensor_element(tensor, indices):
    # Navigate through the tensor using the given indices
    return tensor[indices]

if __name__ == "__main__":
    # Specify the path to your .pt file
    file_path = '/MPC_DynamicSys/sharedVol/train_data/nmpc/1st/'
    file_name_x = 'x0_ini_20x10_noise_20_step_80_hor_40.pt'
    file_name_u = 'u_ini_20x10_noise_20_step_80_hor_40.pt'
    complete_file_path_x = file_path + file_name_x
    complete_file_path_u = file_path + file_name_u
    
    # Load the tensor
    tensor_x : torch.tensor = load_tensor_from_file(complete_file_path_x)
    tensor_u : torch.tensor = load_tensor_from_file(complete_file_path_u)
    
    # Print some information about the tensor
    print(f"Tensor x shape: {tensor_x.shape}")
    print(f"Tensor u shape: {tensor_u.shape}")
    
    # Specify the index you want to navigate to
    step = list(range(CONTROL_STEP))  # Generates a sequence from 0 to CONTROL_STEP-1

    plt.figure(figsize=(20, 15))
    ##################################### x: guess = pos, plot all initila state group, control step, noise ##################
    plt.subplot(4, 1, 1)
    
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
    plt.subplot(4, 1, 2)
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
        
    
    ##################################### X: guess = neg, plot all initila state group, control step, noise ##################
    plt.subplot(4, 1, 3)
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
    plt.subplot(4, 1, 4)
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
    
    plt.show()
    print("over-----------------------------------")