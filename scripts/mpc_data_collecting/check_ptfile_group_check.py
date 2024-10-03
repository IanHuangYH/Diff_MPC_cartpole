import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

INITIAL_GUESS_NUM = 2
INITIAL_STATE_GROUP = 1
NOISE_NUM = 2
CONTROL_STEP = 4
IDX_THETA = 2
IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * CONTROL_STEP

IDX_NOISE_START = INITIAL_GUESS_NUM * INITIAL_STATE_GROUP * CONTROL_STEP 
IDX_START_GROUP = CONTROL_STEP*NOISE_NUM 
IDX_NOISE_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * NOISE_NUM * CONTROL_STEP

GROUP_NUM = (NOISE_NUM + 1) * CONTROL_STEP

def load_tensor_from_file(file_path):
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    return tensor

def access_tensor_element(tensor, indices):
    # Navigate through the tensor using the given indices
    return tensor[indices]

if __name__ == "__main__":
    # Specify the path to your .pt file
    file_path = '/MPC_DynamicSys/sharedVol/train_data/nmpc/test2/'
    file_name_x = 'x0_ini_10x20_noise_20_step_80_hor_64.pt'
    file_name_u = 'u_ini_10x20_noise_20_step_80_hor_64.pt'
    file_name_j = 'J_ini_10x20_noise_20_step_80_hor_64.pt'
    
    x_list = []
    u_list = []
    j_list = []
    x_normal = list()
    x_noise = list()
    u_normal = list()
    u_noise = list()
    j_normal = list()
    j_noise = list()
    for j in range(INITIAL_GUESS_NUM):
        for i in range(INITIAL_STATE_GROUP):
            groupfilename = 'guess_' + str(j) + '_ini_' + str(i) + '_'
            complete_file_path_x = file_path + groupfilename + file_name_x
            complete_file_path_u = file_path + groupfilename + file_name_u
            complete_file_path_j = file_path + groupfilename + file_name_j
            x_list.append(load_tensor_from_file(complete_file_path_x))
            u_list.append(load_tensor_from_file(complete_file_path_u))
            j_list.append(load_tensor_from_file(complete_file_path_j))
            x_normal.append(x_list[i][0:CONTROL_STEP][:])
            x_noise.append(x_list[i][CONTROL_STEP:CONTROL_STEP*(1+NOISE_NUM)][:])
            u_normal.append(u_list[i][0:CONTROL_STEP][:][:])
            u_noise.append(u_list[i][CONTROL_STEP:CONTROL_STEP*(1+NOISE_NUM)][:][:])
            j_normal.append(j_list[i][0:CONTROL_STEP])
            j_noise.append(j_list[i][CONTROL_STEP:CONTROL_STEP*(1+NOISE_NUM)])
    
    tensor_x = torch.cat([torch.cat(x_normal,dim=0), torch.cat(x_noise,dim=0)],dim=0) 
    tensor_u = torch.cat([torch.cat(u_normal,dim=0), torch.cat(u_noise,dim=0)],dim=0)
    tensor_j = torch.cat([torch.cat(j_normal,dim=0), torch.cat(j_noise,dim=0)],dim=0)

    
    
    # Print some information about the tensor
    print(f"Tensor x shape: {tensor_x.shape}")
    print(f"Tensor u shape: {tensor_u.shape}")
    print(f"Tensor J shape: {tensor_j.shape}")
    
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
            j_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                j_noise_list_control_loop = j_noise_list_control_loop + [tensor_u[IDX_NOISE_START + j * IDX_START_GROUP + k + NOISE_NUM*m, 0,0].tolist()]
        plt.plot(step, j_noise_list_control_loop, 'm--')     
   
    # normal
    for i in range(INITIAL_STATE_GROUP):
        j_list_control_loop = tensor_u[i*CONTROL_STEP:i*CONTROL_STEP + CONTROL_STEP, 0, 0].tolist()
        plt.plot(step, j_list_control_loop)
        
    
    
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
        j_list_control_loop = tensor_j[IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP:IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP + i*CONTROL_STEP + CONTROL_STEP].tolist()
        plt.plot(step, j_list_control_loop)
    
    plt.ylabel('J')
    plt.grid()
    plt.title('guess=neg')
    
    plt.show()
    print("over-----------------------------------")
