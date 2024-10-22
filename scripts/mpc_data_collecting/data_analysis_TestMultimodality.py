import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


IDX_SELECT_INI_GROUP = 0
NUM_PLOT_IN_ONE_GROUP = 10

HOR = 64
NUM_INITIAL_X = 10 #10
NUM_INITIAL_THETA = 15 #15
NOISE_NUM = 15 #15
CONTROL_STEP = 50
INITIAL_GUESS_NUM = 1
PLOT_NORMAL = 0
PLOT_NOISE = 1


MODE_POS_THRESHOLD = 3.5
MODE_NEG_THRESHOLD = 2.65
FILTER_MAX_U = 5000

RESULT_PATH = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/Random_also_noisedata_decayguess1_112500_ulimit6000/'

INITIAL_STATE_GROUP = NUM_INITIAL_THETA * NUM_INITIAL_X
IDX_THETA = 2
IDX_NORMAL_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * CONTROL_STEP
NUM_SINGLE_THETA_X_CONTROL_STEP = 750

IDX_NOISE_START = INITIAL_GUESS_NUM * INITIAL_STATE_GROUP * CONTROL_STEP 
IDX_START_GROUP = CONTROL_STEP*NOISE_NUM 
IDX_NOISE_START_AFTER_INITIALGUESS_GROUP = INITIAL_STATE_GROUP * NOISE_NUM * CONTROL_STEP


filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INITIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx
FIG_NAME = 'ShowDataMultimodality.png'


def load_tensor_from_file(file_path):
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    return tensor

def access_tensor_element(tensor, indices):
    # Navigate through the tensor using the given indices
    return tensor[indices]

def findMaxMinList(DataList, nStep):
# DataList is [[1,2,3,...],[4,5,6,...],[]...], return [1,2,3,4,...], [1,2,3,4,...]
    max_data_list = []
    min_data_list = []
    for idx_step in range(nStep):
        step_values = [group[idx_step] for group in DataList]
        max_data_list.append(max(step_values))
        min_data_list.append(min(step_values))
    return max_data_list, min_data_list

def PlotMultiModality(Mode1Data, Mode1MaxData, Mode1MinData, 
                      Mode2Data, Mode2MaxData, Mode2MinData, 
                      Mode1_dark_col, Mode1_light_col, 
                      Mode2_dark_col, Mode2_light_col,
                      Xlabel, YLabel, Title):
    # pos mode
    for plot_data in Mode1Data:
        plt.plot(step, plot_data, color=Mode1_dark_col)
        
    # Fill the area between the min and max values
    plt.fill_between(step, Mode1MinData, Mode1MaxData, color=Mode1_light_col, alpha=0.5, label='Range')

    # Plot min and max as lines
    plt.plot(step, Mode1MinData, label="Min", color=Mode1_dark_col, linestyle='--')
    plt.plot(step, Mode1MaxData, label="Max", color=Mode1_dark_col, linestyle='--')
    
    # pos mode
    for plot_data in Mode2Data:
        plt.plot(step, plot_data, color=Mode2_dark_col)
        
    # Fill the area between the min and max values
    plt.fill_between(step, Mode2MinData, Mode2MaxData, color=Mode2_light_col, alpha=0.5, label='Range')

    # Plot min and max as lines
    plt.plot(step, Mode2MinData, label="Min", color=Mode2_dark_col, linestyle='--')
    plt.plot(step, Mode2MaxData, label="Max", color=Mode2_dark_col, linestyle='--')
    
    # label
    plt.xlabel(Xlabel)
    plt.ylabel(YLabel)
    plt.grid()
    plt.title(Title)
    

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

    
    NumTooBig_noise_u = 0
    u_nor_all = []
    u_nor_mode_pos = []
    u_nor_mode_neg = []
    j_nor_all = []
    j_nor_mode_pos = []
    j_nor_mode_neg = []
    theta_nor_all = []
    theta_nor_mode_pos = []
    theta_nor_mode_neg = []
    
    u_noise_all = []
    u_noise_mode_pos = []
    u_noise_mode_neg = []
    j_noise_all = []
    j_noise_mode_pos = []
    j_noise_mode_neg = []
    theta_noise_all = []
    theta_noise_mode_pos = []
    theta_noise_mode_neg = []
    
    for i in range(NUM_PLOT_IN_ONE_GROUP):
        ################## normal ####################################################################################################
        nStartIdx_nor = (i*NUM_INITIAL_THETA + IDX_SELECT_INI_GROUP) * CONTROL_STEP
        nEndIdx_nor = nStartIdx_nor + CONTROL_STEP
        theta_list_control_loop = tensor_x[nStartIdx_nor:nEndIdx_nor, IDX_THETA].tolist()
        u_list_control_loop = tensor_u[nStartIdx_nor:nEndIdx_nor, 0, 0].tolist()
        j_list_control_loop = tensor_j[nStartIdx_nor:nEndIdx_nor].tolist()
        
        if (max(abs(x) for x in u_list_control_loop) < FILTER_MAX_U): # only plot meaningful u
            theta_nor_all.append(theta_list_control_loop)
            u_nor_all.append(u_list_control_loop)
            j_nor_all.append(j_list_control_loop)    
        
        
        ################### noise #####################################################################################################
        theta_noise_for_single_state = []
        u_noise_for_single_state = []
        j_noise_for_single_state = []
        for k in range(NOISE_NUM):
            theta_noise_list_control_loop = []
            u_noise_list_control_loop = []
            j_noise_list_control_loop = []
            for m in range(CONTROL_STEP):
                nIdx = IDX_NOISE_START + (i*NUM_INITIAL_THETA + IDX_SELECT_INI_GROUP) * CONTROL_STEP * NOISE_NUM + k + NOISE_NUM*m
                theta_noise_list_control_loop = theta_noise_list_control_loop + [tensor_x[nIdx, IDX_THETA].tolist()]
                u_noise_list_control_loop = u_noise_list_control_loop + [tensor_u[nIdx, 0,0].tolist()]
                j_noise_list_control_loop = j_noise_list_control_loop + [tensor_j[nIdx].tolist()]
                if tensor_u[nIdx, 0,0].abs() > FILTER_MAX_U:
                    print("too big! (idx_group,idx_noise_num,control_step)=(",i,k,m,")=",tensor_u[nIdx, 0,0])
                    NumTooBig_noise_u += 1
                    
            
            if (max(abs(x) for x in u_noise_list_control_loop) < FILTER_MAX_U): # only plot meaningful u
                theta_noise_for_single_state.append(theta_noise_list_control_loop)
                u_noise_for_single_state.append(u_noise_list_control_loop)
                j_noise_for_single_state.append(j_noise_list_control_loop)
                theta_noise_all.append(theta_noise_list_control_loop)
                u_noise_all.append(u_noise_list_control_loop)
                j_noise_all.append(j_noise_list_control_loop)
            else:
                #print("(idx_group,idx_noise_num)=(",j,k,")","abs(u) too big=",max(u_noise_list_control_loop, key=abs))
                pass
        
        ################ seperate mode ##################################################################################################
        if(theta_list_control_loop[CONTROL_STEP-1] > MODE_POS_THRESHOLD):
            theta_nor_mode_pos.append(theta_list_control_loop)
            theta_noise_mode_pos.append(theta_list_control_loop)
            theta_noise_mode_pos.extend(theta_noise_for_single_state)
            u_nor_mode_pos.append(u_list_control_loop)
            u_noise_mode_pos.append(u_list_control_loop)
            u_noise_mode_pos.extend(u_noise_for_single_state)
            j_nor_mode_pos.append(j_list_control_loop)
            j_noise_mode_pos.append(j_list_control_loop)
            j_noise_mode_pos.extend(j_noise_for_single_state)
        elif(theta_list_control_loop[CONTROL_STEP-1] < MODE_NEG_THRESHOLD):
            theta_nor_mode_neg.append(theta_list_control_loop)
            theta_noise_mode_neg.append(theta_list_control_loop)
            theta_noise_mode_neg.extend(theta_noise_for_single_state)
            u_nor_mode_neg.append(u_list_control_loop)
            u_noise_mode_neg.append(u_list_control_loop)
            u_noise_mode_neg.extend(u_noise_for_single_state)
            j_nor_mode_neg.append(j_list_control_loop)
            j_noise_mode_neg.append(j_list_control_loop)
            j_noise_mode_neg.extend(j_noise_for_single_state)
        else:
            print("no mode happen, last theta = ", theta_list_control_loop[CONTROL_STEP-1][IDX_THETA])
        
            
    print("NumTooBig_noise_u=",NumTooBig_noise_u)     
   
    
    max_theta_perstep_mode_pos = []
    min_theta_perstep_mode_pos = []
    max_u_perstep_mode_pos = []
    min_u_perstep_mode_pos = []
    max_j_perstep_mode_pos = []
    min_j_perstep_mode_pos = []
    
    max_theta_perstep_mode_neg = []
    min_theta_perstep_mode_neg = []
    max_u_perstep_mode_neg = []
    min_u_perstep_mode_neg = []
    max_j_perstep_mode_neg = []
    min_j_perstep_mode_neg = []
    
    # mode pos
    max_theta_perstep_mode_pos, min_theta_perstep_mode_pos = findMaxMinList(theta_noise_mode_pos, CONTROL_STEP)
    max_u_perstep_mode_pos, min_u_perstep_mode_pos = findMaxMinList(u_noise_mode_pos, CONTROL_STEP)
    max_j_perstep_mode_pos, min_j_perstep_mode_pos = findMaxMinList(j_noise_mode_pos, CONTROL_STEP)
    
    # mode negative
    max_theta_perstep_mode_neg, min_theta_perstep_mode_neg = findMaxMinList(theta_noise_mode_neg, CONTROL_STEP)
    max_u_perstep_mode_neg, min_u_perstep_mode_neg = findMaxMinList(u_noise_mode_neg, CONTROL_STEP)
    max_j_perstep_mode_neg, min_j_perstep_mode_neg = findMaxMinList(j_noise_mode_neg, CONTROL_STEP)
    

    
    plt.subplot(3, 1, 1)
    PlotMultiModality(theta_nor_mode_pos, max_theta_perstep_mode_pos, min_theta_perstep_mode_pos, 
                      theta_nor_mode_neg, max_theta_perstep_mode_neg, min_theta_perstep_mode_neg, 
                      'tab:blue', 'lightsteelblue', 'darkorange', 'bisque',
                      'Step','theta','state in dataset')
    
    
    plt.subplot(3, 1, 2)
    PlotMultiModality(u_nor_mode_pos, max_u_perstep_mode_pos, min_u_perstep_mode_pos, 
                      u_nor_mode_neg, max_u_perstep_mode_neg, min_u_perstep_mode_neg, 
                      'tab:blue', 'lightsteelblue', 'darkorange', 'bisque',
                      'Step','ctrl','control input in dataset')
    
    plt.subplot(3, 1, 3)
    PlotMultiModality(j_nor_mode_pos, max(j_nor_mode_pos), min(j_nor_mode_pos), 
                      j_nor_mode_neg, max(j_nor_mode_neg), min(j_nor_mode_neg), 
                      'tab:blue', 'lightsteelblue', 'darkorange', 'bisque',
                      'Step','cost','cost in dataset')

    
    plt.show()
    print("over-----------------------------------")
    fig_save_path = RESULT_PATH + FIG_NAME
    plt.savefig(fig_save_path)
