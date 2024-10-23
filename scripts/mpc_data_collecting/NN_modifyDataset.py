import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# data saving folder
folder_path = "/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/Random_also_noisedata_decayguess1_112500"
CONTROL_STEP = 50
HOR = 64
NN_INITILA_X = 10 #10, 5
NN_INITIAL_THETA = 15 #15, 6
NN_NOISE_NUM = 15 #15, 3
NN_filename_idx = '_ini_'+str(NN_INITILA_X)+'x'+str(NN_INITIAL_THETA)+'_noise_'+str(NN_NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
NN_X0_CONDITION_DATA_NAME = 'x0' + NN_filename_idx
NN_OUTPUT_NAME = 'x0_4DOF_' + NN_filename_idx


# x0 loading 
X_path = os.path.join(folder_path,NN_X0_CONDITION_DATA_NAME)
X_output_Path = os.path.join(folder_path,NN_OUTPUT_NAME)
x0_5 = torch.load(X_path) 
print(f'x0_120000 -- {x0_5.size()}')
# x0_6400 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/x0-tensor_6400-4.pt") 

# Copy data from the fifth column (index 4) to the third column (index 2)
x0_5[:, 2] = x0_5[:, 4]
# Remove the fifth column (index 4) by slicing
x0_4= x0_5[:, :4]  # This keeps only the first four columns
print(f'x0_4 -- {x0_4.size()}')



# save
torch.save(x0_4, X_output_Path)
