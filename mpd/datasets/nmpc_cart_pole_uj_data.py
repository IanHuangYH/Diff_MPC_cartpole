import abc
import os.path

import git
import numpy as np
import torch
from torch.utils.data import Dataset

from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml

INITILA_X = 10
INITIAL_THETA = 20
INITIAL_GUESS = 2
CONTROL_STEP = 80
NOISE_NUM = 20
HOR = 64
# training data amount
TRAINING_DATA_AMOUNT = INITILA_X * INITIAL_THETA * INITIAL_GUESS * CONTROL_STEP * (NOISE_NUM+1)

dataset_base_dir = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data' 

# Data Name Setting
filename_idx = '_ini_'+str(INITILA_X)+'x'+str(INITIAL_THETA)+'_noise_'+str(NOISE_NUM)+'_step_'+str(CONTROL_STEP)+'_hor_'+str(HOR)+'.pt'
X0_CONDITION_DATA_NAME = 'x0' + filename_idx
U_DATA_FILENAME = 'u' + filename_idx
J_DATA_FILENAME = 'j' + filename_idx

class NMPC_UJ_Dataset(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 **kwargs):

        self.tensor_args = tensor_args

        self.dataset_subdir = dataset_subdir
        self.base_dir = os.path.join(dataset_base_dir, self.dataset_subdir)

        # -------------------------------- Load inputs data ---------------------------------
        self.field_key_condition = 'condition'
        self.field_key_u = 'inputs'
        self.field_key_j = 'cost'
        self.field_key_uj = 'uj'
        self.fields = {}

        # load data to self.fields
        self.include_velocity = include_velocity
        self.load_inputs()

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_u].shape
        self.n_init = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.inputs_dim = (self.n_support_points, d)

        # normalize the inputs (for the diffusion model)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        
        # normalize cost
        self.cost_dic = {} # for initial the datanormalizer
        self.cost_dic[self.field_key_j] = self.fields[self.field_key_j]
        self.normalizer_j = DatasetNormalizer(self.cost_dic, normalizer='LogMinMaxNormalizer')
        
        # self.fields[self.field_key_condition] = self.condition
        self.normalizer_keys_xu = [self.field_key_u, self.field_key_condition]
        self.normalizer_keys_j = [self.field_key_j]
        self.normalize_all_data(*self.normalizer_keys_xu)

        self.normalize_cost_data(*self.normalizer_keys_j)
        del self.cost_dic
        print(self.fields[self.field_key_j])

    def load_inputs(self):
        # load training inputs
        check = self.tensor_args['device']
        print(f'tensor_device -- {check}')
        # load u and j
        u_load = torch.load(os.path.join(self.base_dir, U_DATA_FILENAME),map_location=self.tensor_args['device'])
        j_load = torch.load(os.path.join(self.base_dir, J_DATA_FILENAME),map_location=self.tensor_args['device'])
        u_load = u_load.float()
        j_load = j_load.float()
        print(f'u_training -- {u_load.shape}')
        print(f'j_training -- {j_load.shape}')
        self.fields[self.field_key_u] = u_load
        self.fields[self.field_key_j] = j_load
    
        

        # x0 condition
        x0_condition =  torch.load(os.path.join(self.base_dir, X0_CONDITION_DATA_NAME),map_location=self.tensor_args['device'])
        x_xdot = x0_condition[:,0:2]
        theta_transform = x0_condition[:,4].unsqueeze(1)
        theta_dot = x0_condition[:,3].unsqueeze(1)
        x0_condition_reduce_theta = torch.cat((x_xdot, theta_transform, theta_dot), dim=1)
        x0_condition_final = x0_condition_reduce_theta.float()
        self.fields[self.field_key_condition] = x0_condition_final
        print(f'condition_list_dimension -- {len(x0_condition_final),len(x0_condition_final[0])}')
        print(f'fields -- {len(self.fields)}')
        
        

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)
            
    def normalize_cost_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer_j(self.fields[f'{key}'], key)

    # only normalize data u but not the condition texts
    def normalize_u_data(self, *keys):
        for key in keys:
            if key == self.field_key_u:
                self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)
            else:
                self.fields[f'{key}_normalized'] = self.fields[f'{key}']


    def __repr__(self):
        msg = f'NMPC_UJ_Dataset\n' \
              f'n_init: {self.n_init}\n' \
              f'inputs_dim: {self.inputs_dim}\n'
        return msg

    def __len__(self):
        return self.n_init

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        field_u_normalized = f'{self.field_key_u}_normalized'
        field_j_normalized = f'{self.field_key_j}_normalized'
        field_uj_normalized = f'{self.field_key_uj}_normalized'
        field_condition_normalized = f'{self.field_key_condition}_normalized'
        
        # cat u, j
        uj_normalized = self.fields[field_u_normalized][index]
        j_normalized = self.fields[field_j_normalized][index]
        uj_normalized[-1,0] = j_normalized
        
        condition_normalized = self.fields[field_condition_normalized][index]

        data = {
            field_uj_normalized: uj_normalized,
            field_condition_normalized: condition_normalized
        }

        return data

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError

    def get_unnormalized(self, index):
        raise NotImplementedError
        traj = self.fields[self.field_key_traj][index][..., :self.state_dim]
        task = self.fields[self.field_key_task][index]
        if not self.include_velocity:
            task = task[self.task_idxs]
        data = {self.field_key_traj: traj,
                self.field_key_task: task,
                }
        if self.variable_environment:
            data.update({self.field_key_env: self.fields[self.field_key_env][index]})

        # hard conditions
        # hard_conds = self.get_hard_conds(tasks)
        hard_conds = self.get_hard_conditions(traj)
        data.update({'hard_conds': hard_conds})

        return data

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_states(self, x):
        return self.unnormalize(x, self.field_key_u)

    def normalize_states(self, x):
        return self.normalize(x, self.field_key_u)

    def unnormalize_condition(self, x):
        return self.unnormalize(x, self.field_key_condition)

    def normalize_condition(self, x):
        return self.normalize(x, self.field_key_condition)