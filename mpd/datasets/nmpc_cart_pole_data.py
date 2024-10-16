import abc
import os.path

import git
import numpy as np
import torch
from torch.utils.data import Dataset

from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml


class NMPC_Dataset(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 j_normalizer='LimitsNormalizer',
                 ux_normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 dataset_base_dir = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC/',
                 u_filename=None,
                 j_filename=None,
                 x0_filename=None, 
                 **kwargs):

        self.tensor_args = tensor_args

        self.dataset_subdir = dataset_subdir
        self.base_dir = dataset_base_dir

        # -------------------------------- Load inputs data ---------------------------------

        self.field_key_condition = 'condition'
        self.field_key_inputs = 'inputs'
        self.fields = {}

        # load data
        self.include_velocity = include_velocity
        self.load_inputs(u_filename, x0_filename)

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_inputs].shape
        self.n_init = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.inputs_dim = (self.n_support_points, d)

        # normalize the inputs (for the diffusion model)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=ux_normalizer)
        # self.fields[self.field_key_condition] = self.condition
        self.normalizer_keys = [self.field_key_inputs, self.field_key_condition] # [self.field_key_u, self.field_key_task]
        self.normalize_all_data(*self.normalizer_keys)

    def load_inputs(self, u_filename:str, x0_filename:str):
        # load training inputs
        check = self.tensor_args['device']
        print(f'tensor_device -- {check}')
        u_load = torch.load(os.path.join(self.base_dir, u_filename),map_location=self.tensor_args['device'])
        u_load = u_load.float()
        print(f'inputs_training -- {u_load.shape}')
    
        self.fields[self.field_key_inputs] = u_load

        # x0 condition
        x0_condition =  torch.load(os.path.join(self.base_dir, x0_filename),map_location=self.tensor_args['device'])
        x_xdot = x0_condition[:,0:2]
        theta_transform = x0_condition[:,4].unsqueeze(1)
        theta_dot = x0_condition[:,3].unsqueeze(1)
        x0_condition_reduce_theta = torch.cat((x_xdot, theta_transform, theta_dot), dim=1)
        x0_condition_reduce_theta = x0_condition_reduce_theta.float()
        
        self.fields[self.field_key_condition] = x0_condition_reduce_theta
        print(f'condition_list_dimension -- {len(x0_condition_reduce_theta),len(x0_condition_reduce_theta[0])}')
        print(f'fields -- {len(self.fields)}')

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)

    # only normalize data u but not the condition texts
    def normalize_u_data(self, *keys):
        for key in keys:
            if key == self.field_key_inputs:
                self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)
            else:
                self.fields[f'{key}_normalized'] = self.fields[f'{key}']


    def __repr__(self):
        msg = f'NMPC_Dataset\n' \
              f'n_init: {self.n_init}\n' \
              f'inputs_dim: {self.inputs_dim}\n'
        return msg

    def __len__(self):
        return self.n_init

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        field_inputs_normalized = f'{self.field_key_inputs}_normalized'
        field_condition_normalized = f'{self.field_key_condition}_normalized'
        inputs_normalized = self.fields[field_inputs_normalized][index]
        # print(f'inputs_training -- {inputs_normalized}')
        condition_normalized = self.fields[field_condition_normalized][index]
        # print(f'condition_normalized -- {condition_normalized}')
        data = {
            field_inputs_normalized: inputs_normalized,
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
        return self.unnormalize(x, self.field_key_inputs)

    def normalize_states(self, x):
        return self.normalize(x, self.field_key_inputs)

    def unnormalize_condition(self, x):
        return self.unnormalize(x, self.field_key_condition)

    def normalize_condition(self, x):
        return self.normalize(x, self.field_key_condition)