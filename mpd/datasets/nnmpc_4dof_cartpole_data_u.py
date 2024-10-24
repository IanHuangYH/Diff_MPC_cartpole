import abc
import os.path

import git
import numpy as np
import torch
from torch.utils.data import Dataset

from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml

class NMPC_4DOF_DATASET(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 dataset_base_dir = None,
                 u_filename=None,
                 x0_filename=None,
                 **kwargs):

        self.tensor_args = tensor_args

        self.dataset_subdir = dataset_subdir
        self.base_dir = dataset_base_dir

        self.field_key_inputs = 'inputs'
        self.field_key_condition = 'condition'
        self.fields = {}

        # load data
        self.include_velocity = include_velocity
        # self.map_task_id_to_trajectories_id = {}
        # self.map_trajectory_id_to_task_id = {}
        self.load_inputs(u_filename, x0_filename)

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_inputs].shape
        self.n_init = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.inputs_dim = (self.n_support_points, d)

        # normalize the inputs (for the diffusion model)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        # self.fields[self.field_key_condition] = self.condition
        self.normalizer_keys = [self.field_key_inputs, self.field_key_condition] # [self.field_key_inputs, self.field_key_task]
        self.normalize_all_data(*self.normalizer_keys)

    def load_inputs(self, u_filename:str, x0_filename:str):
        # load training inputs
        check = self.tensor_args['device']
        inputs_load = torch.load(os.path.join(self.base_dir, u_filename),map_location=self.tensor_args['device']) 
        inputs_load = inputs_load.float()
        inputs_training = inputs_load
        print(f'inputs_training -- {inputs_training.shape}')
        self.fields[self.field_key_inputs] = inputs_training

        # x0 condition
        x0_condition =  torch.load(os.path.join(self.base_dir, x0_filename),map_location=self.tensor_args['device'])
        x0_condition = x0_condition.float()
        print(f'x0 condition -- {x0_condition.shape}')
        self.fields[self.field_key_condition] = x0_condition

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
        msg = f'InputsDataset\n' \
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

        # # build hard conditions
        # hard_conds = self.get_hard_conditions(condition_normalized)
        # data.update({'hard_conds': hard_conds})

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


# class InputsHardDataset(InputsDatasetBase):

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    # def get_hard_conditions(self, condition):
        # condition x0
        # hard_x = condition # dimension 1*4*1
        # start_state_pos = self.robot.get_position(traj[0])
        # goal_state_pos = self.robot.get_position(traj[-1])

        # if self.include_velocity:
        #     # If velocities are part of the state, then set them to zero at the beggining and end of a trajectory
        #     start_state = torch.cat((start_state_pos, torch.zeros_like(start_state_pos)), dim=-1)
        #     goal_state = torch.cat((goal_state_pos, torch.zeros_like(goal_state_pos)), dim=-1)
        # else:
        #     start_state = start_state_pos
        #     goal_state = goal_state_pos

        # if normalize:
        #     hard_x = self.normalizer.normalize(start_state, key=self.field_key_inputs)
        #     start_state = self.normalizer.normalize(start_state, key=self.field_key_inputs)
        #     goal_state = self.normalizer.normalize(goal_state, key=self.field_key_inputs)

        # if horizon is None:
        #     horizon = self.n_support_points
        # hard_conds = {
        #     hard_x
            # horizon - 1: goal_state
        # }
        # return hard_conds