import os
from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local
from mpd.datasets.nmpc_cart_pole_data import TRAINING_DATA_AMOUNT

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################################################################################################

####### Setting ######

# training data folder
DATASET_SUBDIR = 'CartPole-NMPC' # the folder of the training data files (location: /root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/CartPole-LMPC)

# learning parameters
BATCH_SIZE = 512
LEARNING_RATE = 3e-3

EPOCHES = 300 # times that the whole data should be trained

MODEL_SAVED_PATH = '/MPC_DynamicSys/sharedVol/model/nmpc'

# LAUNCHER

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

# N_EXPS_IN_PARALLEL = 4
N_EXPS_IN_PARALLEL = 1

N_CORES = N_EXPS_IN_PARALLEL * 12
# MEMORY_SINGLE_JOB = 12000
MEMORY_SINGLE_JOB = 1200
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GPU_IDX = 1
GRES = 'gpu:'+str(GPU_IDX) if USE_CUDA else None
CONDA_ENV = 'mpd'


exp_name = f'train_nmpc_diffusion_CartPole'

launcher = Launcher(
    exp_name=exp_name,
    exp_file='nmpc_cart_pole_train',
    # project_name='project01234',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True
)

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP

dataset_subdir_l = [
    DATASET_SUBDIR
]

include_velocity_l = [
    False
]

use_ema_l = [
    True
]

variance_schedule_l = [
    'exponential'
]

n_diffusion_steps_l = [
    25,
]

predict_epsilon_l = [
    True
]

dim = 32 # ???

unet_dim_mults_option_l = [
    0
]


batch_size = BATCH_SIZE
lr = LEARNING_RATE


wandb_options = dict(
    wandb_mode='disabled',  # "online", "offline" or "disabled"
    wandb_entity='scoreplan',
    wandb_project=exp_name
)

########################################################################################################################
# RUN

for dataset_subdir, include_velocity, use_ema, variance_schedule, n_diffusion_steps, predict_epsilon, unet_dim_mults_option in \
        product(dataset_subdir_l, include_velocity_l, use_ema_l, variance_schedule_l, n_diffusion_steps_l, predict_epsilon_l, unet_dim_mults_option_l):

    launcher.add_experiment(
        dataset_subdir__=dataset_subdir,
        include_velocity__=include_velocity,
        use_ema__=use_ema,
        variance_schedule__=variance_schedule,
        n_diffusion_steps__=n_diffusion_steps,
        predict_epsilon__=predict_epsilon,
        unet_dim_mults_option__=unet_dim_mults_option,

        lr=lr,

        batch_size=batch_size,

        num_train_steps = TRAINING_DATA_AMOUNT*EPOCHES/BATCH_SIZE, 

        model_saving_address = MODEL_SAVED_PATH,

        # steps_til_ckpt=50000,
        steps_til_ckpt=10000, # 10000

        # steps_til_summary=20000,
        steps_til_summary=2000,

        **wandb_options,
        wandb_group=f'{dataset_subdir}-{include_velocity}-{use_ema}-{variance_schedule}-{n_diffusion_steps}-{predict_epsilon}-{unet_dim_mults_option}',

        debug=False,
        
        gpu_idx = GPU_IDX,
    )

launcher.run(LOCAL, TEST)