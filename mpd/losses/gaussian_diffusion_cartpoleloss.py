import torch

from mpd.models import build_context


class GaussianDiffusionCartPoleLoss:

    def __init__(self):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        """
        inputs_normalized = input_dict[f'{dataset.field_key_inputs}_normalized']
        # print(f"inputs_normalized -- {inputs_normalized.shape}")

        context = input_dict[f'{dataset.field_key_condition}_normalized']
        # print(f"context-- {context}")

        hard_conds = None
        # print(f"hard_conds-- {hard_conds}")

        loss, info = diffusion_model.loss(inputs_normalized, context, hard_conds)

        loss_dict = {'diffusion_loss': loss}

        return loss_dict, info
    
class GaussianDiffusionNMPC_UJ_Loss:

    def __init__(self):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        """
        uj_normalized_from_data = input_dict[f'{dataset.field_key_uj}_normalized']

        context = input_dict[f'{dataset.field_key_condition}_normalized']

        hard_conds = None

        loss, info = diffusion_model.loss(uj_normalized_from_data, context, hard_conds)

        loss_dict = {'diffusion_loss': loss}

        return loss_dict, info
