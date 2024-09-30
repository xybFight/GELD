
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
import numpy as np
import random
import torch

# Path Config
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging
from utils.utils import create_logger, copy_all_src
from TSPTrainer_new import TSPTrainer as Trainer

##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')
mode = 'train'
training_data_path = b+"/SL_training_data/train_TSP100_n100w-001.txt"

# the SL-trained model path
model_load_path = 'result/Here'
model_load_epoch = 1

env_params = {
    'data_path': training_data_path,
    'mode': mode,
    'sub_path': True
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 128,
}


optimizer_params = {
    'optimizer': {
        'lr': 1e-5,
                 },
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 50,
    'train_episodes': 512,
    'train_batch_size': 64,
    'val_batch_size': 512,
    'val_beam_batch_size': 512,
    'beam_size': 16,
    'max_limit': 5,
    'per_batch': 5,
    'best_limit': 3,
    'problem_size_init': 100,
    'problem_size_max': 1000,
    'model_load_path': model_load_path,
    'model_load_epoch': model_load_epoch,

    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/None',
        'epoch': 1,
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'log.txt'
    }
}

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    seed_everything(2024)
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 8
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
