##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
import random
import torch
##########################################################################################
# Path Config
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
##########################################################################################
# import
import logging
import numpy as np
from utils.utils import create_logger, copy_all_src

# for post-processing
# from TSPTester_inTSPlin_post import TSPTester as Tester

from TSPTester_inTSPlib import TSPTester as Tester

########### Frequent use parameters  ##################################################

test_in_tsplib = True  # test in tsplib or not

########### model to load ###############

model_load_path = 'result/pre_trained_model'
model_load_epoch = 49
mode = 'test'


env_params = {
    'mode': mode,
    'test_in_tsplib': test_in_tsplib,
    'data_path': None,
    'sub_path': False,
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 128,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 1,
    'test_batch_size': 1,
    'beam_size': 16,
    'num_PRC': 1000,
    'beam': True,
    'PRC': True
}

logger_params = {
    'log_file': {
        'desc': f'test__tsp_real_world',
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

def main_test(epoch, path):
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }
    seed_everything(2024)
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    path = model_load_path
    score_optimal, score_student, gap = main_test(model_load_epoch, path)
