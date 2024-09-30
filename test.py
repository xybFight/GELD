DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
import random
import torch

# Path Config
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging
import numpy as np
from utils.utils import create_logger, copy_all_src
from TSPTester import TSPTester as Tester

#############################################################


problem_size = 10000

model_load_path = 'result/pre_trained_model'
model_load_epoch = 49
mode = 'test'

##########################################################################################

env_params = {
    'mode': mode,
    'data_path': None,
    'sub_path': False
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
    'test_episodes': 200,
    'test_batch_size': 200,
    'beam_size': 16,
    'num_PRC': 1000,

    # use or not use
    # 'beam': False,
    'beam': True,
    # 'PRC':False
    'PRC': True
}

logger_params = {
    'log_file': {
        'desc': f'test__synthetic_tsp',
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

def main_test(epoch, path, size=None, disribution=None):
    if DEBUG_MODE:
        _set_debug_mode()
    create_logger(**logger_params)
    _print_config()

    seed_everything(2024)
    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run(size=size, disribution=disribution)
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
    allin = []
    for distribution in ["uniform", "clustered", "explosion", "implosion"]:
        for i in [100, 500, 1000, 5000, 10000]:
            if i == 100:
                tester_params['test_episodes'] = 200
                tester_params['test_batch_size'] = 200
            if i == 500:
                tester_params['test_episodes'] = 200
                tester_params['test_batch_size'] = 100
            elif i == 1000:
                tester_params['test_episodes'] = 200
                tester_params['test_batch_size'] = 200
            elif i == 5000:
                tester_params['test_episodes'] = 20
                tester_params['test_batch_size'] = 20
            elif i == 10000:
                tester_params['test_episodes'] = 20
                tester_params['test_batch_size'] = 20
            score_optimal, score_student, gap = main_test(model_load_epoch, path, size=i, disribution=distribution)
