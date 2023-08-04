# Copyright (c) 2023 42dot. All rights reserved.
import os
import yaml 
from collections import defaultdict
 
import torch

_NUSC_CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
_DDAD_CAM_LIST = ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']
_REL_CAM_DICT = {0: [1,2], 1: [0,3], 2: [0,4], 3: [1,5], 4: [2,5], 5: [3,4]}


def camera2ind(cameras):
    """
    This function transforms camera name list to indices 
    """    
    indices = []
    for cam in cameras:
        if cam in _DDAD_CAM_LIST:
            ind = _DDAD_CAM_LIST.index(cam)
        elif cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict        
    

def get_config(config, mode='train', weight_path=None):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print('Experiment: ', cfg_name)

        _log_path = os.path.join(cfg['data']['log_dir'], cfg_name)
        cfg['data']['log_path'] = _log_path
        cfg['data']['save_weights_root'] = os.path.join(_log_path, 'models')
        if weight_path == None:
            weight_path = os.path.join(_log_path, 'models', cfg['load']['weights'])
        cfg['data']['load_weights_dir'] = weight_path
        cfg['data']['num_cams'] = len(cfg['data']['cameras'])
        cfg['model']['mode'] = mode
        cfg['data']['rel_cam_list'] = get_relcam(cfg['data']['cameras'])
        
        if mode == 'train':
            cfg['eval']['syn_visualize'] = False # for pretrained 
            
        elif mode == 'eval':
            cfg['ddp']['world_size'] = 1
            cfg['ddp']['gpus'] = [0]
            cfg['training']['batch_size'] = cfg['eval']['eval_batch_size']
            cfg['training']['depth_flip'] = False
    return cfg

    
def pretty_ts(ts):
    """
    This function prints amount of time taken in user friendly way.
    """
    second = int(ts)
    minute = second // 60
    hour = minute // 60
    return f'{hour:02d}h{(minute%60):02d}m{(second%60):02d}s'


def cal_depth_error(pred, target):
    """
    This function calculates depth error using various metrics.
    """
    abs_rel = torch.mean(torch.abs(pred-target) / target)
    sq_rel = torch.mean((pred-target).pow(2) / target)
    rmse = torch.sqrt(torch.mean((pred-target).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(target) - torch.log(pred)).pow(2)))

    thresh = torch.max((target/pred), (pred/ target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3