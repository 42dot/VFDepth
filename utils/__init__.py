# Copyright (c) 2023 42dot. All rights reserved.
from .ddp import setup_ddp, clear_ddp
from .logger import Logger
from .misc import get_config
from .visualize import aug_depth_params

__all__ = ['setup_ddp', 'clear_ddp', 'Logger',
           'get_config', 'aug_depth_params']


import sys


_LIBS = ['./external/packnet_sfm', './external/dgp', './external/monodepth2']    

def setup_env():       
    if not _LIBS[0] in sys.path:        
        for lib in _LIBS:
            sys.path.append(lib)

setup_env()