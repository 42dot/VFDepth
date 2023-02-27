# Copyright (c) 2023 42dot. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp(rank, world_size, manual_seed=True):
    """
    This function sets distributed data parallel(ddp) module for mutli-gpu training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    if manual_seed:
        random_seed = 42 + rank
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)    
        
    torch.cuda.set_device(rank)
    
    
def clear_ddp():
    """
    This function clears ddp training.
    """
    dist.destroy_process_group() 