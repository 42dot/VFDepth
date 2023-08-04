# Copyright (c) 2023 42dot All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse 
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="True"
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

import utils
from models import VFDepthAlgo
from trainer import VFDepthTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='VFDepth training script')
    parser.add_argument('--config_file', default ='./configs/ddad/ddad_surround_fusion.yaml', type=str, help='Config yaml file')
    args = parser.parse_args()
    return args


def train(cfg):    
    model = VFDepthAlgo(cfg, 0)
    trainer = VFDepthTrainer(cfg, 0)
    trainer.learn(model)

    
def train_ddp(rank, cfg):
    print("Training on rank %d."%rank)
    utils.setup_ddp(rank, cfg['ddp']['world_size'])
    model = VFDepthAlgo(cfg, rank)
    trainer = VFDepthTrainer(cfg, rank)
    trainer.learn(model)
    utils.clear_ddp()


if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train')

    # DDP training
    if cfg['ddp']['ddp_enable'] == True:
        import torch.multiprocessing as mp
        mp.spawn(train_ddp, nprocs=cfg['ddp']['world_size'], args=(cfg,), join=True)
    else:
        train(cfg)
