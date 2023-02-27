# Copyright (c) 2023 42dot. All rights reserved.
import argparse 
 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
    
import utils
from models import VFDepthAlgo
from trainer import VFDepthTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='VFdepth evaluation script')
    parser.add_argument('--config_file', default ='./configs/surround_fusion.yaml', type=str, help='Config yaml file')
    parser.add_argument('--weight_path', default = None, type=str, help='Pretrained weight path')
    args = parser.parse_args() 
    return args


def test(cfg):
    print("Evaluating")
    model = VFDepthAlgo(cfg, 0)
    trainer = VFDepthTrainer(cfg, 0, use_tb = False)
    trainer.evaluate(model, vis_results = cfg['eval']['eval_visualize'])

    
def test_ddp(rank, cfg):
    print("Evaluating")
    utils.setup_ddp(rank, cfg['ddp']['world_size'])
    model = VFDepthAlgo(cfg, rank)
    trainer = VFDepthTrainer(cfg, rank, use_tb = False)
    trainer.evaluate(model, vis_results = cfg['eval']['eval_visualize'])
    utils.clear_ddp()    
    

if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='eval', weight_path = args.weight_path)
        
    # Evaluating on DDP trained model
    if cfg['ddp']['ddp_enable'] == True:
        import torch.multiprocessing as mp
        mp.spawn(test_ddp, nprocs=cfg['ddp']['world_size'], args=(cfg,))
    else:
        test(cfg)
