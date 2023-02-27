# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn

from external.layers import ResnetEncoder, DepthDecoder


class MonoDepthNet(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """    
    def __init__(self, cfg):
        super(MonoDepthNet, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']        
        scales = cfg['training']['scales']
                       
        self.depth_encoder = ResnetEncoder(num_layers, pretrained, 1)
        del self.depth_encoder.encoder.fc # For ddp training
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, scales)
        
    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        outputs = self.depth_decoder(depth_feature)
        return outputs