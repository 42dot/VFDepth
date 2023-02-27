# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn

from external.layers import ResnetEncoder, PoseDecoder


class MonoPoseNet(nn.Module):
    """
    Pytorch module for a pose network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """
    def __init__(self, cfg):
        super(MonoPoseNet, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']    
        
        self.pose_encoder = ResnetEncoder(num_layers, pretrained, num_input_images = 2)
        del self.pose_encoder.encoder.fc # For ddp training
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, 
                                        num_input_features=1, 
                                        num_frames_to_predict_for=1)
                                        
    def forward(self, inputs, frame_ids, cam):
        pose_inputs = [inputs['color_aug', f_i, 0][:,cam,...] for f_i in frame_ids]
        input_images = torch.cat(pose_inputs, 1)
        pose_feature = [self.pose_encoder(input_images)]
        axis_angle, translation = self.pose_decoder(pose_feature)
        return axis_angle, torch.clamp(translation, -4.0, 4.0) # for DDAD dataset