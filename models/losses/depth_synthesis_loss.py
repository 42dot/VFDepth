# Copyright (c) 2023 42dot. All rights reserved.
import torch

from .loss_util import compute_masked_loss
from .multi_cam_loss import MultiCamLoss


class DepthSynLoss(MultiCamLoss):
    """
    Class for depth synthesis loss calculation
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        
    def compute_aug_losses(self, output, scale):
        """
        This function computes depth augmentation loss(consistency, smoothness).
        """
        pred_losses = []
        pred_masks = []

        aug_depth = output[('depth', scale, 'aug')]
        tform_depth = output[('tform_depth', scale)]
        tform_mask = output[('tform_depth_mask', scale)]                

        for n_d in range(len(tform_depth)):
            tform_d = tform_depth[n_d]
            tform_m = tform_mask[n_d]
            pred_loss = (aug_depth - tform_d).abs() / (aug_depth + tform_d + 1e-8)
            pred_loss = torch.clamp(pred_loss, 0., 1.)
            pred_losses.append(pred_loss)
            pred_masks.append(tform_m)
        
        pred_losses = torch.cat(pred_losses, dim=0)
        pred_masks = torch.cat(pred_masks, dim=0)
        depth_con_loss = compute_masked_loss(pred_losses, pred_masks)
        
        disp_aug = output[('disp', scale, 'aug')]
        mean_disp_aug = disp_aug.mean(2, True).mean(3, True)
        norm_disp_aug = disp_aug / (mean_disp_aug + 1e-8)
    
        grad_disp_x = torch.abs(norm_disp_aug[:, :, :, :-1] - norm_disp_aug[:, :, :, 1:])
        grad_disp_y = torch.abs(norm_disp_aug[:, :, :-1, :] - norm_disp_aug[:, :, 1:, :])
        depth_sm_loss = grad_disp_x.mean() + grad_disp_y.mean()
        return depth_con_loss, depth_sm_loss      
    
    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            reprojection_loss  = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)
            
            kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
            spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)
            
            # depth synthesis
            depth_con_loss, depth_sm_loss = self.compute_aug_losses(target_view, scale)
            depthsyn_loss = self.depth_con_coeff * depth_con_loss + self.depth_sm_coeff * depth_sm_loss
            
            cam_loss += reprojection_loss
            cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss             
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            cam_loss += depthsyn_loss
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['reproj_loss'] = reprojection_loss.item()
                loss_dict['spatio_loss'] = spatio_loss.item()
                loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
                loss_dict['depth_loss'] = depthsyn_loss.item()
                loss_dict['depth_sm_loss'] = depth_sm_loss.item()
                loss_dict['depth_con_loss'] = depth_con_loss.item()                    
                loss_dict['smooth'] = smooth_loss.item()

                # log statistics
                self.get_logs(loss_dict, target_view, cam)                       
        
        cam_loss /= len(self.scales)
        loss_dict['cam_loss'] = cam_loss.item()
        return cam_loss, loss_dict