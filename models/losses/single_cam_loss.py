# Copyright (c) 2023 42dot. All rights reserved.
import torch

from .loss_util import compute_photometric_loss, compute_edg_smooth_loss, compute_masked_loss, compute_auto_masks
from .base_loss import BaseLoss

_EPSILON = 0.00001


class SingleCamLoss(BaseLoss):
    """
    Class for single camera(temporal only) loss calculation
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)        

    def compute_reproj_loss(self, inputs, target_view, cam=0, scale=0, ref_mask=None):
        """
        This function computes reprojection loss using auto mask. 
        """
        reprojection_losses = []
        for frame_id in self.frame_ids[1:]:
            reproj_loss_args = {
                'pred': target_view[('color', frame_id, scale)],
                'target': inputs['color',0, 0][:, cam, ...]
            }
            reprojection_losses.append(
                compute_photometric_loss(**reproj_loss_args)
            )                
            
        reprojection_losses = torch.cat(reprojection_losses, 1)
        reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
        
        identity_reprojection_losses = []
        for frame_id in self.frame_ids[1:]:
            identity_reproj_loss_args = {
                'pred': inputs[('color', frame_id, 0)][:, cam, ...],
                'target': inputs['color',0, 0][:, cam, ...]              
            }
            identity_reprojection_losses.append(
                compute_photometric_loss(**identity_reproj_loss_args)
            )

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        identity_reprojection_losses = identity_reprojection_losses + \
                                        _EPSILON * torch.randn(identity_reprojection_losses.shape).to(self.rank)
        identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)             
           
        # find minimum losses
        reprojection_auto_mask = compute_auto_masks(reprojection_loss, identity_reprojection_loss)
        reprojection_auto_mask *= ref_mask

        target_view[('reproj_loss', scale)] = reprojection_auto_mask * reprojection_loss
        target_view[('reproj_mask', scale)] = reprojection_auto_mask
        return compute_masked_loss(reprojection_loss, reprojection_auto_mask)
   
    def compute_smooth_loss(self, inputs, target_view, cam = 0, scale = 0, ref_mask=None):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        color = inputs['color', 0, scale][:, cam, ...]
        disp = target_view[('disp', scale)]
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-8)
        return compute_edg_smooth_loss(color, norm_disp)
        
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

            reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)

            cam_loss += reprojection_loss
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['reproj_loss'] = reprojection_loss.item()            
                loss_dict['smooth'] = smooth_loss.item()

                # log statistics
                self.get_logs(loss_dict, target_view, cam)                    
        
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict       