# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_util import Projection


class ViewRendering(nn.Module):
    """
    Class for rendering images from given camera parameters and pixel wise depth information
    """
    def __init__(self, cfg, rank):
        super().__init__()
        self.read_config(cfg)
        self.rank = rank
        self.project = self.init_project_imgs(rank)      
            
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_project_imgs(self, rank):
        project_imgs = {}
        project_imgs = Projection(
                self.batch_size, self.height, self.width, rank)
        return project_imgs    
    
    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var + 1e-16)     
    
    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T, scale=0):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        
        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped

    def get_virtual_depth(self, src_depth, src_mask, src_invK, src_K, tar_depth, tar_invK, tar_K, T, min_depth, max_depth, scale=0):
        """
        This function backward-warp source depth into the target coordinate.
        src -> target
        """       
        # transform source depth
        b, _, h, w = src_depth.size()    
        src_points = self.project.backproject(src_invK, src_depth)
        src_points_warped = torch.matmul(T[:, :3, :], src_points)
        src_depth_warped = src_points_warped.reshape(b, 3, h, w)[:, 2:3, :, :]

        # reconstruct depth: backward-warp source depth to the target coordinate
        pix_coords = self.project(tar_depth, torch.inverse(T), tar_invK, src_K)  
        depth_warped = F.grid_sample(src_depth_warped, pix_coords, mode='bilinear', 
                                        padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_depth = torch.isnan(depth_warped)
        depth_warped[inf_depth] = 2.0
        inf_regions = torch.isnan(mask_warped)
        mask_warped[inf_regions] = 0
        
        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=1, keepdim=True) > 0

        # range handling
        valid_depth_min = (depth_warped > min_depth)
        depth_warped[~valid_depth_min] = min_depth
        valid_depth_max = (depth_warped < max_depth)
        depth_warped[~valid_depth_max] = max_depth
        return depth_warped, (~invalid_mask).float() * mask_warped * valid_depth_min * valid_depth_max        
        
    def forward(self, inputs, outputs, cam, rel_pose_dict):
        # predict images for each scale(default = scale 0 only)
        source_scale = 0
        
        # ref inputs
        ref_color = inputs['color', 0, source_scale][:,cam, ...]        
        ref_mask = inputs['mask'][:, cam, ...]
        ref_K = inputs[('K', source_scale)][:,cam, ...]
        ref_invK = inputs[('inv_K', source_scale)][:,cam, ...]  
        
        # output
        target_view = outputs[('cam', cam)]
        
        for scale in self.scales:           
            ref_depth = target_view[('depth', scale)]          
            for frame_id in self.frame_ids[1:]:                
                # for temporal learning
                T = target_view[('cam_T_cam', 0, frame_id)]
                src_color = inputs['color', frame_id, source_scale][:, cam, ...] 
                src_mask = inputs['mask'][:, cam, ...]
                warped_img, warped_mask = self.get_virtual_image(
                    src_color, 
                    src_mask, 
                    ref_depth, 
                    ref_invK, 
                    ref_K, 
                    T, 
                    source_scale
                )
                
                if self.intensity_align:
                    warped_img = self.get_norm_image_single(
                        ref_color, 
                        ref_mask,
                        warped_img, 
                        warped_mask
                    )
                
                target_view[('color', frame_id, scale)] = warped_img
                target_view[('color_mask', frame_id, scale)] = warped_mask            

            # spatio-temporal learning
            if self.spatio or self.spatio_temporal:
                for frame_id in self.frame_ids:
                    overlap_img = torch.zeros_like(ref_color)
                    overlap_mask = torch.zeros_like(ref_mask)       
                    
                    for cur_index in self.rel_cam_list[cam]:
                        # for partial surround view training
                        if cur_index >= self.num_cams: 
                            continue

                        src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
                        src_mask = inputs['mask'][:, cur_index, ...]
                        src_K = inputs[('K', source_scale)][:, cur_index, ...]                        
                        
                        rel_pose = rel_pose_dict[(frame_id, cur_index)]
                        warped_img, warped_mask = self.get_virtual_image(
                            src_color, 
                            src_mask, 
                            ref_depth, 
                            ref_invK, 
                            src_K,
                            rel_pose, 
                            source_scale
                        )

                        if self.intensity_align:
                            warped_img = self.get_norm_image_single(
                                ref_color, 
                                ref_mask,
                                warped_img, 
                                warped_mask
                            )

                        # assuming no overlap between warped images
                        overlap_img = overlap_img + warped_img
                        overlap_mask = overlap_mask + warped_mask
                    
                    target_view[('overlap', frame_id, scale)] = overlap_img
                    target_view[('overlap_mask', frame_id, scale)] = overlap_mask
                    
            # depth augmentation at a novel view
            if self.aug_depth:
                tform_depth = []
                tform_mask = []

                aug_ext = inputs['extrinsics_aug'][:, cam, ...]
                aug_ext_inv = torch.inverse(aug_ext)                
                aug_K, aug_invK = ref_K, ref_invK
                aug_depth = target_view[('depth', scale, 'aug')]

                for i, curr_index in enumerate(self.rel_cam_list[cam] + [cam]):
                    # for partial surround view training
                    if curr_index >= self.num_cams: 
                        continue

                    src_ext = inputs['extrinsics'][:, curr_index, ...]                        
                    
                    src_depth = outputs[('cam', curr_index)][('depth', scale)]
                    src_mask = inputs['mask'][:, curr_index, ...]                
                    src_invK = inputs[('inv_K', source_scale)][:,curr_index, ...]
                    src_K = inputs[('K', source_scale)][:,curr_index, ...]

                    # current view to the novel view
                    rel_pose = torch.matmul(aug_ext_inv, src_ext)
                    warp_depth, warp_mask = self.get_virtual_depth(
                        src_depth, 
                        src_mask, 
                        src_invK, 
                        src_K,
                        aug_depth, 
                        aug_invK, 
                        aug_K, 
                        rel_pose,
                        self.min_depth,
                        self.max_depth
                    )

                    tform_depth.append(warp_depth)
                    tform_mask.append(warp_mask)

                target_view[('tform_depth', scale)] = tform_depth
                target_view[('tform_depth_mask', scale)] = tform_mask

        outputs[('cam', cam)] = target_view