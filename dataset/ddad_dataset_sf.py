# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms

from external.utils import Camera, generate_depth_map, make_list
from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample

_DEL_KEYS= ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'contexts', 'splitname'] 


def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    # resize transform
    resize_transform = transforms.Resize(image_shape, interpolation=Image.ANTIALIAS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transforms.ToTensor()    
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))
    with open(fname, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def align_dataset(sample, scales, contexts):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    aug_contexts = sample['rgb_context']
    org_images = sample['rgb_original']
    org_contexts= sample['rgb_context_original']    

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales 
    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:,:2,:] /= (2**scale)
        
        sample[('K', scale)] = scaled_K.copy()
        sample[('inv_K', scale)]= np.linalg.pinv(scaled_K).copy()

        resized_org = F.interpolate(org_images, 
                                          size=(w//(2**scale),h//(2**scale)),
                                          mode = 'bilinear',
                                          align_corners=False)
        resized_aug = F.interpolate(aug_images, 
                                          size=(w//(2**scale),h//(2**scale)), 
                                          mode = 'bilinear',
                                          align_corners=False)            
            
        sample[('color', 0, scale)] = resized_org
        sample[('color_aug', 0, scale)] = resized_aug

    # for context data
    for idx, frame in enumerate(contexts):
        sample[('color', frame, 0)] = org_contexts[idx]        
        sample[('color_aug',frame, 0)] = aug_contexts[idx]
        
    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]
    return sample


class DDADdatasetSF(DGPDataset):
    """
    Superclass for DGP dataset loaders of the packnet-sfm repository.
    """
    def __init__(self, *args, with_mask, scale_range, **kwargs):
        super().__init__(*args, **kwargs)
        self.cameras = kwargs['cameras']
        self.scales = np.arange(scale_range+2) 

        ## self-occ masks 
        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.mask_path = os.path.join(cur_path, 'ddad_mask')
        file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
        
        self.mask_idx_dict = pd.read_pickle(file_name)
        self.mask_loader = mask_loader_scene
        
        datum_names = self.cameras + ['lidar']
        self.dataset = SynchronizedSceneDataset(self.path,
                        split=self.split,
                        datum_names=datum_names,
                        backward_context=self.bwd,
                        forward_context=self.fwd,
                        requested_annotations=None,
                        only_annotated_datums=False,
                        )        

    def generate_depth_map_sf(self, sample_idx, datum_idx, filename):
        """
        This function follows structure of dgp_dataset/generate_depth_map in packnet-sfm. 
        Due to the version issue with dgp, minor revision was made to get the correct value.
        """      
        # Generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
        # Load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # Otherwise, create, save and return
        else:
            # Get pointcloud
            scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
            pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(
                                scene_idx, sample_idx_in_scene, self.depth_type)

            # Create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            
            # Generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            
            # Save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            # Return depth map
            return depth
    
    def get_filename_sf(self, sample_idx, datum_idx):
        """
        This function is defined to meet dgp version(v1.4)
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def __getitem__(self, idx):
        # get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # loop over all cameras
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)
            
        # for self-occ mask
        scene_idx, _, _ = self.dataset.dataset_item_index[idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        scene_name = os.path.basename(scene_dir)
        mask_idx = self.mask_idx_dict[int(scene_name)]
        
        for cam in range(self.num_cameras):
            filename = self.get_filename_sf(idx, cam)

            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', cam),
                'contexts': contexts,
                'filename': filename,
                'splitname': '%s_%010d' % (self.split, idx),                
                'rgb': self.get_current('rgb', cam),              
                'intrinsics': self.get_current('intrinsics', cam),
            }

            # if depth is returned
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map_sf(idx, cam, filename)
                })
            # if depth is returned
            if self.with_input_depth:
                data.update({
                    'input_depth': self.generate_depth_map_sf(idx, cam, filename)
                })

            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics': self.get_current('extrinsics', cam).matrix
                })
            # with mask
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, mask_idx, self.cameras[cam])
                })
                
            # if context is returned
            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', cam)
                })

            sample.append(data)

        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]
            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]

        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts)
        return sample