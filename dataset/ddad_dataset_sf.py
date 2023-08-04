# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import pandas as pd

from .data_util import transform_mask_sample, mask_loader_scene, align_dataset

from external.utils import Camera, generate_depth_map, make_list
from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample


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
        # generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
        # load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # otherwise, create, save and return
        else:
            # get pointcloud
            scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
            pc_datum_data, _ = self.dataset.get_point_cloud_from_datum(
                                scene_idx, sample_idx_in_scene, self.depth_type)

            # create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            
            # generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            
            # save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
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
        
        # loop over all cameras
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