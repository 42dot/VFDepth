# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import PIL.Image as pil

import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .data_util import img_loader, mask_loader_scene, align_dataset, transform_mask_sample
from external.dataset import stack_sample


class NuScenesdataset(Dataset):
    """
    Loaders for NuScenes dataset
    """
    def __init__(self, path, split,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 depth_type=None,
                 scale_range=[0],
                 with_pose=None,
                 with_mask=None,
                 ):        
        super().__init__()
        version = 'v1.0-trainval'
        self.path = path
        self.split = split
        self.dataset_idx = 0

        self.cameras = cameras
        self.scales = np.arange(scale_range+2) 
        self.num_cameras = len(cameras)

        self.bwd = back_context
        self.fwd = forward_context
        
        self.has_context = back_context + forward_context > 0
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose

        self.loader = img_loader

        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))        
        self.mask_path = os.path.join(cur_path, 'nuscenes_mask')
        self.mask_loader = mask_loader_scene

        self.dataset = NuScenes(version=version, dataroot=self.path, verbose=True)
        
        # list of scenes for training and validation of model
        with open('dataset/nuscenes/{}.txt'.format(self.split), 'r') as f:
            self.filenames = f.readlines()

    def get_current(self, key, cam_sample):
        """
        This function returns samples for current contexts
        """        
        # get current timestamp rgb sample
        if key == 'rgb':
            rgb_path = cam_sample['filename']
            return self.loader(os.path.join(self.path, rgb_path))
        # get current timestamp camera intrinsics
        elif key == 'intrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return np.array(cam_param['camera_intrinsic'], dtype=np.float32)
        # get current timestamp camera extrinsics
        elif key == 'extrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return self.get_tranformation_mat(cam_param)
        else:
            raise ValueError('Unknown key: ' +key)

    def get_context(self, key, cam_sample):
        """
        This function returns samples for backward and forward contexts
        """
        bwd_context, fwd_context = [], []
        if self.bwd != 0:
            if self.split == 'val': # validation
                bwd_sample = cam_sample
            else:
                bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])
            bwd_context = [self.get_current(key, bwd_sample)]

        if self.fwd != 0:
            if self.split == 'val':
                fwd_sample = cam_sample
            else:
                fwd_sample = self.dataset.get('sample_data', cam_sample['next'])
            fwd_context = [self.get_current(key, fwd_sample)]
        return bwd_context + fwd_context

    def generate_depth_map(self, sample, sensor, cam_sample):
        """
        This function returns depth map for nuscenes dataset,
        result of depth map is saved in nuscenes/samples/DEPTH_MAP
        """        
        # generate depth filename
        filename = '{}/{}.npz'.format(
                        os.path.join(os.path.dirname(self.path), 'samples'),
                        'DEPTH_MAP/{}/{}'.format(sensor, cam_sample['filename']))
                        
        # load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        else:
            lidar_sample = self.dataset.get(
                'sample_data', sample['data']['LIDAR_TOP'])

            # lidar points                
            lidar_file = os.path.join(
                self.path, lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            # lidar -> world
            lidar_pose = self.dataset.get(
                'ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation= Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            # lidar -> ego
            sensor_sample = self.dataset.get(
                'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_rotation = Quaternion(
                sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_translation = np.array(
                sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(
                lidar_points[:, :3], lidar_to_ego_rotation.T)
            ego_lidar_points += lidar_to_ego_translation

            homo_ego_lidar_points = np.concatenate(
                (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)


            # world -> ego
            ego_pose = self.dataset.get(
                    'ego_pose', cam_sample['ego_pose_token'])
            ego_rotation = Quaternion(ego_pose['rotation']).inverse
            ego_translation = - np.array(ego_pose['translation'])[:, None]
            world_to_ego = np.vstack([
                    np.hstack((ego_rotation.rotation_matrix,
                               ego_rotation.rotation_matrix @ ego_translation)),
                    np.array([0, 0, 0, 1])
                    ])

            # Ego -> sensor
            sensor_sample = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(
                sensor_sample['translation'])[:, None]
            sensor_to_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, 
                           sensor_translation)),
                np.array([0, 0, 0, 1])
               ])
            ego_to_sensor = np.linalg.inv(sensor_to_ego)
            
            # lidar -> sensor
            lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
            cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

            # depth > 0
            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            # sensor -> image
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = sensor_sample['camera_intrinsic']
            pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
            pixel_points[:, :2] /= pixel_points[:, 2:3]
            
            # load image for pixel range
            image_filename = os.path.join(
                self.path, cam_sample['filename'])
            img = pil.open(image_filename)
            h, w, _ = np.array(img).shape
            
            # mask points in pixel range
            pixel_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] <= w-1)\
                        & (pixel_points[:,1] >= 0) & (pixel_points[:,1] <= h-1)
            valid_points = pixel_points[pixel_mask].round().int()
            valid_depth = cam_lidar_points[:, 2][pixel_mask]
        
            depth = np.zeros([h, w])
            depth[valid_points[:, 1], valid_points[:,0]] = valid_depth
        
            # save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            return depth

    def get_tranformation_mat(self, pose):
        """
        This function transforms pose information in accordance with DDAD dataset format
        """
        extrinsics = Quaternion(pose['rotation']).transformation_matrix
        extrinsics[:3, 3] = np.array(pose['translation'])
        return extrinsics.astype(np.float32)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # get nuscenes dataset sample
        frame_idx = self.filenames[idx].strip().split()[0]
        sample_nusc = self.dataset.get('sample', frame_idx)
        
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        # loop over all cameras            
        for cam in self.cameras:
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                'idx': idx,
                'sensor_name': cam,
                'contexts': contexts,
                'filename': cam_sample['filename'],
                'rgb': self.get_current('rgb', cam_sample),
                'intrinsics': self.get_current('intrinsics', cam_sample)
            }

            # if depth is returned            
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map(sample_nusc, cam, cam_sample)
                })
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics':self.get_current('extrinsics', cam_sample)
                })
            # if mask is returned
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)
                })        
            # if context is returned
            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', cam_sample)
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
                
