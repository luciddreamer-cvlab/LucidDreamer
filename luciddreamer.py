# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import json
import time
import warnings
from random import randint

warnings.filterwarnings(action='ignore')

import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter

import torch
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline

from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import loadCameraPreset
from utils.loss import l1_loss, ssim
from utils.camera import load_json
from utils.trajectory import get_camerapaths, get_pcdGenPoses


class LucidDreamer:
    def __init__(self, savepath='./'):
        self.opt = GSParams()
        self.cam = CameraParams()
        self.savepath = savepath
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath, exist_ok=True)

        self.gaussians = GaussianModel(self.opt.sh_degree)

        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        
        self.rgb = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16).to('cuda')
        self.d = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

    def run(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, render_camerapath):
        gaussians, default_gallery = self.create(rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps)
        gallery = self.render_video(render_camerapath)
        return (gaussians, default_gallery, gallery)

    def create(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps):
        self.traindata = self.generate_pcd(rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps)
        self.scene = Scene(self.traindata, self.gaussians, self.opt)        
        self.training()
        outfile = self.save_ply()
        default_gallery = self.render_video('LLFF')
        return outfile, default_gallery
    
    def save_ply(self):
        self.scene.gaussians.save_ply(os.path.join(self.savepath, 'gaussiansplatting.ply'))
        return os.path.join(self.savepath, 'gaussiansplatting.ply')

    def render_video(self, preset, gaussians=None, traindata_path=None):
        videofname = os.path.join(self.savepath, f'video_{preset}_60fps.mp4')

        if not hasattr(self, 'gaussians'):
            self.gaussians = gaussians
        
        if not hasattr(self, 'scene'):
            print(traindata_path)

            with open(traindata_path, 'r') as f:
                traindata = json.load(traindata_path)

            print(traindata)

            view_total = loadCameraPreset(traindata=traindata, presetdata=get_camerapaths())
            views = view_total[preset]
            print(view_total.keys())
        else:
            views = self.scene.getPresetCameras(preset)
        
        framelist = []
        depthlist = []
        dmin, dmax = 1e3, 0
        for view in views:
            results = render(view, self.gaussians, self.opt, self.background)
            rendering = results['render']

            dmask = (results['depth']>0)
            depth = (results['depth']*dmask).detach().cpu().numpy()
            depthlist.append(depth)
            if results['depth'][dmask].min().item() < dmin:
                dmin = results['depth'][dmask].min().item()
            if results['depth'][dmask].max().item() > dmax:
                dmax = results['depth'][dmask].max().item()
            
            framelist.append(np.round(rendering.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
        
        imageio.mimwrite(videofname, framelist, fps=60, quality=8)
        return videofname

    def training(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        for iteration in tqdm(range(1, self.opt.iterations + 1)):
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

    def generate_pcd(self, rgb_cond, prompt, negative_prompt, pcdgenpath, seed, diff_steps, progress=gr.Progress()):
        ## processing inputs
        generator=torch.Generator(device='cuda').manual_seed(seed)

        w_in, h_in = rgb_cond.size
        if w_in/h_in > 1.1 or h_in/w_in > 1.1: # if height and width are similar, do center crop
            in_res = max(w_in, h_in)
            image_in, mask_in = np.zeros((in_res, in_res, 3), dtype=np.uint8), 255*np.ones((in_res, in_res, 3), dtype=np.uint8)
            image_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = np.array(rgb_cond)
            mask_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = 0
            image_curr = self.rgb(prompt=prompt, negative_prompt=negative_prompt, generator=generator, 
                              image=Image.fromarray(image_in).resize((self.cam.W, self.cam.H)), 
                              mask_image=Image.fromarray(mask_in).resize((self.cam.W, self.cam.H))).images[0]

        else: # if there is a large gap between height and width, do inpainting
            if w_in > h_in:
                image_curr = rgb_cond.crop((int(w_in/2-h_in/2), 0, int(w_in/2+h_in/2), h_in)).resize((self.cam.W, self.cam.H))
            else: # w <= h
                image_curr = rgb_cond.crop((0, int(h_in/2-w_in/2), w_in, int(h_in/2+w_in/2))).resize((self.cam.W, self.cam.H))

        render_poses = get_pcdGenPoses(pcdgenpath)
        depth_curr = self.d.infer_pil(image_curr)
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])

        ###########################################################################################################################
        # Iterative scene generation
        H, W, K = self.cam.H, self.cam.W, self.cam.K

        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        ### initialize 
        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        progress(0, desc='Dreaming...')
        time.sleep(0.5)

        for j, tqdm_i in enumerate(progress.tqdm(range(1, len(render_poses)), desc='Dreaming')):
            i = j + 1
            if i == len(render_poses) - 1:
                break
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]

            ### Transform world to pixel
            pts_coord_cam2 = R.dot(pts_coord_world) + T
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)

            valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
            round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)
            image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
            image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            round_mask2 = np.zeros((H,W), dtype=np.float32)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1

            round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
            image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)

            mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
            image2 = mask2[...,None]*image2 + (1-mask2[...,None])*0

            mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
            mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
            mask_hf = np.where(mask_hf < 0.3, 0, 1)
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]

            image_curr = self.rgb(prompt=prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps= diff_steps,
                                    image=Image.fromarray(np.round(image2*255.).astype(np.uint8)), 
                                    mask_image=Image.fromarray(np.round((1-mask2[:,:])*255.).astype(np.uint8))).images[0]
            depth_curr = self.d.infer_pil(image_curr)

            t_z2 = torch.tensor(depth_curr)
            sc = torch.ones(1).float().requires_grad_(True)
            optimizer = torch.optim.Adam(params=[sc], lr=0.001)

            for idx in range(100):
                trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]]).requires_grad_(True)
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
                loss = torch.mean((torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    

            with torch.no_grad():
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]

            trans3d = trans3d.detach().numpy()

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32)
            new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2
            vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2

            compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
            compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)

            compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
            homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T

            compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1]
            compensate_depth_zero = np.zeros(4)
            compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)

            pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx]
            pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0)

            # Calculate for masked pixels
            masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
            new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
            new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
            compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
            new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1)
            pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)

        #################################################################################################

        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            'frames': [],
        }

        internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth})

        progress(0, desc='Aligning...')
        time.sleep(0.5)

        for i, tqdm_i in enumerate(progress.tqdm(range(len(render_poses)), desc='Aligning')):
            if i == len(render_poses) - 1:
                break
            for j in range(len(internel_render_poses)):
                idx = i * len(internel_render_poses) + j
                print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')

                ### Transform world to pixel
                Rw2i = render_poses[i,:3,:3]
                Tw2i = render_poses[i,:3,3:4]
                Ri2j = internel_render_poses[j,:3,:3]
                Ti2j = internel_render_poses[j,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

                # Transfrom cam2 to world + change sign of yz axis
                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                pixel_coord_camj = np.matmul(K, pts_coord_camj)

                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)


                x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
                grid = np.stack((x,y), axis=-1).reshape(-1,2)
                imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
                imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

                depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
                depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')

                maskj = np.zeros((H,W), dtype=np.float32)
                maskj[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

                maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

                traindata['frames'].append({
                    'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)), 
                    'transform_matrix': Pc2w.tolist(),
                })

        progress(1, desc='Baking Gaussians...')
        return traindata
