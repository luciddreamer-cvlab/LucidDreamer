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
import glob
import json
import time
import datetime
import warnings
import shutil
from random import randint
from argparse import ArgumentParser

warnings.filterwarnings(action='ignore')

import pickle
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter

import torch
import torch.nn.functional as F
import gradio as gr
from diffusers import (
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline)

from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import loadCameraPreset
from utils.loss import l1_loss, ssim
from utils.camera import load_json
from utils.depth import colorize
from utils.lama import LaMa
from utils.trajectory import get_camerapaths, get_pcdGenPoses


get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)


class LucidDreamer:
    def __init__(self, for_gradio=True, save_dir=None):
        self.opt = GSParams()
        self.cam = CameraParams()
        self.save_dir = save_dir
        self.for_gradio = for_gradio
        self.root = 'outputs'
        self.default_model = 'SD1.5 (default)'
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

        self.gaussians = GaussianModel(self.opt.sh_degree)

        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        
        self.rgb_model = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16).to('cuda')
            # 'stablediffusion/SD1-5', revision='fp16', torch_dtype=torch.float16).to('cuda')
        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
        self.controlnet = None
        self.lama = None
        self.current_model = self.default_model

    def load_model(self, model_name, use_lama=True):
        if model_name is None:
            model_name = self.default_model
        if self.current_model == model_name:
            return
        if model_name == self.default_model:
            self.controlnet = None
            self.lama = None
            self.rgb_model = StableDiffusionInpaintPipeline.from_pretrained(
                #  'runwayml/stable-diffusion-inpainting',
                'stablediffusion/SD1-5',
                revision='fp16',
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to('cuda')
        else:
            if self.controlnet is None:
                self.controlnet = ControlNetModel.from_pretrained(
                    'lllyasviel/control_v11p_sd15_inpaint', torch_dtype=torch.float16)
            if self.lama is None and use_lama:
                self.lama = LaMa('cuda')
            self.rgb_model = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                f'stablediffusion/{model_name}',
                controlnet=self.controlnet,
                revision='fp16',
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to('cuda')
            # self.rgb_model.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        self.current_model = model_name

    def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
        image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
        mask_pil = Image.fromarray(np.round((1 - mask_image) * 255.).astype(np.uint8))
        if self.current_model == self.default_model:
            return self.rgb_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                image=image_pil,
                mask_image=mask_pil,
            ).images[0]

        kwargs = {
            'negative_prompt': negative_prompt,
            'generator': generator,
            'strength': 0.9,
            'num_inference_steps': num_inference_steps,
            'height': self.cam.H,
            'width': self.cam.W,
        }

        image_np = np.round(np.clip(image, 0, 1) * 255.).astype(np.uint8)
        mask_sum = np.clip((image.prod(axis=-1) == 0) + (1 - mask_image), 0, 1)
        mask_padded = pad_mask(mask_sum, 3)
        masked = image_np * np.logical_not(mask_padded[..., None])

        if self.lama is not None:
            lama_image = Image.fromarray(self.lama(masked, mask_padded).astype(np.uint8))
        else:
            lama_image = image

        mask_image = Image.fromarray(mask_padded.astype(np.uint8) * 255)
        control_image = self.make_controlnet_inpaint_condition(lama_image, mask_image)

        return self.rgb_model(
            prompt=prompt,
            image=lama_image,
            control_image=control_image,
            mask_image=mask_image,
            **kwargs,
        ).images[0]

    def d(self, im):
        return self.d_model.infer_pil(im)

    def make_controlnet_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def run(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, render_camerapath, model_name=None, example_name=None):
        gaussians = self.create(
            rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, model_name, example_name)
        gallery, depth = self.render_video(render_camerapath, example_name=example_name)
        return (gaussians, gallery, depth)

    def create(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, model_name=None, example_name=None):
        if self.for_gradio:
            self.cleaner()
            self.load_model(model_name)
        if example_name and example_name != 'DON\'T':
            outfile = os.path.join('examples', f'{example_name}.ply')
            if not os.path.exists(outfile):
                self.traindata = self.generate_pcd(rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps)
                self.scene = Scene(self.traindata, self.gaussians, self.opt)        
                self.training()
            outfile = self.save_ply(outfile)
        else:
            self.traindata = self.generate_pcd(rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps)
            self.scene = Scene(self.traindata, self.gaussians, self.opt)        
            self.training()
            self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            outfile = self.save_ply(os.path.join(self.save_dir, 'gsplat.ply'))
        return outfile
    
    def save_ply(self, fpath=None):
        if fpath is None:
            dpath = os.path.join(self.root, self.timestamp)
            fpath = os.path.join(dpath, 'gsplat.ply')
            os.makedirs(dpath, exist_ok=True)
        if not os.path.exists(fpath):
            self.gaussians.save_ply(fpath)
        else:
            self.gaussians.load_ply(fpath)
        return fpath

    def cleaner(self):
        # Remove the temporary file created yesterday.
        for dpath in glob.glob(os.path.join('/tmp/gradio', '*',  self.root, '*')):
            timestamp = datetime.datetime.strptime(os.path.basename(dpath), '%y%m%d_%H%M%S')
            if timestamp < datetime.datetime.now() - datetime.timedelta(days=1):
                try:
                    shutil.rmtree(dpath)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
        if self.for_gradio:
            # Delete gsplat.ply if exists
            if os.path.exists('./gsplat.ply'):
                os.remove('./gsplat.ply')

    def render_video(self, preset, example_name=None, progress=gr.Progress()):
        if example_name and example_name != 'DON\'T':
            videopath = os.path.join('examples', f'{example_name}_{preset}.mp4')
            depthpath = os.path.join('examples', f'depth_{example_name}_{preset}.mp4')
        else:
            if self.for_gradio:
                os.makedirs(os.path.join(self.root, self.timestamp), exist_ok=True)
                videopath = os.path.join(self.root, self.timestamp, f'{preset}.mp4')
                depthpath = os.path.join(self.root, self.timestamp, f'depth_{preset}.mp4')
            else:
                videopath = os.path.join(self.save_dir, f'{preset}.mp4')
                depthpath = os.path.join(self.save_dir, f'depth_{preset}.mp4')
        if os.path.exists(videopath) and os.path.exists(depthpath):
            return videopath, depthpath
        
        if not hasattr(self, 'scene'):
            views = load_json(os.path.join('cameras', f'{preset}.json'), self.cam.H, self.cam.W)
        else:
            views = self.scene.getPresetCameras(preset)
        
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8

        if self.for_gradio:
            iterable_render = progress.tqdm(views, desc='[4/4] Rendering a video')
        else:
            iterable_render = views

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth']
            framelist.append(
                np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depth = -(depth * (depth > 0)).detach().cpu().numpy()
            dmin_local = depth.min().item()
            dmax_local = depth.max().item()
            if dmin_local < dmin:
                dmin = dmin_local
            if dmax_local > dmax:
                dmax = dmax_local
            depthlist.append(depth)

        progress(1, desc='[4/4] Rendering a video...')

        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=60, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=60, quality=8)
        return videopath, depthpath

    def training(self, progress=gr.Progress()):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        if self.for_gradio:
            iterable_gauss = progress.tqdm(range(1, self.opt.iterations + 1), desc='[3/4] Baking Gaussians')
        else:
            iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # import pdb; pdb.set_trace()
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

            image2 = np.array(Image.fromarray(image_in).resize((self.cam.W, self.cam.H))).astype(float) / 255.0
            mask2 = np.array(Image.fromarray(mask_in).resize((self.cam.W, self.cam.H))).astype(float) / 255.0
            image_curr = self.rgb(
                prompt=prompt,
                image=image2,
                negative_prompt=negative_prompt, generator=generator,
                mask_image=mask2,
            )

        else: # if there is a large gap between height and width, do inpainting
            if w_in > h_in:
                image_curr = rgb_cond.crop((int(w_in/2-h_in/2), 0, int(w_in/2+h_in/2), h_in)).resize((self.cam.W, self.cam.H))
            else: # w <= h
                image_curr = rgb_cond.crop((0, int(h_in/2-w_in/2), w_in, int(h_in/2+w_in/2))).resize((self.cam.W, self.cam.H))

        render_poses = get_pcdGenPoses(pcdgenpath)
        depth_curr = self.d(image_curr)
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
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        if self.for_gradio:
            progress(0, desc='[1/4] Dreaming...')
            iterable_dream = progress.tqdm(range(1, len(render_poses)), desc='[1/4] Dreaming')
        else:
            iterable_dream = range(1, len(render_poses))

        for i in iterable_dream:
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]

            ### Transform world to pixel
            pts_coord_cam2 = R.dot(pts_coord_world) + T  ### Same with c2w*world_coord (in homogeneous space)
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)   #.reshape(3,H,W).transpose(1,2,0).astype(np.float32)

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
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]  # use valid_idx[border_valid_idx] for world1

            image_curr = self.rgb(
                prompt=prompt, image=image2, #Image.fromarray(np.round(image2*255.).astype(np.uint8)),
                negative_prompt=negative_prompt, generator=generator, num_inference_steps=diff_steps,
                mask_image=mask2, #Image.fromarray(np.round((1-mask2[:,:])*255.).astype(np.uint8))
            )
            depth_curr = self.d(image_curr)


            ### depth optimize
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
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) # 3, 1
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

            compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
            compensate_depth_zero = np.zeros(4)
            compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4

            pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
            pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2

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

            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
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

        # render_poses = get_pcdGenPoses(pcdgenpath)
        internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth})

        if self.for_gradio:
            progress(0, desc='[2/4] Aligning...')
            iterable_align = progress.tqdm(range(len(render_poses)), desc='[2/4] Aligning')
        else:
            iterable_align = range(len(render_poses))

        for i in iterable_align:
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
                if len(valid_idxj) == 0:
                    continue
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)


                x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
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

        progress(1, desc='[3/4] Baking Gaussians...')
        return traindata
