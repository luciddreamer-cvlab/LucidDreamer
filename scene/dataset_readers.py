#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import json
from typing import NamedTuple
from pathlib import Path

import imageio
import torch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.gaussian_model import BasicPointCloud
from scene.cameras import MiniCam, Camera
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics import getWorld2View2, focal2fov, fov2focal
from utils.graphics import getProjectionMatrix
from utils.trajectory import get_camerapaths
from utils.sh import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    preset_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    idx = np.random.choice(len(vertices['x']),size=(min(len(vertices['x']), 100_000),),replace=False)
    positions = np.vstack([vertices['x'][idx], vertices['y'][idx], vertices['z'][idx]]).T               if 'x' in vertices else None
    colors = np.vstack([vertices['red'][idx], vertices['green'][idx], vertices['blue'][idx]]).T / 255.0 if 'red' in vertices else None
    normals = np.vstack([vertices['nx'][idx], vertices['ny'][idx], vertices['nz'][idx]]).T              if 'nx' in vertices else None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, preset=None, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 5 == 2 or idx % 5 == 0]
        test_cam_infos  = [c for idx, c in enumerate(cam_infos) if not (idx % 5 == 2 or idx % 5 == 0)]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if preset:
        preset_cam_infos = readCamerasFromPreset('/home/chung/workspace/gaussian-splatting/poses_supplementary', f"{preset}.json")
    else:
        preset_cam_infos = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           preset_cameras=preset_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            # if os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy')):
            #     depth = np.load(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy'))
            #     if os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png')):
            #         mask = imageio.v3.imread(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png'))[:,:,0]/255.
            #     else:
            #         mask = np.ones_like(depth)
            #     final_depth = depth*mask
            # else:
            #     final_depth = None
                
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


def readCamerasFromPreset(path, transformsfile):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        FOV = contents["camera_angle_x"]*1.2

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(np.concatenate((c2w, np.array([0,0,0,1]).reshape(1,4)), axis=0))
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # R = c2w[:3,:3]
            # T = - np.transpose(R).dot(c2w[:3,3])

            image = Image.fromarray(np.zeros((512,512)), "RGB")
            FovY = focal2fov(fov2focal(FOV, 512), image.size[0]) 
            FovX = focal2fov(fov2focal(FOV, 512), image.size[1]) 
            # FovX, FovY = contents["camera_angle_x"], contents["camera_angle_x"]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='None', image_name='None', width=image.size[1], height=image.size[0]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, preset=None, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if preset:
        preset_cam_infos = readCamerasFromPreset('/home/chung/workspace/gaussian-splatting/poses_supplementary', f"{preset}.json")
    else:
        preset_cam_infos = None
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           preset_cameras=preset_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def loadCamerasFromData(traindata, white_background):
    cameras = []

    fovx = traindata["camera_angle_x"]
    frames = traindata["frames"]
    for idx, frame in enumerate(frames):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = frame["image"] if "image" in frame else None
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        loaded_mask = np.ones_like(norm_data[:, :, 3:4])

        fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx

        image = torch.Tensor(arr).permute(2,0,1)
        loaded_mask = None #torch.Tensor(loaded_mask).permute(2,0,1)
        
        ### torch로 바꿔야함 
        cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, 
                                gt_alpha_mask=loaded_mask, image_name='', uid=idx, data_device='cuda'))
            
    return cameras


def loadCameraPreset(traindata, presetdata):
    cam_infos = {}
    ## camera setting (for H, W and focal)
    fovx = traindata["camera_angle_x"] * 1.2
    W, H = traindata["frames"][0]["image"].size
    # W, H = traindata["W"], traindata["H"]

    for camkey in presetdata:
        cam_infos[camkey] = []
        for idx, frame in enumerate(presetdata[camkey]["frames"]):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fovy = focal2fov(fov2focal(fovx, W), H)
            FovY = fovy 
            FovX = fovx
        
            znear, zfar = 0.01, 100
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

            cam_infos[camkey].append(MiniCam(width=W, height=H, fovy=FovY, fovx=FovX, znear=znear, zfar=zfar,
                                          world_view_transform=world_view_transform, full_proj_transform=full_proj_transform))
            
    return cam_infos


def readDataInfo(traindata, white_background):
    print("Reading Training Transforms")
    
    train_cameras = loadCamerasFromData(traindata, white_background)
    preset_minicams = loadCameraPreset(traindata, presetdata=get_camerapaths())
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cameras)

    pcd = BasicPointCloud(points=traindata['pcd_points'].T, colors=traindata['pcd_colors'], normals=None)

    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras,
                           test_cameras=[],
                           preset_cameras=preset_minicams,
                           nerf_normalization=nerf_normalization,
                           ply_path='')
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}