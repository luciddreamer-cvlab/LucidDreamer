import os
import argparse
from PIL import Image

from luciddreamer import LucidDreamer


if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LucidDreamer')
    # Input options
    parser.add_argument('--image', '-img', type=str, default='examples/Image015_animelakehouse.jpg', help='Input image for scene generation')
    parser.add_argument('--text', '-t', type=str, default='examples/Image015_animelakehouse.txt', help='Text prompt for scene generation')
    parser.add_argument('--neg_text', '-nt', type=str, default='', help='Negative text prompt for scene generation')

    # Camera options
    parser.add_argument('--campath_gen', '-cg', type=str, default='LookDown', choices=['LookDown', 'LookAround', 'Rotate_360'], help='Camera extrinsic trajectories for scene generation')
    parser.add_argument('--campath_render', '-cr', type=str, default='LLFF', choices=['Back_and_forth', 'LLFF', 'Headbanging'], help='Camera extrinsic trajectories for video rendering')

    # Inpainting options
    parser.add_argument('--seed', type=int, default=1, help='Manual seed for running Stable Diffusion inpainting')
    parser.add_argument('--diff_steps', type=int, default=50, help='Number of inference steps for running Stable Diffusion inpainting')

    # Save options
    parser.add_argument('--save_dir', '-s', type=str, default='', help='Save directory')

    args = parser.parse_args()


    ### input (example)
    rgb_cond = Image.open(args.image)

    if args.text.endswith('.txt'):
        with open(args.text, 'r') as f:
            txt_cond = f.readline()
    else:
        txt_cond = args.text

    if args.neg_text.endswith('.txt'):
        with open(args.neg_text, 'r') as f:
            neg_txt_cond = f.readline()
    else:
        neg_txt_cond = args.neg_text

    # Make default save directory if blank
    if args.save_dir == '':
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        args.save_dir = f'./results/{img_name}_{args.campath_gen}_{args.seed}'

    ld = LucidDreamer(args.save_dir)
    ld.create(rgb_cond, txt_cond, neg_txt_cond, args.campath_gen, args.seed, args.diff_steps)
    ld.render_video(args.campath_render)
