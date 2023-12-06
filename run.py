from PIL import Image

from luciddreamer import LucidDreamer
import pickle
import imageio


if __name__ == "__main__":
    ### option 
    seed = 1
    move_scenegen = "LookDown"
    move_video = "LLFF" # 'back', 'llff', 'headbanging'

    ### input (example)
    rgb_cond = Image.open("examples/Image015_animelakehouse.jpg")
    txt_cond = "anime style, animation, best quality, a boat on lake, trees and rocks near the lake. a house and port in front of a house"
    neg_txt_cond = ""

    ld = LucidDreamer()
    ld.create(rgb_cond, txt_cond, neg_txt_cond, move_scenegen, seed, diff_steps=50)
    ld.render_video(move_video)
