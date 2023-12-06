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
import gradio as gr
from luciddreamer import LucidDreamer


css = """
#run-button {
  background: coral;
  color: white;
}
"""

ld = LucidDreamer()

with gr.Blocks(css=css) as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div>
            <h1 >LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes</h1>
            <h5 style="margin: 0;">If you like our project, please visit our Github, too! ✨✨✨ More features are waiting!</h5>
            </br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href='https://arxiv.org/abs/2311.13384'>
                    <img src="https://img.shields.io/badge/Arxiv-2311.13384-red">
                </a>
                &nbsp;
                <a href='https://luciddreamer-cvlab.github.io'>
                    <img src='https://img.shields.io/badge/Project-LucidDreamer-green' alt='Project Page'>
                </a>
                &nbsp;
                <a href='https://github.com/ironjr/LucidDreamer'>
                    <img src='https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer?label=Github&color=blue'>
                </a>
                &nbsp;
                <a href='https://twitter.com/_ironjr_'>
                    <img src='https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_'>
                </a>
            </div>
        </div>
        </div>
        """
    )

    with gr.Row():

        result_gallery = gr.Video(label='Custom View Video', show_label=True)

        result_default_gallery = gr.Video(label='Default View 3D', show_label=True)

        result_ply_file = gr.File(label='Gaussian splatting PLY', show_label=True)


    with gr.Row():

        input_image = gr.Image(
            label='Image prompt',
            sources='upload',
            type='pil',
        )

        with gr.Column():
            prompt = gr.Textbox(
                label='Text prompt',
                value='A cozy livingroom',
            )
            n_prompt = gr.Textbox(
                label='Negative prompt',
                value='photo frame, frame, boarder, simple color, inconsistent',
            )
            gen_camerapath = gr.Radio(
                label='Camera trajectory for scene generation',
                choices=['Rotate_360', 'LookAround', 'LookDown'],
            )
            seed = gr.Slider(
                label='Seed',
                minimum=1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )
            diff_steps = gr.Slider(
                label='SD inpainting steps',
                minimum=1,
                maximum=50,
                step=1,
                value=30,
            )
            render_camerapath = gr.Radio(
                label='Camera trajectory for video rendering',
                choices=['Back_and_forth', 'LLFF', 'Headbanging'],
            )

        with gr.Column():
            run_button = gr.Button(value='Run! (it may take a while)', elem_id='run-button')

            gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <div>
                    <h3>...or you can run in two steps</h3>
                    <h4>(hint: press STEP 2 if you have already baked Gaussians).</h4>
                </div>
                </div>
                """
            )

            with gr.Row():
                gaussian_button = gr.Button(value='STEP 1: Get Gaussians')
                render_button = gr.Button(value='STEP 2: Render A Video')

    ips = [input_image, prompt, n_prompt, gen_camerapath, seed, diff_steps, render_camerapath]

    run_button.click(fn=ld.run, inputs=ips, outputs=[result_ply_file, result_default_gallery, result_gallery])
    gaussian_button.click(fn=ld.create, inputs=ips[:-1], outputs=[result_ply_file, result_default_gallery])
    render_button.click(fn=ld.render_video, inputs=ips[-1:], outputs=[result_gallery])

    gr.Examples(
        examples=[
            [
                'examples/Image015_animelakehouse.jpg',
                'anime style, animation, best quality, a boat on lake, trees and rocks near the lake. a house and port in front of a house',
                'photo frame, frame, boarder, simple color, inconsistent',
                'LookDown',
                1,
                50,
                'Back_and_forth',
            ],
            [
                'examples/Image003_fantasy.jpg',
                'A vibrant, colorful floating community city, clouds above a beautiful, enchanted landscape filled with whimsical flora, enchanted forest landscape, Magical and dreamy woodland with vibrant green foliage and sparkling flowers, Landscape with twisted trees and vines, natural lighting and dark shadows, unique fantastical elements like floating islands and floating orbs, Highly detailed vegetation and foliage, deep contrast and color vibrancy,', # texture and intricate details in a floating element',
                'photo frame, frame, boarder, simple color, inconsistent',
                'LookAround',
                4,
                50,
                'Back_and_forth',
            ],
            [
                'examples/image020.png',
                'High-resolution photography kitchen design, wooden floor, small windows opening onto the garden, Bauhaus furniture and decoration, high ceiling, beige blue salmon pastel palette, interior design magazine, cozy atmosphere; 8k, intricate detail, photorealistic, realistic light, wide angle, kinfolk photography, A+D architecture, Kitchen Sink, Basket of fruits and vegetables, a bottle of drinking water, walls painted magazine style photo, looking towards a sink under a window, with a door on the left of the sink with a 25 cm distance from the kitchen, the kitchen is an L shaped starting from the right corner, on the far right a fridge nest to it a stove, next the dishwasher then the sink, a smokey grey kitchen with modern touches, taupe walls, a taup ceiling with spotlights inside the ceiling with 90 cm distance, wooden parquet floor',
                'photo frame, frame, boarder, simple color, inconsistent',
                'Rotate_360',
                1,
                50,
                'Headbanging',
            ],
        ],
        inputs=ips,
        outputs=[result_ply_file],
        fn=ld.create,
        cache_examples=False,
    )


if __name__ == '__main__':
    demo.queue(max_size=20).launch()
