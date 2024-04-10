# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import argparse
import cv2
import torchvision.transforms as transforms

from termcolor import colored
from models.layout_generator import get_layout_from_prompt
from models.text_segmenter.unet import UNet


from huggingface_hub import HfApi
from pathlib import Path

to_pil_image = transforms.ToPILImage()

ada_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])


def load_stablediffusion():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)      
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe

def test_stablediffusion(prompt, save_path, num_images_per_prompt=4,
                              pipe=None, generator=None):
    images = pipe(prompt, num_inference_steps=50, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))

def load_deepfloyd_if():
    from diffusers import DiffusionPipeline
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    # stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_1.enable_model_cpu_offload()
    stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16",
                                                torch_dtype=torch.float16)
    # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_2.enable_model_cpu_offload()
    safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker,
                      "watermarker": stage_1.watermarker}
    stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules,
                                                torch_dtype=torch.float16)
    # stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_3.enable_model_cpu_offload()
    return stage_1, stage_2, stage_3


def test_deepfloyd_if(stage_1, stage_2, stage_3, prompt, save_path, num_images_per_prompt=4, generator=None):
    idx = num_images_per_prompt - 1  # if the last image of a case exists, then return
    new_save_path = save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_' + str(idx) + '/')
    if os.path.exists(new_save_path):
        return
    if not stage_1 or not stage_2 or not stage_3:
        stage_1, stage_2, stage_3  = load_deepfloyd_if()
    if generator is None:
        generator = torch.manual_seed(0)
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    stage_1.set_progress_bar_config(disable=True)
    stage_2.set_progress_bar_config(disable=True)
    stage_3.set_progress_bar_config(disable=True)
    images = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                    output_type="pt",  num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image = stage_2(image=image.unsqueeze(0), prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                        generator=generator, output_type="pt").images
        image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
        # image = to_pil_image(image[0].cpu())
        new_save_path = save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/')
        image[0].save(new_save_path)


def load_controlnet(pretrained_path):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained(pretrained_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,
                                                             safety_checker=None, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe


def test_controlnet_cannyedge(prompt, save_path, pretrained_path_canny, args, num_images_per_prompt=4,
                              pipe=None, generator=None, low_threshold=100, high_threshold=200):
    '''ref: https://github.com/huggingface/diffusers/blob/131312caba0af97da98fc498dfdca335c9692f8c/docs/source/en/api/pipelines/stable_diffusion/controlnet.mdx'''

    if pipe is None:
        pipe = load_controlnet(pretrained_path_canny)
    
    args.prompt = prompt 
    image, segmentation_mask_from_pillow = get_layout_from_prompt(args)
    
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    control_image.save(save_path.replace('/images/', '/control_images/'))

    images = pipe(prompt, control_image, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))

# def test_controlnet_segmask(prompt, save_path, pretrained_path_seg, args, num_images_per_prompt=4,
#                               pipe=None, generator=None, low_threshold=100, high_threshold=200):
#     from diffusers.utils import load_image
#     if pipe is None:
#         pipe = load_controlnet(pretrained_path_seg)

#     # # load character-level segmenter
#     # segmenter = UNet(3, 96, True).cuda()
#     # segmenter = torch.nn.DataParallel(segmenter)
#     # segmenter.load_state_dict(torch.load(args.character_segmenter_path))
#     # segmenter.eval()
#     # print(f'{colored("[√]", "green")} Text segmenter is successfully loaded.')
#     args.prompt = prompt 
#     _ , segmentation_mask_from_pillow = get_layout_from_prompt(args)
#     segmentation_mask = np.array(segmentation_mask_from_pillow)
#     segmentation_mask = segmentation_mask[:, :, None]
#     segmentation_mask = np.concatenate([segmentation_mask, segmentation_mask, segmentation_mask], axis=2)
#     segmentation_mask = Image.fromarray(segmentation_mask) # (512, 512, 3)


#     images = pipe(prompt, segmentation_mask, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
#     for idx, image in enumerate(images):
#         image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/')



def test_controlnet_segmask(prompt, save_path, pretrained_path_seg, args, num_images_per_prompt=4,
                              pipe=None, generator=None):

    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    if pipe is None:
        pipe = load_controlnet(pretrained_path_seg)

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    args.prompt = prompt 
    image, segmentation_mask_from_pillow = get_layout_from_prompt(args)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
      outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    control_image.save(save_path.replace('/images/', '/control_images/'))


    images = pipe(prompt, control_image, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))

def test_controlnet_segmask_glyph(prompt, save_path, pretrained_path_seg, args, num_images_per_prompt=4,
                              pipe=None, generator=None):

    if pipe is None:
        pipe = load_controlnet(pretrained_path_seg)

    args.prompt = prompt 
    control_image, segmentation_mask_from_pillow = get_layout_from_prompt(args)
    
    control_image.save(save_path.replace('/images/', '/control_images/'))


    images = pipe(prompt, control_image, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))

def test_controlnet_scribble(prompt, save_path, pretrained_path_seg, args, num_images_per_prompt=4,
                              pipe=None, generator=None):

    from controlnet_aux import PidiNetDetector, HEDdetector
    if pipe is None:
        pipe = load_controlnet(pretrained_path_seg)

    args.prompt = prompt 
    image, segmentation_mask_from_pillow = get_layout_from_prompt(args)
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

    control_image = processor(image, scribble=True)

    control_image.save(save_path.replace('/images/', '/control_images/'))

    images = pipe(prompt, control_image, num_inference_steps=20, generator=generator, num_images_per_prompt=num_images_per_prompt).images
    for idx, image in enumerate(images):
        image.save(save_path.replace('.jpg', '_' + str(idx) + '.jpg').replace('/images/', '/images_'+ str(idx) +'/'))



def MARIOEval_generate_results(root, dataset, args, method='controlnet_seg', num_images_per_prompt=4, split=0, total_split=1):
    root_eval = os.path.join(root, "MARIOEval")
    root_res = os.path.join(root, "generation", method)

    for idx in range(num_images_per_prompt): 
        os.makedirs(os.path.join(root_res, dataset, 'images_' + str(idx)), exist_ok=True)
    os.makedirs(os.path.join(root_res, dataset, 'control_images'), exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(0)
    
    if method == 'controlnet_canny':
        pipe = load_controlnet('lllyasviel/control_v11p_sd15_canny')
    elif method == 'controlnet_seg':
        pipe = load_controlnet('lllyasviel/control_v11p_sd15_seg')
    elif method == 'controlnet_seg_glyph':
        pipe = load_controlnet('lllyasviel/control_v11p_sd15_seg')
    elif method == 'controlnet_scribble':
        pipe = load_controlnet('lllyasviel/control_v11p_sd15_scribble')
    elif method == 'stablediffusion':
        pipe = load_stablediffusion()
    elif method == 'deepfloyd':
        stage_1, stage_2, stage_3 = load_deepfloyd_if() 
   
    with open(os.path.join(root_eval, dataset, dataset + '.txt'), 'r') as fr:
        prompts = fr.readlines()
        prompts = [_.strip() for _ in prompts]

    for idx, prompt in tqdm(enumerate(prompts)): 
        if idx < split * len(prompts) / total_split or idx > (split + 1) * len(prompts) / total_split:
            continue
        if  method == 'controlnet_canny':
            test_controlnet_cannyedge(prompt=prompt, pretrained_path_canny='lllyasviel/control_v11p_sd15_canny', num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator, args=args) 
        elif method == 'controlnet_seg':
            test_controlnet_segmask(prompt=prompt, pretrained_path_seg='lllyasviel/control_v11p_sd15_seg', num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator, args=args)
        elif method == 'controlnet_seg_glyph':
            test_controlnet_segmask_glyph(prompt=prompt, pretrained_path_seg='lllyasviel/control_v11p_sd15_seg', num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator, args=args)
        elif method == 'controlnet_scribble':
            test_controlnet_scribble(prompt=prompt, pretrained_path_seg='lllyasviel/control_v11p_sd15_scribble', num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator, args=args)

        elif  method == 'stablediffusion':
            test_stablediffusion(prompt=prompt, num_images_per_prompt=num_images_per_prompt,
                                  save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                                  pipe=pipe, generator=generator) 
        elif method == 'deepfloyd':
            test_deepfloyd_if(stage_1, stage_2, stage_3, num_images_per_prompt=num_images_per_prompt,
                              save_path=os.path.join(root_res, dataset, 'images', str(idx) + '.jpg'),
                              prompt=prompt, generator=generator)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='TMDBEval500',
        required=False,
        choices=['TMDBEval500', 'OpenLibraryEval500', 'LAIONEval4000',
                 'ChineseDrawText', 'DrawBenchText', 'DrawTextCreative']
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/eval",
        required=True, 
    )
    parser.add_argument(
        "--method",
        type=str,
        default='controlnet_seg', 
        required=False,
        choices=['controlnet_canny', 'controlnet_seg','controlnet_seg_glyph','controlnet_scribble', 'deepfloyd', 'stablediffusion', 'textdiffuser']
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--split", 
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--total_split", 
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--font_path", 
        type=str, 
        default='arial.ttf', 
        required=False,
        help="The path of font for visualization."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    MARIOEval_generate_results(root=args.root, dataset=args.dataset, method=args.method,
                               split=args.split, total_split=args.total_split, args=args) 