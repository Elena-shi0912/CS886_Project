#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import cv2
import logging
import math
import os
import random
import shutil
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import DictConfig

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from util import scale_tensors

if is_wandb_available():
    import wandb



logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs



def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)



def main(
    run_name: str,
    config: DictConfig,):
    
    model_args = config.model
    train_args = config.train
    data_args = config.data

    logging_dir = os.path.join(model_args.output_dir, config.logging.dir)
    logging_dir = os.path.join(logging_dir, run_name)

    if model_args.output_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "models"), exist_ok=True)


    accelerator_project_config = ProjectConfiguration(project_dir=logging_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name=config.logging.wandb_project,
        config=OmegaConf.to_container(config),
        init_kwargs={"wandb": {"name": run_name, "dir": logging_dir}},
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if train_args.seed is not None:
        set_seed(train_args.seed)

    
    # load scheduler, tokenizer and models
    noise_scheduler = DDPMScheduler.from_pretrained(model_args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_args.model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_args.model_path, subfolder="unet")
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    
    if train_args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()


    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

        
    from datasets import Dataset
    lines = open(data_args.train_dataset_index_file).readlines()
    random.shuffle(lines)
    train_dataset = Dataset.from_dict({"image": lines, "text": lines}) 
    dataset = {
        'train': train_dataset,
    }

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names


    dataset_name_mapping = {
        "MARIO-10M": ("image", "text"), 
    }
    
    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            
            caption = caption.strip()
            first, second = caption.split('_')
            try:
                caption = open(f'{data_args.dataset_path}/{first}/{second}/caption.txt').readlines()[0]
            except:
                caption = 'null'
                print('erorr of caption')
                
            if data_args.drop_caption and is_train and random.random() < 0.1: 
                caption = '' # drop caption with 10% probability
                        
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    # Please not that Crop is not suitable for this task as texts may be incomplete during cropping
    train_transforms = transforms.Compose( 
        [
            transforms.ToTensor(),
        ]
    )
    
    
    def generate_random_rectangles(image):
        # randomly generate 0~3 masks
        rectangles = []
        box_num = random.randint(0, 3)
        for i in range(box_num):
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            w = random.randint(16, 256)
            h = random.randint(16, 96) 
            angle = random.randint(-45, 45)
            p1 = (x, y)
            p2 = (x + w, y)
            p3 = (x + w, y + h)
            p4 = (x, y + h)
            center = ((x + x + w) / 2, (y + y + h) / 2)
            p1 = rotate_point(p1, center, angle)
            p2 = rotate_point(p2, center, angle)
            p3 = rotate_point(p3, center, angle)
            p4 = rotate_point(p4, center, angle)
            rectangles.append((p1, p2, p3, p4))
        return rectangles


    def rotate_point(point, center, angle):
        # rotation
        angle = math.radians(angle)
        x = point[0] - center[0]
        y = point[1] - center[1]
        x1 = x * math.cos(angle) - y * math.sin(angle)
        y1 = x * math.sin(angle) + y * math.cos(angle)
        x1 += center[0]
        y1 += center[1]
        return int(x1), int(y1)


    def box2point(box):
        # convert string to list
        box = box.split(',')
        box = [int(i)//(512//512) for i in box] 
        points = [(box[0],box[1]),(box[2],box[3]),(box[4],box[5]),(box[6],box[7])]
        return points
    
    
    def get_mask(ocrs):
        # the two branches are trained at a certain ratio
        if random.random() <= data_args.mask_all_ratio: 
            image_mask = Image.new('L', (512,512), 1)
            return image_mask
        
        image_mask = Image.new('L', (512,512), 0)
        draw_image_mask = ImageDraw.ImageDraw(image_mask)
        for ocr in ocrs:
            ocr = ocr.strip()
            _, box, _ = ocr.split()
            if random.random() < 0.5: # each box is masked with 50% probability
                points = box2point(box)
                draw_image_mask.polygon(points, fill=1) 
        
        blank = Image.new('RGB', (512, 512), (0, 0, 0))
        rectangles = generate_random_rectangles(blank) # get additional masks (can mask non-text areas)
        for rectangle in rectangles:
            draw_image_mask.polygon(rectangle, fill=1)
        
        return image_mask


    def preprocess_train(examples):
        # preprocess the training data
                
        images = []
        segmentation_masks = []
        image_masks = []
        for image in examples[image_column]:
            image = image.strip()
            first, second = image.split('_')
            image_path = f'{data_args.dataset_path}/{first}/{second}/{first}_{second}_stretched.jpg'
            ocrs = open(f'{data_args.dataset_path}/{first}/{second}/ocr.txt').readlines() 
            
            image = Image.open(image_path).convert("RGB")
            
            image_mask = get_mask(ocrs)
            image_mask_np = np.array(image_mask)
            image_mask_tensor = torch.from_numpy(image_mask_np)
            images.append(image) 
            
            if data_args.no_pos_con:
                segmentation_mask = np.load(f'{data_args.dataset_path}/{first}/{second}/charseg.npy') * 0 
            elif data_args.no_con:
                segmentation_mask = (np.load(f'{data_args.dataset_path}/{first}/{second}/charseg.npy') > 0).astype(np.float32) 
            else:
                segmentation_mask = np.load(f'{data_args.dataset_path}/{first}/{second}/charseg.npy') 

            if data_args.segmentation_mask_aug: # 10% dilate / 10% erode / 10% drop
                random_value = random.random()
                if random_value < 0.6:
                    pass
                elif random_value < 0.7:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=1)
                elif random_value < 0.8:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=1)
                elif random_value < 0.85:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=2)
                elif random_value < 0.9:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=2)
                else:
                    drop_mask = np.random.rand(*segmentation_mask.shape) < 0.1
                    segmentation_mask[drop_mask] = 0 # set character to non-character with 10% probability
            
            segmentation_mask = Image.fromarray(segmentation_mask.astype(np.uint8))
            segmentation_masks.append(segmentation_mask)
            segmentation_masks = [image.convert("RGB") for image in segmentation_masks]
            conditioning_image_transforms = transforms.Compose(
                [
                    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                ]
            )
            segmentation_masks = [conditioning_image_transforms(image) for image in segmentation_masks]
            image_masks.append(image_mask_tensor)
            
        examples["images"] = [train_transforms(image).sub_(0.5).div_(0.5) for image in images] 
        examples["prompts"] = tokenize_captions(examples)
        examples["segmentation_masks"] = segmentation_masks
        examples["image_masks"] = image_masks 
        
        return examples

    with accelerator.main_process_first():
        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=train_args.seed).select(range(train_args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples): 
        images = torch.stack([example["images"] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()
        prompts = torch.stack([example["prompts"] for example in examples])
        image_masks = torch.cat([example["image_masks"].unsqueeze(0) for example in examples],0)
        #segmentation_masks = torch.cat([torch.from_numpy(example["segmentation_masks"]).unsqueeze(0).unsqueeze(0) for example in examples], dim=0)
        segmentation_masks = torch.stack([example["segmentation_masks"] for example in examples])
        segmentation_masks = segmentation_masks.to(memory_format=torch.contiguous_format).float()
        return {"images": images, "prompts": prompts, "segmentation_masks": segmentation_masks, "image_masks": image_masks}


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_args.batch_size,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = train_args.batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if train_args.resume_from_checkpoint:
        path = os.path.join(config.logging.dir, train_args.resume_from_checkpoint)

        if path is None:
            accelerator.print(
                f"Checkpoint '{train_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(model_args.output_dir, path))
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * train_args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * train_args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    ce_criterion = torch.nn.CrossEntropyLoss()
    
    # import segmenter for calculating loss
    from models.text_segmenter.unet import UNet
    from collections import OrderedDict
    
    segmenter = UNet(4,96, True).cuda() 
    state_dict = torch.load(model_args.character_aware_loss_ckpt, map_location='cpu') 
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    segmenter.load_state_dict(new_state_dict)
    segmenter.eval()
    
    image_logs = None
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # get image mask
                image_masks = batch["image_masks"]
                
                # apply mask to image
                masked_images = batch["images"] * (1 - image_masks).unsqueeze(1)
                masked_latents = vae.encode(masked_images.to(dtype=weight_dtype)).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                # get segmentation mask
                segmentation_masks = batch["segmentation_masks"]
                # image_masks_256 = F.interpolate(image_masks.unsqueeze(1), size=(256, 256), mode='nearest')
                # segmentation_masks = image_masks_256 * segmentation_masks 
                image_masks_512 = F.interpolate(image_masks.unsqueeze(1), size=(512, 512), mode='nearest')
                segmentation_masks = image_masks_512 * segmentation_masks
                # latents_masks = F.interpolate(image_masks.unsqueeze(1), size=(64, 64), mode='nearest')

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompts"])[0]
                
                # let controlnet condition on segmentation mask ONLY
                controlnet_image = segmentation_masks.to(dtype=weight_dtype)
                #print(noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape, controlnet_image.shape)
                

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                def get_x0_from_noise(noise_scheduler, noise, t, x_t): # add this function
                    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noise.device, dtype=noise.dtype)
                    x_0 = 1 / torch.sqrt(noise_scheduler.alphas_cumprod[t][:,None,None,None]) * x_t  -  torch.sqrt(1 / noise_scheduler.alphas_cumprod[t][:,None,None,None] - 1) * noise
                    return x_0
                
                pred_x0 = get_x0_from_noise(noise_scheduler, model_pred, timesteps, noisy_latents)
                resized_charmap = F.interpolate(batch["segmentation_masks"].float(), size=(64, 64), mode="nearest").long()
                
                # if train_args.use_ocr:
                #     # import the ocr
                #     import easyocr
                #     reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
                #     result = reader.readtext('chinese.jpg')
                # log images in diffusion process
                pred_image = vae.decode(pred_x0.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                log_images = {
                    "target_images": batch["images"],
                    "output_images": pred_image,
                    "controlnet condition": controlnet_image,
                }

                for k, tensor in log_images.items():
                    if tensor is None:
                        continue
                    log_images[k] = [wandb.Image(tensor[i, :, :, :]) for i in range(tensor.shape[0])]
                
                ce_loss = ce_criterion(segmenter(pred_x0.float()), resized_charmap.squeeze(1))
                mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
                loss = mse_loss + ce_loss * train_args.character_aware_loss_lambda 
                
                avg_loss = accelerator.gather(loss.repeat(train_args.train_batch_size)).mean()
                train_loss += avg_loss.item() / train_args.gradient_accumulation_steps
                
                accelerator.log(
                    {"train_loss": train_loss, "ce_loss": ce_loss, "mse_loss": mse_loss, **log_images},
                    step=global_step,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, train_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=train_args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % train_args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if train_args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(model_args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= train_args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - train_args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(model_args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(model_args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            #accelerator.log(logs, step=global_step)

            if global_step >= train_args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(model_args.output_dir)

        

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(run_name=args.name, config=config)
    