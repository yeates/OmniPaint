#!/usr/bin/env python
"""
OmniPaint object insertion CLI using static text embeddings.

This script inserts a subject into a background within a masked region,
using precomputed text embeddings loaded from NPZ for deterministic behavior.
"""

import os
import sys
# Ensure project root (parent of scripts/) is on sys.path for `from src ...` imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Tuple, Optional
import threading

import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers.pipelines import FluxPipeline
from carvekit.api.high import HiInterface

from src.condition import Condition
from src.generate import generate, seed_everything
from src.embedding_loader import load_npz_embeddings


MAX_LENGTH = 1024
DEFAULT_LORA = "weights/omnipaint_insert.safetensors"
DEFAULT_EMBED_PATH = "./demo_assets/embeddings/insert.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="OmniPaint object insertion (static embeddings)")
    parser.add_argument("--background", type=str, required=True, help="Background image file or directory")
    parser.add_argument("--mask", type=str, required=True, help="Mask image file or directory (white=insertion area)")
    parser.add_argument("--subject", type=str, required=True, help="Subject image file or directory to insert")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for the model, e.g. cuda:0")
    parser.add_argument("--carvekit_device", type=str, default="cuda:0", help="Device for CarveKit background removal")
    parser.add_argument("--lora_weights", type=str, default=DEFAULT_LORA, help="Path to LoRA weights for insertion")
    parser.add_argument("--embed_path", type=str, default=DEFAULT_EMBED_PATH, help="Path to NPZ embeddings to use")
    parser.add_argument("--skip_bg_removal", action="store_true", help="Skip background removal for subject images")
    return parser.parse_args()


def resize_with_aspect_ratio(img: Image.Image, max_length: int = MAX_LENGTH) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_length:
        scale = max_length / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def load_model(device: str, lora_weights: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights(lora_weights, adapter_name="insertion")
    pipe.set_adapters(["insertion"])
    return pipe


def init_carvekit(device: str) -> Tuple[HiInterface, threading.Lock]:
    carvekit_lock = threading.Lock()
    carvekit_interface = HiInterface(
        object_type="object",
        batch_size_seg=5,
        batch_size_matting=1,
        device=device,
        seg_mask_size=640,
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False,
    )
    return carvekit_interface, carvekit_lock


def _get_device(pipe):
    return getattr(pipe, "_execution_device", getattr(pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")))


def load_static_embeddings(pipe: FluxPipeline, embed_path: str):
    device = _get_device(pipe)
    dtype = pipe.transformer.dtype
    return load_npz_embeddings(embed_path, device=device, dtype=dtype)


def remove_background(input_image_path: str, carvekit_interface: HiInterface, carvekit_lock: threading.Lock) -> Image.Image:
    with carvekit_lock:
        images_without_background = carvekit_interface([input_image_path])
        cat_wo_bg = np.array(images_without_background[0])
        rgb = cat_wo_bg[..., :3]
        alpha = cat_wo_bg[..., 3:] / 255.0
        white_bg = np.ones_like(rgb) * 255
        blended = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(blended)


def process_single_image(
    pipe: FluxPipeline,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    background_path: str,
    mask_path: str,
    subject_path: str,
    output_path: str,
    seed: int,
    num_steps: int,
    carvekit_interface: Optional[HiInterface] = None,
    carvekit_lock: Optional[threading.Lock] = None,
    skip_bg_removal: bool = False,
) -> None:
    condition_img = Image.open(background_path).convert("RGB")
    condition_img = resize_with_aspect_ratio(condition_img)

    if skip_bg_removal:
        subject_img = Image.open(subject_path).convert("RGB")
    else:
        if carvekit_interface is None or carvekit_lock is None:
            raise ValueError("CarveKit not initialized but background removal requested")
        subject_img = remove_background(subject_path, carvekit_interface, carvekit_lock)
    subject_img = subject_img.resize((512, 512), Image.Resampling.LANCZOS)

    mask = Image.open(mask_path).convert("L").resize(condition_img.size, Image.Resampling.LANCZOS)
    inverted_mask = ImageOps.invert(mask)

    black_background = Image.new("RGB", condition_img.size, (0, 0, 0))
    composite_img = Image.composite(condition_img, black_background, inverted_mask)

    conditions = [
        Condition("insertion", composite_img),
        Condition("insertion", subject_img, position_delta=(0, -32)),
    ]
    seed_everything(seed)

    width = (composite_img.size[0] // 8) * 8
    height = (composite_img.size[1] // 8) * 8

    result_img = generate(
        pipe,
        conditions=conditions,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        prompt=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
    ).images[0]

    result_img = result_img.resize(condition_img.size, Image.Resampling.LANCZOS)
    result_img.save(output_path)


def get_image_triplets(background_path: str, mask_path: str, subject_path: str) -> list[Tuple[str, str, str, str]]:
    triplets = []
    if os.path.isfile(background_path):
        if not os.path.isfile(mask_path):
            raise ValueError(f"Mask file not found: {mask_path}")
        if not os.path.isfile(subject_path):
            raise ValueError(f"Subject file not found: {subject_path}")
        base_name = Path(background_path).stem
        triplets.append((background_path, mask_path, subject_path, f"{base_name}_inserted.png"))
    else:
        if not os.path.isdir(background_path):
            raise ValueError(f"Background directory not found: {background_path}")
        if not os.path.isdir(mask_path):
            raise ValueError(f"Mask directory not found: {mask_path}")
        if not os.path.isdir(subject_path):
            raise ValueError(f"Subject directory not found: {subject_path}")
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        # Support either aligned filenames or numeric-suffixed demo names
        for bg_file in sorted(os.listdir(background_path)):
            bg_path = os.path.join(background_path, bg_file)
            if not (os.path.isfile(bg_path) and Path(bg_file).suffix.lower() in img_exts):
                continue
            base_name = Path(bg_file).stem
            # Try direct basename match first
            candidates = []
            for ext in img_exts:
                candidates.append((os.path.join(mask_path, f"{base_name}{ext}"), os.path.join(subject_path, f"{base_name}{ext}")))
            # Also try demo-style: background-XX -> mask-XX, subject-XX
            if base_name.startswith("background-"):
                suffix = base_name.split("background-")[-1]
                for ext in img_exts:
                    candidates.append((os.path.join(mask_path, f"mask-{suffix}{ext}"), os.path.join(subject_path, f"subject-{suffix}{ext}")))
            mask_full_path = None
            subject_full_path = None
            for mask_candidate, subj_candidate in candidates:
                if mask_full_path is None and os.path.isfile(mask_candidate):
                    mask_full_path = mask_candidate
                if subject_full_path is None and os.path.isfile(subj_candidate):
                    subject_full_path = subj_candidate
                if mask_full_path and subject_full_path:
                    break
            if mask_full_path and subject_full_path:
                triplets.append((bg_path, mask_full_path, subject_full_path, f"{base_name}_inserted.png"))
            else:
                missing = []
                if not mask_full_path:
                    missing.append("mask")
                if not subject_full_path:
                    missing.append("subject")
                print(f"Warning: Missing {', '.join(missing)} for {bg_file}, skipping...")
    return triplets


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_model(args.device, args.lora_weights)
    prompt_embeds, pooled_prompt_embeds, text_ids = load_static_embeddings(pipe, args.embed_path)
    carvekit_interface = None
    carvekit_lock = None
    if not args.skip_bg_removal:
        carvekit_interface, carvekit_lock = init_carvekit(args.carvekit_device)
    triplets = get_image_triplets(args.background, args.mask, args.subject)
    if not triplets:
        raise ValueError("No valid image triplets found")
    for bg_path, mask_path, subject_path, output_name in tqdm(triplets, desc="Processing images"):
        output_path = os.path.join(args.output_dir, output_name)
        try:
            process_single_image(
                pipe=pipe,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                background_path=bg_path,
                mask_path=mask_path,
                subject_path=subject_path,
                output_path=output_path,
                seed=args.seed,
                num_steps=args.steps,
                carvekit_interface=carvekit_interface,
                carvekit_lock=carvekit_lock,
                skip_bg_removal=args.skip_bg_removal,
            )
        except Exception as e:
            print(f"Error processing {bg_path}: {str(e)}")
            continue
    print(f"\nProcessing complete. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()


