#!/usr/bin/env python
"""
OmniPaint object removal CLI using static text embeddings.

This script removes objects within a masked region using precomputed text
embeddings loaded from NPZ for deterministic behavior.
"""

import os
import sys
# Ensure project root (parent of scripts/) is on sys.path for `from src ...` imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers.pipelines import FluxPipeline

from src.condition import Condition
from src.generate import generate, seed_everything
from src.embedding_loader import load_npz_embeddings


MAX_LENGTH = 1024
DEFAULT_LORA = "weights/omnipaint_remove.safetensors"
DEFAULT_EMBED_PATH = "./demo_assets/embeddings/remove.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="OmniPaint object removal (static embeddings)")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--mask", type=str, required=True, help="Mask image file or directory (white=object to remove)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for the model, e.g. cuda:0")
    parser.add_argument("--lora_weights", type=str, default=DEFAULT_LORA, help="Path to LoRA weights for removal")
    parser.add_argument("--embed_path", type=str, default=DEFAULT_EMBED_PATH, help="Path to NPZ embeddings to use")
    return parser.parse_args()


def resize_with_aspect_ratio(img: Image.Image, max_length: int = MAX_LENGTH) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_length:
        scale = max_length / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def load_model(device: str, lora_weights: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights(lora_weights, adapter_name="removal")
    pipe.set_adapters(["removal"])
    return pipe


def _get_device(pipe):
    return getattr(pipe, "_execution_device", getattr(pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")))


def load_static_embeddings(pipe: FluxPipeline, embed_path: str):
    device = _get_device(pipe)
    dtype = pipe.transformer.dtype
    return load_npz_embeddings(embed_path, device=device, dtype=dtype)


def process_single_image(
    pipe: FluxPipeline,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    image_path: str,
    mask_path: str,
    output_path: str,
    seed: int,
    num_steps: int,
) -> None:
    condition_img = Image.open(image_path).convert("RGB")
    condition_img = resize_with_aspect_ratio(condition_img)

    mask = Image.open(mask_path).convert("L").resize(condition_img.size, Image.Resampling.LANCZOS)
    inverted_mask = ImageOps.invert(mask)

    black_background = Image.new("RGB", condition_img.size, (0, 0, 0))
    composite_img = Image.composite(condition_img, black_background, inverted_mask)

    condition = Condition("removal", composite_img)
    seed_everything(seed)

    width = (condition_img.size[0] // 8) * 8
    height = (condition_img.size[1] // 8) * 8

    result_img = generate(
        pipe,
        conditions=[condition],
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


def get_image_pairs(input_path: str, mask_path: str) -> list[Tuple[str, str, str]]:
    pairs = []
    if os.path.isfile(input_path):
        if not os.path.isfile(mask_path):
            raise ValueError(f"Mask file not found: {mask_path}")
        base_name = Path(input_path).stem
        pairs.append((input_path, mask_path, f"{base_name}_removed.png"))
    else:
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path not found: {input_path}")
        if not os.path.isdir(mask_path):
            raise ValueError(f"Mask directory not found: {mask_path}")
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img_file in sorted(os.listdir(input_path)):
            img_path = os.path.join(input_path, img_file)
            if not (os.path.isfile(img_path) and Path(img_file).suffix.lower() in img_exts):
                continue
            base_name = Path(img_file).stem
            candidates = []
            # Direct basename match
            for ext in img_exts:
                candidates.append(os.path.join(mask_path, f"{base_name}{ext}"))
            # Demo-style: 5.jpg -> 5.png within removal_samples
            if base_name.isdigit():
                for ext in img_exts:
                    candidates.append(os.path.join(mask_path, f"{base_name}{ext}"))
            mask_full_path = None
            for candidate in candidates:
                if os.path.isfile(candidate):
                    mask_full_path = candidate
                    break
            if mask_full_path:
                pairs.append((img_path, mask_full_path, f"{base_name}_removed.png"))
            else:
                print(f"Warning: No mask found for {img_file}, skipping...")
    return pairs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_model(args.device, args.lora_weights)
    prompt_embeds, pooled_prompt_embeds, text_ids = load_static_embeddings(pipe, args.embed_path)
    pairs = get_image_pairs(args.input, args.mask)
    if not pairs:
        raise ValueError("No valid image-mask pairs found")
    for img_path, mask_path, output_name in tqdm(pairs, desc="Processing images"):
        output_path = os.path.join(args.output_dir, output_name)
        try:
            process_single_image(
                pipe=pipe,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                image_path=img_path,
                mask_path=mask_path,
                output_path=output_path,
                seed=args.seed,
                num_steps=args.steps,
            )
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    print(f"\nProcessing complete. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()


