import torch
import numpy as np
import cv2
import os
import json
import argparse
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def is_mask_mostly_contained(mask_a, mask_b, contain_threshold=0.9):
    intersection = np.logical_and(mask_a, mask_b).sum()
    area_a = mask_a.sum()
    return (intersection / area_a) > contain_threshold if area_a > 0 else False


def filter_masks(masks, iou_threshold=0.8, contain_threshold=0.9, mask_ratio=0.00):
    filtered_masks = []
    masks = sorted(masks, key=lambda x: -x["area"])

    for mask in masks:
        if mask["area"] / (mask["segmentation"].shape[0] * mask["segmentation"].shape[1]) < mask_ratio:
            continue
        keep = True
        for filtered_mask in filtered_masks:
            iou = compute_iou(mask["segmentation"], filtered_mask)
            if iou > iou_threshold or is_mask_mostly_contained(mask["segmentation"], filtered_mask, contain_threshold):
                keep = False
                break
        if keep:
            filtered_masks.append(mask["segmentation"])
    return filtered_masks


def classify_masks(mask_list, reference_mask):
    masks_intersect = []
    masks_contained = []
    for i, mask in enumerate(mask_list):
        intersection = np.logical_and(np.array(mask), np.array(reference_mask))
        if np.all(np.array(mask) <= np.array(reference_mask)):
            masks_contained.append(i)
        elif np.any(intersection):
            masks_intersect.append(i)
    return masks_intersect, masks_contained


def is_adjacent(mask1, mask2):
    kernel = np.ones((5, 5), np.uint8)
    mask1 = mask1.astype(np.uint8) * 255
    mask2 = mask2.astype(np.uint8) * 255
    mask1_dilated = cv2.dilate(mask1, kernel, iterations=3)
    mask2_dilated = cv2.dilate(mask2, kernel, iterations=3)
    return np.any(mask1_dilated & mask2) or np.any(mask2_dilated & mask1)


def crop_masked_region(mask, image):
    result = image.copy()
    result[mask == 0] = [255, 255, 255]
    coords = cv2.findNonZero(mask.astype(np.uint8)*255)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_result = result[y:y+h, x:x+w]
    cropped_mask_result = mask[y:y+h, x:x+w]
    return cropped_result, cropped_mask_result


def resize_mask(mask):
    mask = np.array(mask.resize((16, 16), Image.NEAREST)).flatten()
    mask = np.append(mask, True)
    return mask


def calc_dinov2_score(image1, mask1, image2, mask2, processor, model):
    with torch.no_grad():
        image1 = image1.resize((224, 224))
        mask1 = mask1.resize((224, 224), Image.NEAREST)
        inputs1 = processor(images=image1, return_tensors="pt").to("cuda")
        outputs1 = model(**inputs1)
        last_hidden_states1 = outputs1.last_hidden_state
        img_features1 = last_hidden_states1 / last_hidden_states1.norm(p=2, dim=-1, keepdim=True)
        img_features1 = img_features1.squeeze(0).cpu().numpy()
        mask1_resized = resize_mask(mask1)
        valid_patches1 = img_features1[mask1_resized.flatten()]
        final_feature1 = np.mean(valid_patches1, axis=0)
        final_feature1 = final_feature1 / np.linalg.norm(final_feature1)

        image2 = image2.resize((224, 224))
        mask2 = mask2.resize((224, 224))
        inputs2 = processor(images=image2, return_tensors="pt").to("cuda")
        outputs2 = model(**inputs2)
        last_hidden_states2 = outputs2.last_hidden_state
        img_features2 = last_hidden_states2 / last_hidden_states2.norm(p=2, dim=-1, keepdim=True)
        img_features2 = img_features2.squeeze(0).cpu().numpy()
        mask2_resized = resize_mask(mask2)
        valid_patches2 = img_features2[mask2_resized.flatten()]
        final_feature2 = np.mean(valid_patches2, axis=0)
        final_feature2 = final_feature2 / np.linalg.norm(final_feature2)

        cos_sim = np.dot(final_feature1, final_feature2)
        return 1 - cos_sim


def calculate(image_path, ref_mask, mask_generator, processor, model):
    image = Image.open(image_path).convert('RGB')
    ref_mask = ref_mask.convert("1") if isinstance(ref_mask, Image.Image) else Image.fromarray(ref_mask).convert("1")
    masks = mask_generator.generate(np.array(image))
    masks = filter_masks(masks)

    bound_mask_index_li, center_mask_index_li = classify_masks(masks, ref_mask)
    bound_mask_li = [masks[b] for b in bound_mask_index_li]
    center_mask_li = [masks[c] for c in center_mask_index_li]
    bound_mask_cropped_li = [b & np.array(ref_mask) for b in bound_mask_li]

    center_mask_pixel_li = [np.sum(np.array(cm)) for cm in center_mask_li]
    final_score = 0
    remove_score = 0

    left, upper, right, lower = ref_mask.getbbox()
    width, height = right - left, lower - upper
    left, right = max(0, left - width // 3), min(image.width, right + width // 3)
    upper, lower = max(0, upper - height // 3), min(image.height, lower + height // 3)
    bbox_mask = Image.new("1", (image.width, image.height), 0)
    ImageDraw.Draw(bbox_mask).rectangle([left, upper, right, lower], fill=1)
    bbox_mask = Image.fromarray(np.array(bbox_mask) ^ np.array(ref_mask))
    score = calc_dinov2_score(image, ref_mask, image, bbox_mask, processor, model)
    remove_score += score
    final_score += score

    for j, cm in enumerate(center_mask_li):
        cm = np.array(cm)
        merged_bm = np.zeros_like(cm).astype(bool)
        total = 0
        for bm in bound_mask_cropped_li:
            bm = np.array(bm)
            if is_adjacent(bm, cm):
                merged_bm |= bm
                total += 1
        if total != 0:
            bm_img_cropped, bm_mask_cropped = crop_masked_region(merged_bm, np.array(image))
            cm_img_cropped, cm_mask_cropped = crop_masked_region(cm, np.array(image))
            bm_image, bm_mask = Image.fromarray(bm_img_cropped), Image.fromarray(bm_mask_cropped).convert("L")
            cm_image, cm_mask = Image.fromarray(cm_img_cropped), Image.fromarray(cm_mask_cropped).convert("L")
            dinov2_score = calc_dinov2_score(bm_image, bm_mask, cm_image, cm_mask, processor, model)
            final_score += (center_mask_pixel_li[j] * dinov2_score / sum(center_mask_pixel_li))

    return final_score


def calculate_objects(image_path, mask_path, mask_generator, processor, model):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary_mask)
    masks = []
    for i in range(1, num_labels):
        single_mask = np.zeros_like(mask)
        single_mask[labels == i] = 255
        masks.append(single_mask)
    total_score = 0
    for m in masks:
        total_score += calculate(image_path, Image.fromarray(m).convert("1"), mask_generator, processor, model)
    return total_score, len(masks)


def main():
    parser = argparse.ArgumentParser(description="Compute CFD Score")
    parser.add_argument("--input_path", type=str, required=True, help="Path to mat_results folder")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask folder")
    args = parser.parse_args()

    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to("cuda")

    avg = 0
    input_files = os.listdir(args.input_path)
    for name in tqdm(input_files):
        p1 = os.path.join(args.input_path, name)
        p2 = os.path.join(args.mask_path, name)
        score, nums = calculate_objects(p1, p2, mask_generator, processor, model)
        avg += score / nums
    avg /= len(input_files)
    print("CFD Score:", avg)


if __name__ == "__main__":
    main()



