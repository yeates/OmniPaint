import gradio as gr
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
from gradio_image_annotation import image_annotator
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter, ImageDraw, ImageOps
from gradio.themes.utils import colors
from carvekit.api.high import HiInterface
import threading

from src.condition import Condition
from src.generate import generate, seed_everything
from src.embedding_loader import load_npz_embeddings


gradient_colors = {
    'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'secondary': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
    'accent': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'success': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    'warning': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
}

# Load and encode logo for header display
with open("./demo_assets/logo.jpg", "rb") as img_file:
    base64_icon = base64.b64encode(img_file.read()).decode()

# Maximum image dimension for processing
MAX_LENGTH = 1024

removal_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:0")
removal_pipe.load_lora_weights("weights/omnipaint_remove.safetensors", adapter_name="removal")
removal_pipe.set_adapters(["removal"])

insertion_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:1")
insertion_pipe.load_lora_weights("weights/omnipaint_insert.safetensors", adapter_name="insertion")
insertion_pipe.set_adapters(["insertion"])

carvekit_lock = threading.Lock()
carvekit_interface = HiInterface(
    object_type="object", batch_size_seg=5, batch_size_matting=1,
    device='cuda:2' if torch.cuda.is_available() else 'cpu', seg_mask_size=640,
    matting_mask_size=2048, trimap_prob_threshold=231, trimap_dilation=30,
    trimap_erosion_iters=5, fp16=False
)


REMOVE_EMBED_PATH = "./demo_assets/embeddings/remove.npz"
INSERT_EMBED_PATH = "./demo_assets/embeddings/insert.npz"

def _get_device(pipe):
    return getattr(
        pipe,
        "_execution_device",
        getattr(pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )

removal_device = _get_device(removal_pipe)
insertion_device = _get_device(insertion_pipe)
removal_dtype = removal_pipe.transformer.dtype
insertion_dtype = insertion_pipe.transformer.dtype

REMOVE_PROMPT_EMBEDS, REMOVE_POOLED_EMBEDS, REMOVE_TEXT_IDS = load_npz_embeddings(
    REMOVE_EMBED_PATH, device=removal_device, dtype=removal_dtype
)
INSERT_PROMPT_EMBEDS, INSERT_POOLED_EMBEDS, INSERT_TEXT_IDS = load_npz_embeddings(
    INSERT_EMBED_PATH, device=insertion_device, dtype=insertion_dtype
)


def preprocess_image(image):
    """Initialize image preprocessing with empty tracking states"""
    return image, gr.State([]), gr.State([]), image

def get_point(point_type, tracking_points, trackings_input_label, first_frame_path, evt: gr.SelectData):
    tracking_points.value.append(evt.index)
    trackings_input_label.value.append(1 if point_type == "include" else 0)
    
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    radius = int(0.02 * min(w, h))
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    for index, track in enumerate(tracking_points.value):
        color = (0, 255, 0, 255) if trackings_input_label.value[index] == 1 else (255, 0, 0, 255)
        cv2.circle(transparent_layer, track, radius, color, -1)

    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)
    
    return tracking_points, trackings_input_label, selected_point_map

# Initialize CPU autocast for optimal performance
torch.autocast(device_type="cpu", dtype=torch.float32).__enter__()

def show_mask(mask, ax, random_color=False, borders=True):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """Generate visualization of segmentation masks"""
    combined_images = []  
    mask_images = []      

    for i, (mask, _) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        plt.axis('off')

        combined_filename = f"combined_image_{i+1}.jpg"
        plt.savefig(combined_filename, format='jpg', bbox_inches='tight')
        combined_images.append(combined_filename)
        plt.close()

        mask_image = np.zeros_like(image, dtype=np.uint8)
        mask_layer = (mask > 0).astype(np.uint8) * 255
        for c in range(3):
            mask_image[:, :, c] = mask_layer

        mask_filename = f"mask_image_{i+1}.png"
        Image.fromarray(mask_image).save(mask_filename)
        mask_images.append(mask_filename)

    return combined_images, mask_images

def sam_process(input_image, checkpoint, tracking_points, trackings_input_label):
    image = np.array(Image.open(input_image).convert("RGB"))
    
    checkpoint_map = {
        "tiny": ("./checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        "small": ("./checkpoints/sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
        "base-plus": ("./checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
        "large": ("./checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml")
    }
    
    sam2_checkpoint, model_cfg = checkpoint_map[checkpoint]
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda:3")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(tracking_points.value),
        point_labels=np.array(trackings_input_label.value),
        multimask_output=False,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    _, mask_results = show_masks(image, masks, scores, 
                                  point_coords=np.array(tracking_points.value), 
                                  input_labels=np.array(trackings_input_label.value), 
                                  borders=True)

    return mask_results[0]

def remove_background(input_image_path):
    with carvekit_lock:
        images_without_background = carvekit_interface([input_image_path])
        cat_wo_bg = np.array(images_without_background[0])
        rgb = cat_wo_bg[..., :3]
        alpha = cat_wo_bg[..., 3:] / 255.0
        white_bg = np.ones_like(rgb) * 255
        blended = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(blended)

def resize_with_aspect_ratio(img, max_length=MAX_LENGTH, resize_mode="longest"):
    """Resize image while maintaining aspect ratio
    Args:
        img: PIL Image
        max_length: Maximum dimension size
        resize_mode: 'longest' or 'shortest' edge resize
    """
    w, h = img.size
    if resize_mode == "longest":
        if max(w, h) > max_length:
            scale = max_length / max(w, h)
            return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:  # shortest edge
        if min(w, h) > max_length:
            scale = max_length / min(w, h)
            return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def any_to_PIL(image):
    """Convert various image formats to PIL Image"""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        return Image.fromarray(image.cpu().numpy())
    elif isinstance(image, Image.Image):
        return image
    elif isinstance(image, str) and os.path.isfile(image):
        return Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def omnipaint_removal(input_image, input_mask, seed, num_inference_steps, backbone, cached_state, resize_mode="longest"):
    """Perform object removal using FLUX model"""
    # Check if required inputs are provided
    if input_image is None:
        raise gr.Error("Please upload an image first")
    if input_mask is None:
        raise gr.Error("Please create or upload a mask first")
    
    condition_img = resize_with_aspect_ratio(Image.open(input_image).convert("RGB"), resize_mode=resize_mode)
    
    mask = Image.fromarray(input_mask).convert('L').resize(condition_img.size)
    inverted_mask = ImageOps.invert(mask)
    
    black_background = Image.new("RGB", condition_img.size, (0, 0, 0))
    composite_img = Image.composite(condition_img, black_background, inverted_mask)
    
    condition = Condition("removal", composite_img)
    seed_everything(seed)
    
    result_img = generate(
        removal_pipe,
        conditions=[condition],
        width=(condition_img.size[0] // 8) * 8,
        height=(condition_img.size[1] // 8) * 8,
        num_inference_steps=num_inference_steps,
        prompt=None,
        prompt_embeds=REMOVE_PROMPT_EMBEDS,
        pooled_prompt_embeds=REMOVE_POOLED_EMBEDS,
        text_ids=REMOVE_TEXT_IDS,
    ).images[0].resize(condition_img.size)
    
    result = {
        "input_img": input_image,
        "mask_img": input_mask,
        "result_img": result_img
    }
    
    
    return result_img, condition_img, result

def omnipaint_insert(input_image, input_mask, input_ref_img, seed, num_inference_steps, backbone, cached_state, resize_mode="longest"):
    """Perform object insertion using FLUX model"""
    # Check if required inputs are provided
    if input_image is None:
        raise gr.Error("Please upload a background image first")
    if input_mask is None:
        raise gr.Error("Please create or upload a mask first")
    if input_ref_img is None:
        raise gr.Error("Please upload a subject image to insert")
    
    condition_img = resize_with_aspect_ratio(Image.open(input_image).convert("RGB"), resize_mode=resize_mode)
    subject_img = remove_background(input_ref_img).resize((512, 512))
    
    mask = Image.open(input_mask).convert("L").resize(condition_img.size)
    inverted_mask = ImageOps.invert(mask)
    
    black_background = Image.new("RGB", condition_img.size, (0, 0, 0))
    composite_img = Image.composite(condition_img, black_background, inverted_mask)
    
    condition = [Condition("insertion", composite_img), Condition("insertion", subject_img, position_delta=(0, -32))]
    seed_everything(seed)
    
    result_img = generate(
        insertion_pipe,
        conditions=condition,
        width=(composite_img.size[0] // 8) * 8,
        height=(composite_img.size[1] // 8) * 8,
        num_inference_steps=num_inference_steps,
        prompt=None,
        prompt_embeds=INSERT_PROMPT_EMBEDS,
        pooled_prompt_embeds=INSERT_POOLED_EMBEDS,
        text_ids=INSERT_TEXT_IDS,
    ).images[0].resize(condition_img.size)
    
    result = {
        "input_img": input_image,
        "mask_img": input_mask,
        "ref_img": input_ref_img,
        "result_img": result_img
    }
    
    
    return result_img, condition_img, result

def get_mask(annotations):
    image_shape = annotations["image"].shape
    bbox = [
        int(annotations["boxes"][0]["xmin"]),
        int(annotations["boxes"][0]["ymin"]),
        int(annotations["boxes"][0]["xmax"]),
        int(annotations["boxes"][0]["ymax"]),
    ]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    xmin, ymin, xmax, ymax = bbox
    mask[ymin:ymax, xmin:xmax] = 255
    return annotations["image"], mask, gr.update(visible=True)

def switch_page(page):
    return (
        gr.update(visible=page == "🧹 Object Removal"), 
        gr.update(visible=page == "🖌️ Object Insertion")
    )

def process_example_insertion(img, mask, ref, seed, steps, backbone, mask_method):
    """Process insertion examples by passing through all parameters"""
    return img, mask, ref, seed, steps, backbone, img, mask_method




# Compact integrated header with title and navigation
header_html = """
<div style="
    display: flex; 
    align-items: center;
    justify-content: space-between;
    padding: 24px 32px; 
    margin-bottom: 24px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
">
    <!-- Animated background particles -->
    <div style="
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.2) 0%, transparent 50%);
        animation: particleFloat 8s ease-in-out infinite alternate;
    "></div>
    
    <!-- Left section: Logo and title -->
    <div style="display: flex; align-items: center; gap: 16px; z-index: 2; position: relative;">
        <div style="
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            box-shadow: 0 6px 24px rgba(102, 126, 234, 0.4);
            animation: logoFloat 3s ease-in-out infinite alternate;
        ">
            <img src="data:image/png;base64,""" + base64_icon + """ " style="
                height: 48px; 
                filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
            " />
        </div>
        
        <div>
            <h1 style="
                font-size: 42px;
                font-weight: 800;
                font-family: 'Inter', 'Helvetica Neue', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                background-size: 200% 200%;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: gradientText 4s ease-in-out infinite;
                margin: 0;
                letter-spacing: -1px;
                line-height: 1;
            ">OmniPaint</h1>
            <p style="
                font-size: 14px;
                color: rgba(71, 85, 105, 0.7);
                font-weight: 400;
                margin: 4px 0 0 0;
                font-family: 'Inter', sans-serif;
            ">Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting</p>
        </div>
    </div>
    
    <!-- Right section: Navigation links -->
    <div style="
        display: flex;
        gap: 16px;
        z-index: 2;
        position: relative;
        flex-wrap: wrap;
    ">
        <a href="https://github.com/yeates/OmniPaint" style="
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
            color: #1f2937;
            text-decoration: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.3);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(102, 126, 234, 0.3)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(102, 126, 234, 0.2)'">
            🐙 GitHub
        </a>
        
        <a href="https://www.yongshengyu.com/OmniPaint-Page/" style="
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.8) 0%, rgba(245, 87, 108, 0.8) 100%);
            color: #1f2937;
            text-decoration: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.3s ease;
            border: 1px solid rgba(240, 147, 251, 0.3);
            box-shadow: 0 2px 8px rgba(240, 147, 251, 0.2);
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(240, 147, 251, 0.3)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(240, 147, 251, 0.2)'">
            🌐 Project
        </a>
        
        <a href="https://arxiv.org/pdf/2503.08677" style="
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.8) 0%, rgba(0, 242, 254, 0.8) 100%);
            color: #1f2937;
            text-decoration: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.3s ease;
            border: 1px solid rgba(79, 172, 254, 0.3);
            box-shadow: 0 2px 8px rgba(79, 172, 254, 0.2);
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(79, 172, 254, 0.3)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(79, 172, 254, 0.2)'">
            📄 Paper
        </a>
    </div>
</div>

<style>
@keyframes particleFloat {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-12px) rotate(6deg); }
}

@keyframes logoFloat {
    0% { transform: translateY(0px) scale(1); }
    100% { transform: translateY(-4px) scale(1.05); }
}

@keyframes gradientText {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
</style>
"""

# Curated example datasets for object removal
sample_remove_list = [
    [
        "./demo_assets/removal_samples/images/5.jpg",
        "./demo_assets/removal_samples/masks/5.png",
        0,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "./demo_assets/removal_samples/images/5.jpg",
    ],
    [
        "./demo_assets/removal_samples/images/1.jpg",
        "./demo_assets/removal_samples/masks/1.png",
        42,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "./demo_assets/removal_samples/images/1.jpg",
    ],
    [
        "./demo_assets/removal_samples/images/6.jpg",
        "./demo_assets/removal_samples/masks/6.png",
        123,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "./demo_assets/removal_samples/images/6.jpg",
    ],
    [
        "./demo_assets/removal_samples/images/9.jpg",
        "./demo_assets/removal_samples/masks/9.png",
        42424242,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "./demo_assets/removal_samples/images/9.jpg",
    ],
    [
        "./demo_assets/removal_samples/images/13.jpg",
        "./demo_assets/removal_samples/masks/13.png",
        250308677,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "./demo_assets/removal_samples/images/13.jpg",
    ],
]

# Curated example datasets for object insertion
sample_insertion_list = [
    [
        "./demo_assets/insertion_samples/backgrounds/background-2.png",
        "./demo_assets/insertion_samples/masks/mask-2.png",
        "./demo_assets/insertion_samples/subjects/subject-2.png",
        297,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "Upload Mask",
    ],
    [
        "./demo_assets/insertion_samples/backgrounds/background-1.png",
        "./demo_assets/insertion_samples/masks/mask-1.png",
        "./demo_assets/insertion_samples/subjects/subject-1.png",
        700,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "Upload Mask",
    ],
    [
        "./demo_assets/insertion_samples/backgrounds/background-22.png",
        "./demo_assets/insertion_samples/masks/mask-22.png",
        "./demo_assets/insertion_samples/subjects/subject-22.png",
        42424242,
        28,
        "FLUX.1-dev (Recommend to set inference step to 28)",
        "Upload Mask",
    ],
]

# Sophisticated CSS with modern aesthetics and animations
css = """
/* Import premium fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global theme with dark luxury aesthetic */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow-luxury: 0 8px 32px rgba(0, 0, 0, 0.3);
    --shadow-glow: 0 0 20px rgba(102, 126, 234, 0.3);
}

/* Main container with light animated background */
.gradio-container {
    background: linear-gradient(-45deg, #f8f9ff, #f0f2ff, #e8edff, #f5f7ff);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass morphism cards with light theming */
.gr-block, .gr-form, .gr-panel {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: rgba(51, 65, 85, 0.9) !important;
}

/* Gradio components with light theme */
.gr-input, .gr-box, .gr-padded, .gr-panel, 
.gr-slider, .gr-radio, .gr-checkbox, .gr-upload,
.gr-file, .gr-image, .gr-textbox {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 12px !important;
    color: rgba(51, 65, 85, 0.9) !important;
}

/* Slider specific styling */
.gr-slider .gr-slider-track {
    background: rgba(255, 255, 255, 0.2) !important;
}

.gr-slider .gr-slider-thumb {
    background: var(--primary-gradient) !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
}

/* Radio button styling */
.gr-radio .gr-radio-option {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    color: rgba(51, 65, 85, 0.9) !important;
    margin: 4px !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
}

.gr-radio .gr-radio-option:checked {
    background: var(--primary-gradient) !important;
    color: white !important;
    border-color: rgba(102, 126, 234, 0.6) !important;
}

/* Upload area styling */
.gr-upload, .gr-file {
    background: rgba(255, 255, 255, 0.7) !important;
    border: 2px dashed rgba(102, 126, 234, 0.4) !important;
    color: rgba(51, 65, 85, 0.8) !important;
}

.gr-upload:hover, .gr-file:hover {
    border-color: rgba(102, 126, 234, 0.7) !important;
    background: rgba(255, 255, 255, 0.9) !important;
}

/* Text and label styling for light theme */
.gr-block label, .gr-form label, .gr-panel label,
.gr-textbox label, .gr-slider label, .gr-dropdown label,
.gr-radio label, .gr-checkbox label, .gr-upload label {
    color: rgba(51, 65, 85, 0.9) !important;
    font-weight: 500 !important;
    text-shadow: none !important;
}

/* Input field content */
.gr-textbox input, .gr-dropdown select, .gr-slider input,
.gr-textbox textarea, .gr-number input {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    color: rgba(51, 65, 85, 0.9) !important;
    border-radius: 8px !important;
}

/* Dropdown options */
.gr-dropdown select option {
    background: rgba(255, 255, 255, 0.95) !important;
    color: rgba(51, 65, 85, 0.9) !important;
}

/* General text content */
.gr-markdown, .gr-html {
    color: rgba(51, 65, 85, 0.9) !important;
}

/* Override text colors for light theme */
div[class*="gr-"], span[class*="gr-"], p[class*="gr-"] {
    color: rgba(51, 65, 85, 0.9) !important;
}

/* Component text colors */
.gr-radio, .gr-checkbox, .gr-slider, .gr-upload {
    color: rgba(51, 65, 85, 0.9) !important;
}

/* Value display text */
.gr-slider .gr-slider-value, .gr-number .gr-number-value {
    color: rgba(51, 65, 85, 0.9) !important;
    background: rgba(255, 255, 255, 0.8) !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
}

/* Premium button styling */
.gr-button {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    box-shadow: var(--shadow-luxury) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:active {
    transform: translateY(0) !important;
}

/* Ripple effect for buttons */
.gr-button::before {
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.3) !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.6s, height 0.6s !important;
}

.gr-button:active::before {
    width: 300px !important;
    height: 300px !important;
}

/* Special styling for primary actions */
.gr-button[variant="primary"] {
    background: var(--accent-gradient) !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    box-shadow: var(--shadow-glow) !important;
}

/* Image containers with elegant borders */
.gr-image {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-luxury) !important;
    border: 2px solid var(--glass-border) !important;
    transition: all 0.3s ease !important;
}

.gr-image:hover {
    transform: scale(1.02) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* Input fields with modern styling */
.gr-textbox, .gr-slider {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* Radio buttons centered and styled */
.center-radio .gr-radio {
    justify-content: center !important;
}

.gr-radio .gr-radio-option {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    margin: 4px !important;
    transition: all 0.3s ease !important;
}

.gr-radio .gr-radio-option:hover {
    background: var(--primary-gradient) !important;
    transform: translateY(-2px) !important;
}

/* Dropdown menus */
.gr-dropdown {
    background: var(--glass-bg) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* Loading animations */
.gr-loading {
    background: conic-gradient(from 0deg, #667eea, #764ba2, #667eea) !important;
    border-radius: 50% !important;
    animation: spin 1s linear infinite !important;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 4px;
}

/* Enhanced examples section */
.gr-examples {
    background: var(--glass-bg) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

/* Markdown content styling */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    background: var(--primary-gradient) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    color: transparent !important;
    font-weight: 700 !important;
}

/* Links with hover effects */
a {
    color: #4facfe !important;
    text-decoration: none !important;
    transition: all 0.3s ease !important;
    position: relative !important;
}

a:hover {
    color: #00f2fe !important;
    text-shadow: 0 0 10px rgba(79, 172, 254, 0.5) !important;
}

a::after {
    content: '' !important;
    position: absolute !important;
    bottom: -2px !important;
    left: 0 !important;
    width: 0 !important;
    height: 2px !important;
    background: var(--accent-gradient) !important;
    transition: width 0.3s ease !important;
}

a:hover::after {
    width: 100% !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    
    .gr-button {
        padding: 10px 20px !important;
        font-size: 14px !important;
    }
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(header_html)
    
    # Stylish mode selector with custom styling
    radio = gr.Radio(
        ["🧹 Object Removal", "🖌️ Object Insertion"], 
        label="✨ Step 1: Choose Your Creative Mode", 
        value="🧹 Object Removal",
        elem_classes=["center-radio"],
        info="Select whether you want to remove objects or insert new ones"
    )
    
    removal_cache = gr.State({})
    insertion_cache = gr.State({})
    
    with gr.Column(visible=True) as page_a:
        first_frame_path = gr.State()
        tracking_points = gr.State([])
        trackings_input_label = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="
                    padding: 12px 16px;
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    border-radius: 12px;
                    margin-bottom: 12px;
                    border: 1px solid rgba(102, 126, 234, 0.2);
                ">
                    <h3 style="
                        font-size: 16px;
                        font-weight: 600;
                        color: rgba(51, 65, 85, 0.9);
                        margin: 0;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    ">🎯 Step 2: Create Mask via SAM2 or Upload</h3>
                    <p style="
                        font-size: 13px;
                        color: rgba(71, 85, 105, 0.7);
                        margin: 4px 0 0 0;
                    ">Upload an image and click points to select areas for removal</p>
                    <p style="
                        font-size: 12px;
                        color: rgba(102, 126, 234, 0.8);
                        margin: 4px 0 0 0;
                        font-style: italic;
                    ">💡 Tip: Mask accuracy significantly affects the final result quality</p>
                </div>
                """)
                with gr.Row():
                    input_image = gr.Image(
                        label="🖼️ Source Image", 
                        interactive=False, 
                        type="filepath", 
                        visible=False
                    )
                    points_map = gr.Image(
                        label="🎯 Interactive Point Selection", 
                        type="filepath", 
                        interactive=True, 
                        height=420,
                        show_label=True
                    )
                
                with gr.Row():
                    point_type = gr.Radio(
                        label="🎯 Point Selection Mode", 
                        choices=[
                            ("✅ Include (Green)", "include"),
                            ("❌ Exclude (Red)", "exclude")
                        ], 
                        value="include"
                    )
                    clear_points_btn = gr.Button(
                        "🗑️ Clear Points", 
                        variant="secondary"
                    )
                
                checkpoint = gr.Dropdown(
                    label="🤖 SAM2 Model Checkpoint", 
                    choices=[
                        ("🐜 Tiny - Lightning Fast", "tiny"),
                        ("🚀 Small - Balanced", "small"),
                        ("⭐ Base Plus - Enhanced", "base-plus"),
                        ("🔥 Large - Maximum Precision", "large")
                    ], 
                    value="large"
                )
                submit_btn = gr.Button(
                    "🎭 Generate Mask", 
                    variant="secondary",
                    size="lg"
                )
                
                with gr.Row():
                    output_result_mask = gr.Image(
                        label="🎭 Generated Mask", 
                        height=420,
                        show_label=True
                    )
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="
                    padding: 12px 16px;
                    background: linear-gradient(135deg, rgba(67, 233, 123, 0.15) 0%, rgba(56, 249, 215, 0.15) 100%);
                    border-radius: 12px;
                    margin-bottom: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                ">
                    <h3 style="
                        font-size: 16px;
                        font-weight: 600;
                        color: rgba(51, 65, 85, 0.9);
                        margin: 0;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    ">✨ Step 3: Generate Amazing Result</h3>
                    <p style="
                        font-size: 13px;
                        color: rgba(71, 85, 105, 0.7);
                        margin: 4px 0 0 0;
                    ">Adjust parameters and click the remove button to process</p>
                </div>
                """)
                with gr.Row():
                    seed1 = gr.Slider(
                        label="🎲 Random Seed", 
                        minimum=0, 
                        maximum=np.iinfo(np.int32).max, 
                        step=1, 
                        value=42,
                        info="Set seed for reproducible results"
                    )
                    num_inference_steps1 = gr.Slider(
                        label="⚙️ Inference Steps", 
                        minimum=1, 
                        maximum=28, 
                        step=1, 
                        value=28,
                        info="Higher steps = better quality"
                    )
                resize_mode1 = gr.Radio(
                    label="📐 Image Size Preference",
                    choices=[
                        ("Keep detail (may be larger)", "shortest"),
                        ("Save memory (may be smaller)", "longest")
                    ],
                    value="longest",
                    info="Choose how to handle image size"
                )
                
                backbone1 = gr.Dropdown(
                    label="🧠 AI Backbone Model",
                    choices=[("🌌 FLUX.1-dev (Recommended: 28 steps)", "FLUX.1-dev (Recommend to set inference step to 28)")],
                    value="FLUX.1-dev (Recommend to set inference step to 28)"
                )
                
                submit_btn_removal = gr.Button(
                    "🧹 Remove Object", 
                    variant="primary", 
                    size="lg",
                    elem_id="remove-btn"
                )
                
                
                with gr.Row():
                    original_image = gr.Image(
                        label="🖼️ Original Reference", 
                        height=420, 
                        visible=False,
                        show_label=True
                    )
        
        with gr.Row():
            output_comparison = gr.Image(
                label="✨ Final Result - Object Removed", 
                height=640,
                show_label=True
            )
        
        gr.Examples(
            sample_remove_list,
            fn=lambda *args: (None, args[0]),
            inputs=[input_image, output_result_mask, seed1, num_inference_steps1, backbone1, points_map],
            outputs=[output_comparison, original_image]
        )

    with gr.Column(visible=False) as page_b:
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="
                    padding: 12px 16px;
                    background: linear-gradient(135deg, rgba(240, 147, 251, 0.15) 0%, rgba(245, 87, 108, 0.15) 100%);
                    border-radius: 12px;
                    margin-bottom: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                ">
                    <h3 style="
                        font-size: 16px;
                        font-weight: 600;
                        color: rgba(51, 65, 85, 0.9);
                        margin: 0;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    ">🎭 Step 2: Background and Mask</h3>
                    <p style="
                        font-size: 13px;
                        color: rgba(71, 85, 105, 0.7);
                        margin: 4px 0 0 0;
                    ">Upload background image and define insertion area</p>
                </div>
                """)
                
                mask_method = gr.Radio(
                    label="🎨 Mask Creation Method", 
                    choices=[
                        ("✏️ Draw Mask Interactively", "Draw Mask"),
                        ("📁 Upload Custom Mask", "Upload Mask")
                    ], 
                    value="Draw Mask"
                )
                input_image_insertion = gr.Image(
                    label="🖼️ Background Image", 
                    type="filepath", 
                    visible=False, 
                    height=420
                )
                
                annotator = image_annotator(
                    disable_edit_boxes=True, 
                    label="🎨 Upload Background & Draw Selection Box", 
                    height=420
                )
                submit_bn_getmask_insert = gr.Button(
                    "🎭 Generate Mask", 
                    variant="secondary",
                    size="lg"
                )
                
                bbox_mask = gr.Image(
                    label="🎭 Generated Mask", 
                    type="filepath", 
                    height=420, 
                    visible=False,
                    show_label=True
                )
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="
                    padding: 12px 16px;
                    background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
                    border-radius: 12px;
                    margin-bottom: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                ">
                    <h3 style="
                        font-size: 16px;
                        font-weight: 600;
                        color: rgba(51, 65, 85, 0.9);
                        margin: 0;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    ">🎨 Step 3: Object to Insert</h3>
                    <p style="
                        font-size: 13px;
                        color: rgba(71, 85, 105, 0.7);
                        margin: 4px 0 0 0;
                    ">Upload the object you want to insert and adjust settings</p>
                </div>
                """)
                ref_img = gr.Image(
                    label="🎨 Object to Insert", 
                    type="filepath", 
                    height=420,
                    show_label=True
                )
                
                with gr.Row():
                    seed2 = gr.Slider(
                        label="🎲 Random Seed", 
                        minimum=0, 
                        maximum=np.iinfo(np.int32).max, 
                        step=1, 
                        value=42,
                        info="Set seed for reproducible results"
                    )
                    num_inference_steps2 = gr.Slider(
                        label="⚙️ Inference Steps", 
                        minimum=1, 
                        maximum=28, 
                        step=1, 
                        value=28,
                        info="Higher steps = better quality"
                    )
                resize_mode2 = gr.Radio(
                    label="📐 Image Size Preference",
                    choices=[
                        ("Keep detail (may be larger)", "shortest"),
                        ("Save memory (may be smaller)", "longest")
                    ],
                    value="longest",
                    info="Choose how to handle image size"
                )
                
                backbone2 = gr.Dropdown(
                    label="🧠 AI Backbone Model",
                    choices=[("🌌 FLUX.1-dev (Recommended: 28 steps)", "FLUX.1-dev (Recommend to set inference step to 28)")],
                    value="FLUX.1-dev (Recommend to set inference step to 28)"
                )
                
                original_image_insert = gr.Image(
                    label="🖼️ Original Reference", 
                    height=420, 
                    visible=False,
                    show_label=True
                )
                
                submit_btn_insert = gr.Button(
                    "🖌️ Insert Object", 
                    variant="primary", 
                    size="lg",
                    elem_id="insert-btn"
                )
        
        
        with gr.Row():
            insertion_comparison = gr.Image(
                label="✨ Final Result - Object Inserted", 
                height=640,
                show_label=True
            )
        
        gr.Examples(
            sample_insertion_list,
            fn=process_example_insertion,
            inputs=[input_image_insertion, bbox_mask, ref_img, seed2, num_inference_steps2, backbone2, mask_method],
            outputs=[input_image_insertion, bbox_mask, ref_img, seed2, num_inference_steps2, backbone2, annotator, mask_method]
        )
    
    radio.change(
        fn=switch_page, 
        inputs=radio, 
        outputs=[page_a, page_b]
    )

    def toggle_mask_method(choice):
        return gr.update(visible=choice == "Draw Mask"), gr.update(visible=choice == "Upload Mask"), gr.update(visible=choice == "Upload Mask")
    
    mask_method.change(
        fn=toggle_mask_method,
        inputs=[mask_method],
        outputs=[annotator, input_image_insertion, bbox_mask]
    )

    clear_points_btn.click(
        fn=preprocess_image,
        inputs=input_image,
        outputs=[first_frame_path, tracking_points, trackings_input_label, points_map],
        queue=False
    )
    
    points_map.upload(
        fn=preprocess_image,
        inputs=[points_map],
        outputs=[first_frame_path, tracking_points, trackings_input_label, input_image],
        queue=False
    )

    points_map.select(
        fn=get_point,
        inputs=[point_type, tracking_points, trackings_input_label, first_frame_path],
        outputs=[tracking_points, trackings_input_label, points_map],
        queue=False
    )

    submit_btn.click(
        fn=sam_process,
        inputs=[input_image, checkpoint, tracking_points, trackings_input_label],
        outputs=[output_result_mask]
    )

    submit_btn_removal.click(
        fn=omnipaint_removal,
        inputs=[input_image, output_result_mask, seed1, num_inference_steps1, backbone1, removal_cache, resize_mode1],
        outputs=[output_comparison, original_image, removal_cache]
    )
    

    submit_bn_getmask_insert.click(
        fn=get_mask,
        inputs=[annotator],
        outputs=[input_image_insertion, bbox_mask, bbox_mask]
    )

    submit_btn_insert.click(
        fn=omnipaint_insert,
        inputs=[input_image_insertion, bbox_mask, ref_img, seed2, num_inference_steps2, backbone2, insertion_cache, resize_mode2],
        outputs=[insertion_comparison, original_image_insert, insertion_cache]
    )
    

demo.launch(show_api=False, show_error=True, share=True)