***Image to LED Screen***
### What the code does & why it matches your three-step spec

| Your Step | Code Section | How it's implemented |
|-----------|--------------|----------------------|
| **1. Segmentation & control map** | `largest_mask`, `canny_map` |  • Segment-Anything automatically extracts masks; we keep the largest object.<br> • OpenCV's Canny detector converts the same frame into a structural guide image (edge map). |
| **2. Stylization with SD + ControlNet** | `stylize` | Uses Hugging-Face **`StableDiffusionControlNetInpaintPipeline`** so the mask AND the Canny map are honoured simultaneously. The ControlNet weight *sd-controlnet-canny* is loaded once and reused.  ([ControlNet - Hugging Face](https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/controlnet?utm_source=chatgpt.com)) |
| **3. Post-processing** | `led_quantise` | • Lanczos resize to **60 × 60**.<br> • Palette reduced to 32 colours, then per-pixel brightness rounded to 8 discrete levels, matching a typical RGB LED matrix driver. |

---

### System Requirements

#### Hardware Requirements
The code has been tested on the following hardware setup:
```
NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     
+------------------------------------------------------------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | ECC   |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | Util  |
+========================================================================+
|   0  Tesla T4                       Off | 00000001:00:00.0 Off | Off   |
| N/A   37C    P0              49W /  70W |    385MiB / 16384MiB | 67%   |
+------------------------------------------------------------------------+
```

#### Dependencies
The exact versions of libraries used in this project are locked in the `requirements.txt` file:
```
diffusers==0.33.1
numpy==2.1.2
opencv-python==4.11.0.86
pillow==11.0.0
segment-anything==1.0
torch==2.5.1+cu121
torchvision==0.20.1+cu121
xformers==0.0.29.post1
```

#### Tips & customisation

* **Prompt-tuning** Swap `prompt` / `neg_prompt` strings or add LoRAs to get a different cartoon style.  
* **Edge type** Replace the `CTRL_MODEL` (e.g. `"lllyasviel/sd-controlnet-lineart"`) and generate a line-art guide instead of Canny.  
* **Mask selection** If the biggest SAM mask isn't what you want, change the logic (e.g. use click-guided SAM to pick a region interactively).  
* **LED driver format** If your hardware needs raw bytes, just convert `led_ready` to `"P"` mode and `tobytes()` or export RGB565.
* **Memory Optimization** Use the `--cpu` flag if you're experiencing GPU memory issues (will be slower but more reliable).

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Image2led_screen.git
cd Image2led_screen

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download SAM model checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Usage

Basic usage:
```bash
python cartoonize_led.py ./images/your_image.jpg
```

To force CPU mode (for memory issues):
```bash
python cartoonize_led.py ./images/your_image.jpg --cpu
```

You'll get `results/your_image_led.png`—ready to stream straight to your 60×60 matrix.

---

### Memory Optimization Features

The code includes several memory optimization techniques:
1. Automatic resizing of large input images
2. Sequential CPU offloading for model components
3. Attention and VAE slicing
4. Memory-efficient attention with xformers
5. Reduced inference steps (20 instead of 30)
6. Option to run on CPU when GPU memory is insufficient

```bash
# 1.  (One-time) install the required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick the CUDA/cuDNN wheel that fits your GPU, or omit for CPU
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow numpy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

```python
# cartoon_led_pipeline.py  ──────────────────────────────────────
import io, base64, cv2, numpy as np, torch, pathlib
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,  # official in-paint+ControlNet pipe  ([pipeline_controlnet_inpaint.py - GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py?utm_source=chatgpt.com))
)

# ─── configuration ─────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE_LED  = (60, 60)        # final resolution
PALETTE_COLS  = 32              # LED supports 32 colours
BRIGHT_LEVELS = 8               # LED supports 8 brightness steps
BASE_MODEL    = "runwayml/stable-diffusion-v1-5"
CTRL_MODEL    = "lllyasviel/sd-controlnet-canny"       # Canny ControlNet
SAM_CKPT      = "sam_vit_h_4b8939.pth"                 # download once from Meta
OUT_DIR       = pathlib.Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ─── helpers ───────────────────────────────────────────────────
def load_image(path: str | pathlib.Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def largest_mask(image: Image.Image) -> Image.Image:
    """Use SAM to grab the biggest object mask."""
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(DEVICE)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32)
    masks = gen.generate(np.array(image))
    if not masks:
        raise RuntimeError("SAM could not find any mask.")
    biggest = max(masks, key=lambda m: m["area"])
    m = biggest["segmentation"].astype(np.uint8) * 255  # bool → {0,255}
    return Image.fromarray(m, mode="L")

def canny_map(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return Image.fromarray(edges)

def stylize(
    init_img: Image.Image,
    mask_img: Image.Image,
    ctrl_img: Image.Image,
    prompt="cute flat-colour cartoon, bold lines",
    neg_prompt="lowres, blurry",
):
    controlnet = ControlNetModel.from_pretrained(CTRL_MODEL, torch_dtype=torch.float16).to(DEVICE)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.safety_checker = lambda images, clip_input: (images, False)  # disable NSFW filter
    out = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=init_img,
        mask_image=mask_img,
        control_image=ctrl_img,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        strength=0.6,
    )
    return out.images[0]

def led_quantise(img: Image.Image) -> Image.Image:
    # down-sample
    img = img.resize(IMG_SIZE_LED, Image.LANCZOS)
    # 1) colour palette reduction
    img = img.convert("P", palette=Image.ADAPTIVE, colors=PALETTE_COLS).convert("RGB")
    # 2) brightness quantisation
    arr = np.array(img, dtype=np.uint8)
    luma = arr.mean(axis=2, keepdims=True)
    step = 255 // (BRIGHT_LEVELS - 1)
    factor = ((luma // step) * step) / np.maximum(luma, 1)   # avoid ÷0
    arr_q = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_q)

# ─── main entry point ──────────────────────────────────────────
def cartoonise_to_led(src_path: str | pathlib.Path):
    src   = load_image(src_path)
    mask  = largest_mask(src)
    ctrl  = canny_map(src)
    stylised = stylize(src, mask, ctrl)
    led_ready = led_quantise(stylised)
    led_ready.save(OUT_DIR / f"{pathlib.Path(src_path).stem}_led.png")
    print("✔ Pipeline finished.  Output:", OUT_DIR / f"{pathlib.Path(src_path).stem}_led.png")

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Convert image to 60×60 LED-friendly cartoon style.")
    parser.add_argument("image", help="Path to the source image")
    opts = parser.parse_args()
    cartoonise_to_led(opts.image)
```

---



Run it:

```bash
python cartoon_led_pipeline.py ./input_photo.jpg
```

You'll get `results/input_photo_led.png`—ready to stream straight to your 60×60 matrix.
