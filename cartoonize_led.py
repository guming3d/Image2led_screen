# cartoon_led_pipeline.py  ──────────────────────────────────────
# DEPENDENCY ERROR FIX:
# If you encounter: "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
# Run this command to install the missing dependency:
# sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
#
import io, base64, cv2, numpy as np, torch, pathlib
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,  # official in-paint+ControlNet pipe :contentReference[oaicite:0]{index=0}
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
