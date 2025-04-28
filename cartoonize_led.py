# cartoon_led_pipeline.py  ──────────────────────────────────────
# DEPENDENCY ERROR FIX:
# If you encounter: "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
# Run this command to install the missing dependency:
# sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
#
import io, base64, cv2, numpy as np, torch, pathlib, os
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,  # official in-paint+ControlNet pipe :contentReference[oaicite:0]{index=0}
)

# ─── configuration ─────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE_LED  = (60, 60)        # final resolution
MAX_INPUT_SIZE = 512            # Maximum dimension for input images
PALETTE_COLS  = 32              # LED supports 32 colours
BRIGHT_LEVELS = 8               # LED supports 8 brightness steps
BASE_MODEL    = "runwayml/stable-diffusion-v1-5"
CTRL_MODEL    = "lllyasviel/sd-controlnet-canny"       # Canny ControlNet
SAM_VARIANT   = "vit_b"                       #  "vit_b" or "vit_l"
SAM_CKPT      = "sam_vit_b_01ec64.pth"        # download once
OUT_DIR       = pathlib.Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ─── helpers ───────────────────────────────────────────────────
def load_image(path: str | pathlib.Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    
    # Resize large images to prevent OOM errors
    if max(img.size) > MAX_INPUT_SIZE:
        print(f"⚠️ Resizing image from {img.size} to fit within {MAX_INPUT_SIZE}px")
        aspect_ratio = img.width / img.height
        if img.width > img.height:
            new_width = MAX_INPUT_SIZE
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = MAX_INPUT_SIZE
            new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return img

def largest_mask(image: Image.Image) -> tuple[Image.Image, SamAutomaticMaskGenerator, torch.nn.Module]:
    """Use SAM to grab the biggest object mask. Returns mask, generator, and model."""
    sam = sam_model_registry[SAM_VARIANT](checkpoint=SAM_CKPT).to("cpu")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
    masks = gen.generate(np.array(image))
    if not masks:
        # Clean up before raising error
        del sam, gen
        raise RuntimeError("SAM could not find any mask.")
    biggest = max(masks, key=lambda m: m["area"])
    m = biggest["segmentation"].astype(np.uint8) * 255  # bool → {0,255}
    return Image.fromarray(m, mode="L"), gen, sam # Return gen and sam

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
    # Use lower precision where possible
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    
    # Load ControlNet with memory-efficient settings
    print("🔄 Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        CTRL_MODEL, 
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(DEVICE)
    
    # Load the pipeline with memory-efficient settings
    print("🔄 Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    
    # Enable memory-efficient configurations
    print("🔧 Setting up memory optimizations...")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing(slice_size="max")
    pipe.enable_vae_slicing()
    
    # Enable CPU offloading of components when not in use
    if DEVICE == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)  # If not CUDA, just move everything to the device once
        
    # Set to None to disable safety checker properly
    pipe.safety_checker = None
    
    # Use fewer steps to reduce memory usage
    print("🖌️ Running inference...")
    out = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=init_img,
        mask_image=mask_img,
        control_image=ctrl_img,
        num_inference_steps=20,  # Reduced from 30
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,  # Slightly lower than original
        strength=0.6,
    )
    
    # Clean up GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        
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
    print(f"🖼️ Processing image: {src_path}")
    # Load and prepare images
    src = load_image(src_path)
    print(f"📐 Image dimensions: {src.width}x{src.height}")
    
    # Extract mask using SAM
    print("🔍 Generating mask with SAM...")
    mask, sam_gen, sam_model = largest_mask(src)
    
    # Free SAM resources immediately
    print("🧹 Cleaning up SAM resources...")
    del sam_model, sam_gen
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Generate edge map for ControlNet
    print("✏️ Creating edge detection map...")
    ctrl = canny_map(src)
    
    # Run the stylization pipeline
    print("🎨 Starting stylization (this may take a while)...")
    stylised = stylize(src, mask, ctrl)
    
    # Quantize for LED display
    print("📉 Quantizing for LED display...")
    led_ready = led_quantise(stylised)
    
    # Save the result
    output_path = OUT_DIR / f"{pathlib.Path(src_path).stem}_led.png"
    led_ready.save(output_path)
    print(f"✅ Pipeline finished. Output saved to: {output_path}")

if __name__ == "__main__":
    import argparse, sys
    from datetime import datetime
    
    # Set PyTorch memory allocation configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert image to 60×60 LED-friendly cartoon style.")
    parser.add_argument("image", help="Path to the source image")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing (slower but uses less memory)")
    opts = parser.parse_args()
    
    # Override DEVICE if CPU is requested
    if opts.cpu:
        DEVICE = "cpu"
        print("⚠️ Forcing CPU mode (this will be much slower)")
    
    # Print memory info before starting
    if DEVICE == "cuda" and torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💻 GPU: {torch.cuda.get_device_name(0)} with {free_mem:.1f}GB total memory")
    
    # Record start time
    start_time = datetime.now()
    print(f"⏱️ Starting process at {start_time.strftime('%H:%M:%S')}")
    
    try:
        cartoonise_to_led(opts.image)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"⏱️ Process completed in {duration.total_seconds():.1f} seconds")
    except Exception as e:
        print(f"❌ Error: {e}")
        if DEVICE == "cuda":
            print("💡 Try running with --cpu if you're experiencing memory issues")
        raise
