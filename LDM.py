import jax
import numpy as np
import os
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
from IPython.display import display
from PIL import Image
from google.colab import files

# Load the pipeline and parameters
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="flax", dtype=jax.numpy.bfloat16
)

# Define prompt and other parameters
prompt = "a horse in the forest"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

# Determine the number of devices available
num_samples = jax.device_count()

# Prepare inputs for each device
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# Shard inputs and RNG
params = replicate(params)
prng_seed = jax.random.split(prng_seed, num_samples)
prompt_ids = shard(prompt_ids)

# Generate images
images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))

# Save images to the local directory
os.makedirs("generated_images", exist_ok=True)  # Create a folder to save images
for i, img in enumerate(images):
    image_path = f"generated_images/astronaut_rides_horse_{i}.png"
    img.save(image_path)
    print(f"Saved image to {image_path}")
    display(img)  # Display each generated image in Colab

# Optional: Zip and download all images
!zip -r generated_images.zip generated_images
files.download("generated_images.zip")



