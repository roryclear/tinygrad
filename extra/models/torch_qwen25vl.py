import json
import torch
import requests

from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

class Qwen25VLVis():

  def __init__(self):
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load configuration and image preprocessing only.
    config = AutoConfig.from_pretrained(MODEL_ID)
    self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Instantiate the vision tower WITHOUT loading the LLM.
    self.vision = Qwen2_5_VisionTransformerPretrainedModel(
        config.vision_config
    )

    index_path = hf_hub_download( MODEL_ID, "model.safetensors.index.json")
    with open(index_path) as f: weight_map = json.load(f)["weight_map"]

    vision_keys = {
        key: shard
        for key, shard in weight_map.items()
        if key.startswith("visual.")
    }
    shards = sorted(set(vision_keys.values()))
    state = {}

    for shard_name in shards:
        shard_path = (
            shard_name
            if shard_name.endswith(".safetensors")
            and "/" in shard_name
            else hf_hub_download(MODEL_ID, shard_name)
        )

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("visual."):
                    state[key[len("visual."):]] = f.get_tensor(key)

    self.vision.eval()

  def __call__(self, image):
    image = Image.open(requests.get(image, timeout=30, stream=True).raw).convert("RGB")  
    inputs = self.processor.image_processor(images=image, return_tensors="pt")  
    with torch.inference_mode(): outputs = qwen.vision(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    embeddings = outputs.pooler_output
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(embeddings) 

if __name__ == "__main__":
  qwen = Qwen25VLVis()
  qwen(image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
