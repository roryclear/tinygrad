import json
import torch
import cv2
import urllib
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

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
        shard_path = hf_hub_download(MODEL_ID, shard_name)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("visual."):
                    state[key[len("visual."):]] = f.get_tensor(key)
    self.vision.load_state_dict(state, strict=False)
    self.vision.eval()

  def __call__(self, image):

    with urllib.request.urlopen(image, timeout=30) as response:
        image_bytes = np.frombuffer(response.read(), dtype=np.uint8)

    # Decode with OpenCV (BGR format)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image from URL")

    # Convert BGR -> RGB for the Qwen processor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inputs = self.processor.image_processor(images=image, return_tensors="pt")  
    with torch.inference_mode(): outputs = self.vision(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    embeddings = outputs.pooler_output
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(embeddings)
    print(embeddings.sum().numpy())
    assert embeddings.sum().numpy() == -22122.2


if __name__ == "__main__":
  qwen = Qwen25VLVis()
  qwen(image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
