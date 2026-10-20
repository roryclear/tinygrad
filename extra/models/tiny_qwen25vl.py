import json
import torch
import cv2
import urllib
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from tinygrad import Tensor, dtypes

class Qwen25VLVis():
  def __init__(self, tok=None, res:list=[560, 560]): # res is 20x toks_per_img like qwen3
    self.res = res
    self.tok = tok
    self.toks_per_img = (self.res[0] * self.res[1]) // (28*28) # 28x28 tokens per pixel https://www.alibabacloud.com/help/en/model-studio/vision
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load configuration and image preprocessing only.
    config = AutoConfig.from_pretrained(MODEL_ID)
    self.image_processor = Qwen2VLImageProcessor()

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

    if self.tok: # todo
      self.prefix = Tensor(self.tok.encode("<|im_start|>user\n<|vision_start|>"))
      self.suffix = Tensor(self.tok.encode("<|vision_end|>\n<|im_end|>\n"))

  def __call__(self, lang=None, image=None, start_pos=None, end_turn=False):
    if isinstance(image, (bytes, bytearray)): image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image.shape[:2] != self.res:
      target_h, target_w = self.res[:2]
      s = min(target_w / image.shape[1], target_h / image.shape[0])
      r = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)))
      image = cv2.copyMakeBorder(r, (target_h - r.shape[0]) // 2, target_h - r.shape[0] - (target_h - r.shape[0]) // 2, (target_w - r.shape[1]) // 2, target_w - r.shape[1] - (target_w - r.shape[1]) // 2, cv2.BORDER_CONSTANT, value=0)
    self.prefill(lang=lang, image=image, start_pos=start_pos)

    if end_turn: lang._cached_tokens.extend(self.tok.end_turn())
    lang._cached_tokens.extend([0] * (self.toks_per_img + self.prefix.shape[0] + self.suffix.shape[0]))

  def prefill(self, lang=None, image=None, start_pos=None):
    input_ids = Tensor.cat(self.prefix, Tensor.zeros(self.toks_per_img), self.suffix).unsqueeze(0).cast(dtypes.int)
    hidden_states = lang.token_embd(input_ids).cast(dtypes.float)
    inputs = self.image_processor(images=image, return_tensors="pt")
    embeddings = self.vision(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).pooler_output
    embeddings = Tensor(embeddings.detach().numpy())
    hidden_states[:, self.prefix.shape[0]:-self.suffix.shape[0], :] = embeddings.unsqueeze(0)

    for i in range(len(lang.blk)): hidden_states = lang.blk[i](hidden_states, start_pos=start_pos)
    hidden_states.realize()
    
if __name__ == "__main__": 
  qwen = Qwen25VLVis()
  url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
  with urllib.request.urlopen(url, timeout=30) as response: image = np.frombuffer(response.read(), dtype=np.uint8)
  image = cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

  target_h, target_w = 560, 560
  s = min(target_w / image.shape[1], target_h / image.shape[0])
  r = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)))
  image = cv2.copyMakeBorder(r, (target_h - r.shape[0]) // 2, target_h - r.shape[0] - (target_h - r.shape[0]) // 2, (target_w - r.shape[1]) // 2, target_w - r.shape[1] - (target_w - r.shape[1]) // 2, cv2.BORDER_CONSTANT, value=0)

  inputs = qwen.image_processor(images=image, return_tensors="pt")
  embeddings = qwen.vision(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).pooler_output
  print(embeddings, embeddings.sum().item())
  np.testing.assert_allclose(embeddings.sum().item(), -12742.4921875)
  print("done")

