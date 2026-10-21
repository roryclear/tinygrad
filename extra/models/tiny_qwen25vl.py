import json
import torch
import cv2
import urllib
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLPreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import is_flash_attention_requested
from tinygrad import Tensor, dtypes
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Qwen2_5_VLVisionRotaryEmbedding(nn.Module):
    """
    Simple axial 2D rope with same freqs used for H and W grids. The freqs are
    pre-computed using `head-dim//4` which is later used to concat H and W positions.
    The final angles rotate over the whole head dim, no partial rotation involved.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(
        config: Any, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (2, N) — row 0 = h coords, row 1 = w coords
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos, sin

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Any = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths used for packed variable-length attention in Flash Attention kernels.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2_5_VLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2_5_VLRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5_VLRMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

class ModelOutput(OrderedDict):
    def __setitem__(self, key, value):
        super().__setattr__(key, value)

def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int | torch.Tensor,
    include_temporal: bool = False,
    kwargs: dict | None = None,
) -> torch.Tensor:
    if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
        return position_ids

    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        spatial_merge_size = torch.tensor([spatial_merge_size], device=device).expand(len(grid_thw))

    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        if include_temporal:
            tpos_ids = torch.arange(t, device=device).repeat_interleave(h * w)
            position_ids.append(torch.stack([tpos_ids, hpos_ids.repeat(t), wpos_ids.repeat(t)], dim=-1))
        else:
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    return torch.cat(position_ids, dim=0)

def get_vision_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kwargs is not None:
        window_index = kwargs.pop("window_index", None)
        cu_window_seqlens = kwargs.pop("cu_window_seqlens", None)
        if window_index is not None and cu_window_seqlens is not None:
            return window_index, cu_window_seqlens
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=grid_thw.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return window_index, cu_window_seqlens

def get_vision_cu_seqlens(
    grid_thw: torch.Tensor, merge_temporal: bool = False, kwargs: dict | None = None
) -> torch.Tensor:
    dtype = torch.int32
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)

def get_vision_attention_seqlens(
    grid_thw: torch.Tensor,
    config: Any,
    merge_temporal: bool = False,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, int | None]:
    cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)
    max_seqlen = get_max_seqlen(cu_seqlens)
    return cu_seqlens, max_seqlen

def get_max_seqlen(cu_seqlens: torch.Tensor): return (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | list[int] | tuple[int, int] = 14,
        temporal_patch_size: int | list[int] | tuple[int, int] = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config: Any
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": Any,
        "attentions": Any,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.rotary_pos_emb = Qwen2_5_VLVisionRotaryEmbedding(config)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

        self.post_init()

    def permute_input_for_window_attn(self, input_tensor: torch.Tensor, window_index: torch.Tensor, seq_len: int):
        input_tensor = input_tensor.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        input_tensor = input_tensor[window_index, :, :]
        input_tensor = input_tensor.reshape(seq_len, -1)
        return input_tensor

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
            kwargs=kwargs,
        )
        max_window_seqlen = get_max_seqlen(cu_window_seqlens)

        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        hidden_states = self.permute_input_for_window_attn(hidden_states, window_index, seq_len=seq_len)

        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        position_embeddings = tuple(
            self.permute_input_for_window_attn(freq, window_index, seq_len=seq_len) for freq in position_embeddings
        )

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_window_seqlen

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = self.merger(hidden_states)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return ModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

class Qwen25VLVis():
  def __init__(self, tok=None, res:list=[560, 560]): # res is 20x toks_per_img like qwen3
    self.res = res
    self.tok = tok
    self.toks_per_img = (self.res[0] * self.res[1]) // (28*28) # 28x28 tokens per pixel https://www.alibabacloud.com/help/en/model-studio/vision
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

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

