import json
import torch
import cv2
import urllib
import numpy as np
from copy import deepcopy
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.image_processing_utils import ImagesKwargs, SizeDict, validate_typed_dict, partial
from transformers.image_processing_backends import tvF
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessorKwargs
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from tinygrad import Tensor, dtypes
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    cache=None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if not use_gqa_in_sdpa(attention_mask, key, value):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs["enable_gqa"] = True

    q_length = query.shape[2]
    kv_length = key.shape[2]

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    # TODO: under `torch.export` with dynamic shapes, `q_length > 1` is a `SymBool` and `and` returns the first falsy
    #   operand, so `is_causal` can end up being that `SymBool` (e.g. the seamless_m4t / seamless_m4t_v2 speech
    #   encoders). Reordering the conditions fixes it but breaks the compile requirement above, so it should rather be
    #   handled on the exporter side. See https://github.com/huggingface/transformers/pull/46196#discussion_r3717333141
    is_causal = q_length > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # This scenario can only happen during prefill with an empty StaticCache. Technically, since sdpa's `is_causal` mask alignment
    # is upper-left, `is_causal=True` is enough to correctly compute the attention. However, sdpa will only dispatch to
    # flash kernel if and only if q_length == kv_length, therefore it is more efficient to slice here and remove the masked tokens
    # rather than to use the other available kernels for such a case.
    # Note that we never compile prefill, and even if the user is doing it on its own, prefill and decode are 2 separate graphs
    # anyway, so altering the shapes is fine here
    if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]
        # If we have a position_bias, we need to crop it as well (on last dim, which is the kv seq_len dim)
        if position_bias is not None:
            position_bias = position_bias[:, :, :, :q_length]

    # If we have a position_bias, create the correct floating-point mask by combining it with the existing mask, or a causal mask
    # if `is_causal=True`
    if position_bias is not None:
        attention_mask = create_position_bias_mask(position_bias, attention_mask, is_causal, query, key)
        is_causal = False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

class BaseImageProcessor:
    valid_kwargs = ImagesKwargs

    default_to_square = True
    rescale_factor = 1 / 255
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs): pass

    def _set_attributes(self, **kwargs):
        """Resolve and set instance attributes from kwargs and class-level defaults for all valid kwargs."""
        attributes = {}
        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                attributes[key] = kwarg
            else:
                attributes[key] = deepcopy(getattr(self, key, None))
        for key, value in attributes.items():
            setattr(self, key, value)

        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    def __call__(self, images, *args, **kwargs):
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, *args, **kwargs)

    def process_image(self, *args, **kwargs):
        """
        Process a single raw image into the backend's working format.

        Implemented by backend subclasses (`TorchvisionBackend`, `PilBackend`). Converts a raw input
        (PIL Image, NumPy array, or torch Tensor) to the backend's internal format (`torch.Tensor` for
        Torchvision, `np.ndarray` for PIL), handles RGB conversion and ensures channels-first ordering.
        """
        raise NotImplementedError

    def _preprocess(self, *args, **kwargs):
        """
        Perform the actual batch image preprocessing (resize, center crop, rescale, normalize, pad).

        Implemented by backend subclasses (`TorchvisionBackend`, `PilBackend`). Receives a list of
        already-prepared images (in the backend's format, channels-first) and applies the configured
        preprocessing operations. Returns a `BatchFeature` with the processed pixel values.

        Model-specific processors can override this method to implement custom preprocessing logic
        (e.g., patch-based processing in LLaVA-NeXT).
        """
        raise NotImplementedError

    def _prepare_images_structure(
        self,
        images,
        expected_ndims: int = 3,
    ):
        images = self.fetch_images(images)
        return [images]

    def _prepare_image_like_inputs(
        self,
        images,
        *args,
        expected_ndims: int = 3,
        **kwargs,
    ) -> list[Any]:
        """
        Prepare image-like inputs for processing by converting each image via `process_image`.

        Flattens the input structure and applies `process_image` (implemented by the backend) to each
        individual image, converting raw inputs (PIL, NumPy, Tensor) into the backend's working format
        with channels-first ordering.

        Args:
            images (`ImageInput`):
                The image-like inputs to process.
            expected_ndims (`int`, *optional*, defaults to 3):
                The expected number of dimensions for the images.

        Returns:
            `list[torch.Tensor]` or `list[np.ndarray]`: The prepared images in the backend's format,
            with channels-first ordering.
        """
        images = self._prepare_images_structure(images, expected_ndims=expected_ndims)

        process_image_partial = partial(self.process_image, *args, **kwargs)

        has_nested_structure = len(images) > 0 and isinstance(images[0], list | tuple)

        if has_nested_structure:
            processed_images = [[process_image_partial(img) for img in nested_list] for nested_list in images]
        else:
            processed_images = [process_image_partial(img) for img in images]

        return processed_images

    def _preprocess_image_like_inputs(
        self,
        images,
        *args,
        **kwargs,
    ):
        """
        Preprocess image-like inputs by preparing them and dispatching to `_preprocess`.

        This method first calls `_prepare_image_like_inputs` to convert raw inputs into the backend's
        format, then calls `_preprocess` for the actual batch processing. Override this method in
        model-specific processors that need to handle multiple image-like input types (e.g., images
        and segmentation maps) or need custom orchestration of the preprocessing pipeline.
        """
        images = self._prepare_image_like_inputs(images, **kwargs)
        return self._preprocess(images, *args, **kwargs)

    def preprocess(
        self,
        images,
        *args,
        image_like_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # This is a workaround when adding new image like inputs to existing processors where we cannot
        # modify the order of the input arguments for BC. For details see:
        # https://github.com/huggingface/transformers/pull/47199#discussion_r4131643549
        image_like_kwargs = {} if image_like_kwargs is None else image_like_kwargs

        return self._preprocess_image_like_inputs(images, *args, **image_like_kwargs, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        processor_dict = super().to_dict()

        # Filter out None values that are class defaults
        filtered_dict = {}
        for key, value in processor_dict.items():
            if isinstance(value, SizeDict):
                value = dict(value)
            if value is None:
                class_default = getattr(type(self), key, "NOT_FOUND")
                # Keep None if user explicitly set it (class default is non-None)
                if class_default != "NOT_FOUND" and class_default is not None:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value

        filtered_dict.pop("_valid_processor_keys", None)
        filtered_dict.pop("_valid_kwargs_names", None)
        return filtered_dict


class TorchvisionBackend(BaseImageProcessor):
    """Torchvision backend for GPU-accelerated batched image processing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_attributes(**kwargs)

    @property
    def backend(self) -> str: return "torchvision"

    def fetch_images(self, image_url_or_urls: str | list[str] | list[list[str]]):
        if isinstance(image_url_or_urls, (list, tuple)):
            return [self.fetch_images(x) for x in image_url_or_urls]
        else:
            return image_url_or_urls

    def process_image(
        self,
        image,
        device=None,
        **kwargs,
    ) -> "torch.Tensor":
        image = torch.from_numpy(image).contiguous()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        image = image.permute(2, 0, 1).contiguous()

        if device is not None:
            image = image.to(device)

        return image

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize an image using Torchvision."""
        # Convert PIL resample to torchvision interpolation if needed
        if resample is not None:
            if isinstance(resample, (PILImageResampling, int)):
                interpolation = pil_torch_interpolation_mapping[resample]
            else:
                interpolation = resample
        else:
            interpolation = tvF.InterpolationMode.BILINEAR
        if interpolation == tvF.InterpolationMode.LANCZOS and (
            not is_torchvision_greater_or_equal("0.27") or image.device.type != "cpu"
        ):
            logger.warning_once(
                "LANCZOS resample requires torchvision >= 0.27 and processing on CPU; it is not supported on CUDA or for "
                "torchvision < 0.27. Falling back to BICUBIC which approximates LANCZOS. To match the original model exactly"
                ", upgrade torchvision and move the tensors to CPU before resizing, or use a PIL backend image processor instead."
            )
            interpolation = tvF.InterpolationMode.BICUBIC

        if size.shortest_edge and size.longest_edge:
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )

        # Workaround for torch.compile issue with uint8 on AMD GPUs
        if is_torchdynamo_compiling() and is_rocm_platform():
            return self._compile_friendly_resize(image, new_size, interpolation, antialias)
        return tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    @staticmethod
    def _compile_friendly_resize(
        image: "torch.Tensor",
        new_size: tuple[int, int],
        interpolation=None,
        antialias: bool = True,
    ) -> "torch.Tensor":
        """A wrapper around tvF.resize for torch.compile compatibility with uint8 tensors."""
        if image.dtype == torch.uint8:
            image = image.float() / 256
            image = tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)
            image = image * 256
            image = torch.where(image > 255, 255, image)
            image = torch.where(image < 0, 0, image)
            image = image.round().to(torch.uint8)
        else:
            image = tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)
        return image

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        **kwargs,
    ) -> "torch.Tensor":
        """Rescale an image by a scale factor using Torchvision."""
        return image * scale

    def normalize(
        self,
        image: "torch.Tensor",
        mean,
        std,
        inplace: bool = False,
        **kwargs,
    ) -> "torch.Tensor":
        """Normalize an image using Torchvision."""
        return tvF.normalize(image, mean, std, inplace=inplace)

    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device=None,
    ) -> tuple:
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> "torch.Tensor":
        """Rescale and normalize images using Torchvision (fused for efficiency)."""
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        if do_normalize:
            inplace = images.dtype != torch.float32  # Convert copied tensors inplace for speed
            images = self.normalize(images.to(torch.float32), image_mean, image_std, inplace=inplace)
        elif do_rescale:
            images = self.rescale(images, rescale_factor)

        return images

    def center_crop(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        **kwargs,
    ) -> "torch.Tensor":
        """Center crop an image using Torchvision."""
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        image_height, image_width = image.shape[-2:]
        crop_height, crop_width = size.height, size.width

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            image = tvF.pad(image, padding_ltrb, fill=0)
            image_height, image_width = image.shape[-2:]
            if crop_width == image_width and crop_height == image_height:
                return image

        crop_top = int((image_height - crop_height) / 2.0)
        crop_left = int((image_width - crop_width) / 2.0)
        return tvF.crop(image, crop_top, crop_left, crop_height, crop_width)

class Qwen2VLImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = 3 #PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073] #OPENAI_CLIP_MEAN
    image_std = [0.26862954, 0.26130258, 0.27577711] # OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Qwen2VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    def __init__(self, **kwargs):
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else size
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if (max_pixels := kwargs.pop("max_pixels", None)) is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        super().__init__(size=size, **kwargs)

    def preprocess(
        self,
        images: Any,
        **kwargs,
    ):
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: "torch.Tensor",
        size: Any,
        resample: Any,
        factor: int,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input image aspect ratio."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")

        height, width = images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return super().resize(
            image=images,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def patchify(
        self,
        images: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."""
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)

        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )
        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: Any,
        resample: Any,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: Any,
        **kwargs,
    ):  
        grouped_images_index = {0: (0, 0)}
        grouped_images = {0: images[0].unsqueeze(0)}
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            resized_images_grouped[shape] = stacked_images
        resized_images = [resized_images_grouped[0][0]]
        grouped_images_index = {0: (0, 0)}
        grouped_images = {0: resized_images[0].unsqueeze(0)}
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_images_grouped[shape] = patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = [processed_images_grouped[0][0]]
        processed_grids_ordered = [processed_grids[0][0]]
        pixel_values = processed_images[0] if len(processed_images) == 1 else torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w

class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    config: Any
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
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

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
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
        max_window_seqlen = get_max_seqlen(
            cu_window_seqlens, self.config, kwargs=kwargs, kwarg_name="max_window_seqlen"
        )

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

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


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

        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            sdpa_attention_forward(
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
        self.act_fn = nn.functional.silu

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

class Qwen2VLPreTrainedModel(PreTrainedModel):
    config: Any
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

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


