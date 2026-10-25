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
from transformers.modeling_utils import EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin, DistributedMixin, PreTrainedConfig, GenerationConfig, OutputRecorder, Callable, checkpoint, os, KernelConfig, SpecificPreTrainedModelType, LoadStateDictConfig, LoadStateDictInfo, CompileConfig, HfQuantizer, Iterator, inspect, get_type_hints, hub_kernels, is_flash_attention_requested, is_kernel, ALL_ATTENTION_FUNCTIONS, ALL_EXPERTS_FUNCTIONS, ALL_FP8_EXPERTS_FUNCTIONS, sys, LOSS_MAPPING, re, _CAN_RECORD_REGISTRY, get_torch_context_manager_or_global_device, init, ROPE_INIT_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer
from tinygrad import Tensor, dtypes
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

class PreTrainedModel(
    nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin, DistributedMixin
):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings

    Class attributes (overridden by derived classes):

        - **config_class** ([`PreTrainedConfig`]) -- A subclass of [`PreTrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):
    """

    # General model properties
    config_class: type[PreTrainedConfig] | None = None
    generation_config_class: type[GenerationConfig] = GenerationConfig  # default, used with `GenerationMixin`
    _auto_class = None
    base_model_prefix: str = ""
    _is_stateful: bool = False
    model_tags: list[str] | None = None

    # Input-related properties
    main_input_name: str = "input_ids"
    # Attributes used mainly in multimodal LLMs, though all models contain a valid field for these
    # Possible values are: text, image, video, audio and time
    input_modalities: str | list[str] = "text"

    # Device-map related properties
    _no_split_modules: set[str] | list[str] | None = None
    _skip_keys_device_placement: set[str] | list[str] | None = None

    # Specific dtype upcasting
    # `_keep_in_fp32_modules` will upcast to fp32 only if the requested dtype is fp16
    # `_keep_in_fp32_modules_strict` will upcast to fp32 independently if the requested dtype is fp16 or bf16
    _keep_in_fp32_modules: set[str] | list[str] | None = None
    _keep_in_fp32_modules_strict: set[str] | list[str] | None = None

    # Loading-specific properties
    # A dictionary `{"target": "source"}` of checkpoint keys that are potentially tied to one another
    _tied_weights_keys: dict[str, str] = None
    # A list of `re` patterns describing keys to ignore if they are missing from checkpoints to avoid warnings
    _keys_to_ignore_on_load_missing: set[str] | list[str] | None = None
    # A list of `re` patterns describing keys to ignore if they are unexpected in the checkpoints to avoid warnings
    _keys_to_ignore_on_load_unexpected: set[str] | list[str] | None = None
    # A list of keys to ignore when saving the model
    _keys_to_ignore_on_save: set[str] | list[str] | None = None

    # Attention interfaces support properties
    _supports_sdpa: bool = False
    _supports_flash_attn: bool = False
    _supports_flex_attn: bool = False
    # Model's compatible flash kernels (e.g., "kernels-community/flash-mla") defaulting to the first one supported
    # by the current hardware
    _compatible_flash_implementations: list[str] | None = None

    # Set to `False` by models that can never run under context parallelism, whatever their config
    # (attention sinks, for instance, which SDPA cannot express). Models whose *config* rules it out are
    # handled by `supports_context_parallel` below, so this stays `True` for almost everything.
    _supports_context_parallel: bool = True

    # Advanced functionalities support
    supports_gradient_checkpointing: bool = False
    _can_compile_fullgraph: bool = False
    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend: bool = False
    # A mapping describing what outputs can be captured by `capture_outputs` decorator during the forward pass
    _can_record_outputs: dict | None = None

    @property
    def supports_context_parallel(self) -> bool:
        """Whether this model can be trained with context parallelism.

        Context parallelism shards the sequence and can only express full causal attention: the per-layer
        mask is dropped, so a layer using a stricter mask (sliding-window or chunked attention) would
        silently train as full causal instead. A layer carrying a recurrent state along the sequence
        (linear attention) is ruled out for a different reason: the state is never exchanged between ranks.
        """
        if not self._supports_context_parallel:
            return False
        config = self.config.get_text_config()
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            return all(layer_type == "full_attention" for layer_type in layer_types)
        # Models predating `layer_types` (Mistral, for one) apply a sliding window to every layer whenever
        # `sliding_window` is set.
        return getattr(config, "sliding_window", None) is None

    @property
    @torch.compiler.allow_in_graph
    def can_record_outputs(self) -> dict[str, OutputRecorder]:
        """
         Maps output names (e.g., "attentions", "hidden_states")
         to either:
             - A module class (e.g., `LlamaDecoderLayer`), using default index conventions:
                 * index = 0 for a key that contains "hidden_states" (e.g. "hidden_states" or "vision_hidden_states")
                 * index = 1 for any other key: "attentions", "cross_attentions", etc.
             - A class name as a string, when the class is not importable at declaration time.
             - An `OutputRecorder(...)` with `target_class`, optional `index`, and `layer_name`.
             - A list of any of the above, to record outputs from several module types under one key.

         Examples:
             These two are equivalent:

         ```python
             _can_record_outputs = {
                 "attentions": LlamaAttention,
                 "hidden_states": LlamaDecoderLayer
             }

             _can_record_outputs = {
                 "attentions": OutputRecorder(LlamaAttention, index=1),
                 "hidden_states": OutputRecorder(LlamaDecoderLayer, index=0)
             }
        ```

         This means you can record outputs from the same class, by specifying a layer name. Before
         collecting outputs, we check that they come from this layer.

         If you have cross attention that come from `LlamaAttention` and self attention that also
         come from `LlamaAttention` but from `self_attn` you can do this:

         ```python
         class LlamaModel(PreTrainedModel):
             _can_record_outputs = {
                 "attentions": OutputRecorder(LlamaAttention, index=1, layer_name="self_attn"),
                 "cross_attentions": OutputRecorder(LlamaAttention, index=1, layer_name="cross_attn")
             }

        ```
        """
        return self._can_record_outputs or {}

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        """
        `dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = inspect.get_annotations(cls).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
        self.name_or_path = config.name_or_path

        # Make kernel_config an attribute that can be used by the model.
        self.kernel_config = None

        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation,
            is_init_check=True,
            # We need to use this constant that is set through context manager as it cannot be forwarded in the model's __init__
            allow_all_kernels=hub_kernels.ALLOW_ALL_KERNELS,
        )
        # Check the experts implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._experts_implementation_internal = self._check_and_adjust_experts_implementation(
            self.config._experts_implementation
        )
        if self.can_generate():
            # `from_model_config` is a legacy behavior -- we shouldn't set generation flags in the model config
            try:
                self.generation_config = self.generation_config_class.from_model_config(config)
            except NotImplementedError:
                self.generation_config = self.generation_config_class()

        # for initialization of the loss
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs  # added for executorch support only

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        It is also used to obtain all correct static properties (parallelism plans, tied_weights_keys, _keep_in_fp32_modules, etc)
        correctly in the case of composite models (that is, the top level model should know about those properties from its children).
        """
        # Attach the different parallel plans and tied weight keys to the top-most model, so that everything is
        # easily available.
        self.init_parallel_plans()
        # Current submodel should register its tied weights
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        # Current submodel should register its `_keep_in_fp32_modules`
        self._keep_in_fp32_modules = set(self._keep_in_fp32_modules or [])
        self._keep_in_fp32_modules_strict = set(self._keep_in_fp32_modules_strict or [])
        # Current submodel must register its `_no_split_modules`/`_skip_keys_device_placement` as well for device_map
        self._no_split_modules = set(self._no_split_modules or [])
        self._skip_keys_device_placement = set(self._skip_keys_device_placement or [])
        # Current submodel must register the `_keys_to_ignore_on_load_unexpected/missing`
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or [])
        self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing or [])
        self._keys_to_ignore_on_save = set(self._keys_to_ignore_on_save or [])

        # Iterate over children only: as the final model is created, this is enough to gather the properties from all submodels.
        # This works because the way the `__init__` and `post_init` are called on all submodules is depth-first in the graph
        for name, module in self.named_children():
            # Always attach the keys of the children (if the children's config says to NOT tie, then it's empty)
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update({f"{name}.{k}": f"{name}.{v}" for k, v in tied_keys.copy().items()})
            # Record keep_in_fp_32 modules from the children as well
            if keep_fp32 := getattr(module, "_keep_in_fp32_modules", None):
                self._keep_in_fp32_modules.update(keep_fp32)
            if keep_fp32_strict := getattr(module, "_keep_in_fp32_modules_strict", None):
                self._keep_in_fp32_modules_strict.update(keep_fp32_strict)
            # Record `_no_split_modules`/`_skip_keys_device_placement` from the children
            if no_split := getattr(module, "_no_split_modules", None):
                self._no_split_modules.update(no_split)
            if skip_keys := getattr(module, "_skip_keys_device_placement", None):
                self._skip_keys_device_placement.update(skip_keys)
            # Record `_keys_to_ignore_on_load_unexpected/missing` from the children - note that we do not add the name (prefix)
            # of the current module to the child's key, as this is matched by regex anyway and adding prefix could break the regex
            if ignore_unexpected := getattr(module, "_keys_to_ignore_on_load_unexpected", None):
                self._keys_to_ignore_on_load_unexpected.update(ignore_unexpected)
            if ignore_missing := getattr(module, "_keys_to_ignore_on_load_missing", None):
                self._keys_to_ignore_on_load_missing.update(ignore_missing)
            # This one is matched exactly, not by regex, so we need to add the prefix
            if ignore_save := getattr(module, "_keys_to_ignore_on_save", None):
                self._keys_to_ignore_on_save.update({f"{name}.{k}" for k in ignore_save})

        # Maybe initialize the weights and tie the keys
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def dequantize(self, dtype=None):
        """
        Potentially dequantize the model in case it has been quantized by a quantization method that support
        dequantization.
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("You need to first quantize your model in order to dequantize it")

        return hf_quantizer.dequantize(self, dtype=dtype)

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def add_model_tags(self, tags: list[str] | str) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[list[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # For BC on the old `torch_dtype`
        dtype = kwargs.pop("dtype", config.dtype)
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
            # if both kwargs are provided, use `dtype`
            dtype = dtype if dtype != config.dtype else torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Keep the resolved `dtype` on the config so it matches the actual weights, the same way
        # `from_pretrained` does. Also set it on all subconfigs to avoid dtype mismatch. When "auto"
        # dtype with nested models, we can't dispatch different dtype per backbone module
        config.dtype = dtype
        for sub_config_key in config.sub_configs:
            if (sub_config := getattr(config, sub_config_key)) is not None:
                sub_config.dtype = dtype

        # If passing `attn_implementation` as kwargs, respect it (it will be applied recursively on subconfigs)
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        # If passing `experts_implementation` as kwargs, respect it (it will be applied recursively on subconfigs)
        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        # Needed if the attn_implementation is an outside `kernels-community` kernel
        allow_all_kernels = kwargs.get("allow_all_kernels", False)

        init_contexts = [apply_patches()]
        if dtype is not None:
            init_contexts.append(local_torch_dtype(dtype, cls.__name__))
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())

        needs_zero3_init = is_deepspeed_zero3_enabled() and not _is_quantized and not _is_ds_init_called
        if needs_zero3_init:
            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            import deepspeed

            init_contexts.extend(
                [
                    init.no_init_weights(),
                    deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                    set_zero3_state(),
                ]
            )

        # Instantiate the model
        with ContextManagers(init_contexts):
            model = cls(config, **kwargs)
            patch_output_recorders(model)

        # Under ZeRO-3, parameters were partitioned into empty tensors during construction,
        # so weight init was suppressed. Re-initialize using the ZeRO-3 variant which gathers
        # each module's parameters before init to avoid OOM on large models.
        if needs_zero3_init:
            from .integrations.deepspeed import initialize_weights_zero3

            initialize_weights_zero3(model)
            model.tie_weights()

        return model

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`, from the `GenerationMixin`
        or one with a similar interface.

        Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
        for instance, the model instance will have a populated `generation_config` attribute.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` (or a class ending in the same substring) -> can generate
        if "GenerationMixin" in str(cls.__bases__):
            return True
        # The class inherits from a class that can generate (recursive check) -> can generate
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        # Detects whether `prepare_inputs_for_generation` has been overwritten in the model. Prior to v4.45, this
        # was how we detected whether a model could generate.
        if hasattr(cls, "prepare_inputs_for_generation"):  # implicit: doesn't inherit `GenerationMixin`
            logger.warning(
                f"{cls.__name__} has generative capabilities, as `prepare_inputs_for_generation` is explicitly "
                "defined. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, "
                "`PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability "
                "to call `generate` and other related functions."
                "\n  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the "
                "model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes"
                "\n  - If you are the owner of the model architecture code, please modify your model class such that "
                "it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception)."
                "\n  - If you are not the owner of the model architecture class, please contact the model code owner "
                "to update it."
            )
        # Otherwise, can't generate
        return False

    def _flash_attn_import_error(
        self,
        flash_attn_version: int,
        general_availability_check: Callable,
        pkg_availability_check: Callable,
        supported_devices: tuple[tuple[Callable, str], ...],
        custom_supported_devices: tuple[tuple[Callable, str], ...] = (),
        cuda_min_major_version: int | None = None,
    ):
        """
        Checks whether the specified Flash Attention version is supported and if not, searches for the specific reason
        on why it failed - package import and/or device incompatibility issues.

        Args:
            flash_attn_version (`int`):
                The requested version of Flash Attention.
            general_availability_check (`Callable`):
                Checks whether our `is_available` function detects the specific FA version. Failing reasons
                are then checked for one-by-one.
            pkg_availability_check (`Callable`):
                Checks whether the package could theoretically be detected in the environment by the init structures.
                This is not a sure-fire check as device compatibility with FA is just as important.
            supported_devices (`tuple[tuple[Callable, str]]`):
                Essentially a list (for mutable kwargs reasons a tuple) of the supported devices in the format of
                `(device_availability_check, device_name)`, i.e. a pair of the associated device's name and whether
                it is available in the environment.
            custom_supported_devices (`tuple[tuple[Callable, str]]`, *optional*, defaults to `()`):
                Essentially a list (for mutable kwargs reasons a tuple) of the custom supported devices in the format of
                `(device_availability_check, info_message)`. These custom devices have custom logic outside the torch
                ecosystem either via kernels or other packages and hence have early checks for availability.
            cuda_min_major_version (`int`, *optional*):
                The minimum major cuda version supported for this version of Flash Attention. This is mostly
                affecting more recent versions which are more specialized to the features of new hardware.
        """
        # Certain devices have custom workarounds e.g. with their own package distribution (NPU) or via kernels (XPU)
        for device_availability_check, info_message in custom_supported_devices:
            if device_availability_check():
                logger.info(info_message)
                return

        if not general_availability_check():
            preface = f"FlashAttention{flash_attn_version} has been toggled on, but it cannot be used due to the following error:"

            # Can the package be seen in the import structure
            if not pkg_availability_check():
                raise ImportError(
                    f"{preface} the package for FlashAttention{flash_attn_version} doesn't seem to be installed."
                )
            # Minimum version (FA2 only)
            elif flash_attn_version == 2 and not is_flash_attn_greater_or_equal("2.3.3"):
                raise ImportError(f"{preface} FlashAttention{flash_attn_version} requires at least version `2.3.3`.")
            else:
                # Supported devices availability
                device_availability_checks, device_names = zip(*supported_devices)
                if not any(device_availability_check() for device_availability_check in device_availability_checks):
                    raise ImportError(
                        f"{preface} FlashAttention{flash_attn_version} is not available on CPU. Please make sure you are on any of the supported devices: {device_names}."
                    )
                # Cuda major versions (more recent FA versions are specialized to newer cuda devices)
                elif cuda_min_major_version is not None and is_torch_cuda_available():
                    major, _ = torch.cuda.get_device_capability()
                    if major < cuda_min_major_version:
                        raise ImportError(
                            f"{preface} FlashAttention{flash_attn_version} requires compute capability >= {cuda_min_major_version}, but found {torch.cuda.get_device_capability()} with compute capability {major}.x"
                        )

    def _flash_attn_can_dispatch(self, flash_attn_version: int, is_init_check: bool = False) -> bool:
        """
        Check the availability of Flash Attention for a given model.

        Args:
            flash_attn_version (`int`):
                The requested version of Flash Attention.
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        if not self._supports_flash_attn:
            raise ValueError(
                f"{self.__class__.__name__} does not support Flash Attention {flash_attn_version} yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{self.config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if flash_attn_version not in [2, 3, 4]:
            raise ValueError(f"Requested Flash Attention {flash_attn_version} which is not supported.")

        # Check if we can even use the FA version based on the env of the user
        self._flash_attn_import_error(**FLASH_ATTENTION_COMPATIBILITY_MATRIX[flash_attn_version])

        # Check for attention dropout, which is incompatible with newer FA versions
        # (many should not really care about dropout as it is not super effective, hence warning for now)
        if flash_attn_version > 2:
            if hasattr(self.config, "attention_dropout") and self.config.attention_dropout > 0:
                logger.warning_once(
                    f"You are attempting to use Flash Attention {flash_attn_version} with dropout. "
                    "This might lead to unexpected behaviour as this is not supported on recent versions of Flash Attention."
                )

        # People often move dtypes after init so we only warn in those cases
        dtype = self.config.dtype
        if dtype is None:
            logger.warning_once(
                f"You are attempting to use Flash Attention {flash_attn_version} without specifying a dtype. This might lead to unexpected behaviour"
            )
        elif dtype is not None and dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                f"Flash Attention {flash_attn_version} only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {self.__class__.__name__} is {dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                f' or load the model with the `dtype` argument. Example: `model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="flash_attention_{flash_attn_version}", dtype=torch.float16)`'
            )

        # With the early check, the parameters are not yet initialized correctly
        if not is_init_check:
            param_devices = list({param.device for param in self.parameters()})
            if len(param_devices) == 1 and param_devices[0].type == "cpu":
                found_device = False
                for device_availability_check, device_name in FLASH_ATTENTION_COMPATIBILITY_MATRIX[flash_attn_version][
                    "supported_devices"
                ]:
                    if device_availability_check():
                        found_device = True
                        logger.warning_once(
                            f"You are attempting to use Flash Attention {flash_attn_version} with a model not initialized on GPU. Please make sure to have "
                            "access to a GPU and either initialise the model on a GPU by passing a device_map or initialising the model on CPU and then "
                            f"moving it to GPU, e.g. with `model.to('{device_name}')`."
                        )
                        break

                if not found_device:
                    raise ValueError(
                        f"You are attempting to use Flash Attention {flash_attn_version} with a model not initialized on GPU and with no GPU available. "
                        "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                        "or initialising the model on CPU and then moving it to GPU."
                    )

        # If no error raise by this point, we can return `True`
        return True

    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of SDPA for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        if not self._supports_sdpa:
            raise ValueError(
                f"{self.__class__.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )

        return True

    def _grouped_mm_can_dispatch(self) -> bool:
        """
        Check the availability of Grouped MM for a given model.
        """

        if not self._can_set_experts_implementation():
            raise ValueError(f"{self.__class__.__name__} does not support setting experts implementation.")

        # If no error raised by this point, we can return `True`
        return True

    def _flex_attn_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of Flex Attention for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        if not self._supports_flex_attn:
            raise ValueError(
                f"{self.__class__.__name__} does not support an attention implementation through torch's flex_attention."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/34809."
                " If you believe this error is a bug, please open an issue in Transformers GitHub repository"
                ' and load your model with the argument `attn_implementation="eager"` meanwhile.'
                ' Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )
        if not is_torch_flex_attn_available():
            raise ImportError("PyTorch Flex Attention is not supported on TPU.")

        # If no error raise by this point, we can return `True`
        return True

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        """
        Check that the `attn_implementation` exists and is supported by the models, and try to get the kernel from hub if
        it matches hf kernels pattern.

        Args:
            attn_implementation (`str` or `None`):
                The attention implementation to check for existence/validity.
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
            allow_all_kernels (`bool`, optional):
                Whether to load kernels from unverified hub repos, if `attn_implementation` is a custom kernel outside
                of the `kernels-community` hub repository.

        Returns:
            `str`: The final attention implementation to use, including potential fallbacks from sdpa to eager, or from
            None to sdpa (to potentially eager).
        """
        # Deprecation warning for paged| implementations
        if (
            attn_implementation is not None
            and attn_implementation.startswith("paged|")
            and attn_implementation != "paged|eager"
        ):
            warnings.warn(
                "The `paged|` prefix is no longer needed, except for `paged|eager`. Support for paged|sdpa and "
                "paged|*flash* implementations will be removed in v5.23. Please remove the `paged|` prefix.",
                FutureWarning,
                stacklevel=2,
            )
            attn_implementation = attn_implementation.removeprefix("paged|")

        # A kernel the model explicitly lists in `_compatible_flash_implementations` is vouched for by
        # the model author, so authorize loading it even when it lives outside the `kernels-community` org.
        if attn_implementation in (getattr(self, "_compatible_flash_implementations", None) or []):
            allow_all_kernels = True

        # Auto-correct model's default flash implementation if specified
        if attn_implementation is not None:
            compatible_flash_implementations = getattr(self, "_compatible_flash_implementations", None)
            if (
                is_flash_attention_requested(requested_attention_implementation=attn_implementation)
                and compatible_flash_implementations is not None
                and attn_implementation not in compatible_flash_implementations
            ):
                # Prefer the first implementation shipping builds for the current device
                device = get_device_type()
                supported_flash_implementations = [
                    impl
                    for impl in compatible_flash_implementations
                    if device in FLASH_ATTN_KERNEL_DEVICES.get(FLASH_ATTN_KERNEL_FALLBACK.get(impl, impl), (device,))
                ]
                default_flash_implementation = (supported_flash_implementations or compatible_flash_implementations)[0]

                logger.warning_once(
                    f"This model is compatible with the following flash attention implementations: `{compatible_flash_implementations}`. "
                    f"Automatically falling back to `{default_flash_implementation}` instead of `{attn_implementation}`."
                )
                attn_implementation = default_flash_implementation

        applicable_attn_implementation = attn_implementation

        requested_original_flash_attn = False
        if is_flash_attention_requested(requested_attention_implementation=attn_implementation):
            # If FA not installed, do not fail but use kernels instead if possible
            for fa_version in FLASH_ATTENTION_COMPATIBILITY_MATRIX.keys():
                # Check whether we have an original FA requested but not available in the env
                if (
                    attn_implementation == f"flash_attention_{fa_version}"
                    and not FLASH_ATTENTION_COMPATIBILITY_MATRIX[fa_version]["general_availability_check"]()
                ):
                    requested_original_flash_attn = True
                    break

        if (
            self._supports_flash_attn
            and requested_original_flash_attn
            and is_kernels_available()
            and not is_torch_npu_available()
        ):
            applicable_attn_implementation = FLASH_ATTN_KERNEL_FALLBACK[attn_implementation]

            if is_torch_xpu_available() and attn_implementation == "flash_attention_2":
                # On XPU, kernels library is the native implementation
                # Disabling this flag to avoid giving wrong fallbacks on errors and warnings
                requested_original_flash_attn = False

        if is_kernel(applicable_attn_implementation):
            try:
                # preload flash attention here to allow compile with fullgraph
                lazy_import_flash_attention(applicable_attn_implementation, allow_all_kernels=allow_all_kernels)

                # log that we used kernel fallback if successful
                if requested_original_flash_attn:
                    logger.warning_once(
                        f"You do not have `flash_attn` installed, using `{applicable_attn_implementation}` "
                        "from the `kernels` library instead!"
                    )
            except Exception as e:
                # raise the proper exception for requested flash attention
                if requested_original_flash_attn:
                    fa_version = int(attn_implementation[-1])  # "flash_attention_(2|3|...)"
                    self._flash_attn_can_dispatch(flash_attn_version=fa_version, is_init_check=is_init_check)

                # error properly out if a kernel was specifically requested
                raise e
        else:
            applicable_attn_implementation = self.get_correct_attn_implementation(
                applicable_attn_implementation, is_init_check
            )

            # preload flash attention here to allow compile with fullgraph
            if is_flash_attention_requested(requested_attention_implementation=applicable_attn_implementation):
                lazy_import_flash_attention(applicable_attn_implementation)

        return applicable_attn_implementation

    def _check_and_adjust_experts_implementation(self, experts_implementation: str | None) -> str:
        """
        Check that the `experts_implementation` exists and is supported by the models.

        Args:
            experts_implementation (`str` or `None`):
                The experts implementation to check for existence/validity.
        Returns:
            `str`: The final experts implementation to use.
        """
        applicable_experts_implementation = self.get_correct_experts_implementation(experts_implementation)
        return applicable_experts_implementation

    def get_correct_attn_implementation(self, requested_attention: str | None, is_init_check: bool = False) -> str:
        applicable_attention = "sdpa" if requested_attention is None else requested_attention
        if applicable_attention not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
            message = (
                f'Specified `attn_implementation="{applicable_attention}"` is not supported. The only possible arguments are '
                '`attn_implementation="eager"`, `"paged|eager"`'
            )
            # check `supports_flash_attn_2` for BC with custom code. TODO: remove after a few releases
            if self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False):
                message += ", "
                for fa_version in FLASH_ATTENTION_COMPATIBILITY_MATRIX.keys():
                    message += f'`"attn_implementation=flash_attention_{fa_version}"`, '
                message = message[:-2]  # remove trailing comma
            if self._supports_sdpa:
                message += ', `"attn_implementation=sdpa"`'
            if self._supports_flex_attn:
                message += ', `"attn_implementation=flex_attention"`'
            raise ValueError(message + ".")

        # Perform relevant checks
        if is_flash_attention_requested(requested_attention_implementation=applicable_attention) and (
            fa_matched := re.search(r"^flash_attention_(\d)$", applicable_attention)
        ):
            fa_version = int(fa_matched.group(1))  # last digit
            self._flash_attn_can_dispatch(flash_attn_version=fa_version, is_init_check=is_init_check)
        elif "flex_attention" in applicable_attention:
            self._flex_attn_can_dispatch(is_init_check)
        elif "sdpa" in applicable_attention:
            # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
            try:
                self._sdpa_can_dispatch(is_init_check)
            except (ValueError, ImportError) as e:
                if requested_attention is not None and "sdpa" in requested_attention:
                    raise e
                applicable_attention = "eager"

        return applicable_attention

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        applicable_experts = "grouped_mm" if requested_experts is None else requested_experts
        base_experts_fns = ["eager"] + list(set(ALL_EXPERTS_FUNCTIONS.keys()) | set(ALL_FP8_EXPERTS_FUNCTIONS.keys()))
        valid_experts_str_list = [f'`experts_implementation="{fn}"`' for fn in base_experts_fns]
        valid_experts_str_list[-1] = "and " + valid_experts_str_list[-1]
        valid_experts_str = ", ".join(valid_experts_str_list)
        if applicable_experts not in base_experts_fns:
            message = (
                f'Specified `experts_implementation="{applicable_experts}"` is not supported. The only possible arguments are '
                f"{valid_experts_str}."
            )
            raise ValueError(message)

        # Perform relevant checks
        if applicable_experts == "grouped_mm":
            try:
                self._grouped_mm_can_dispatch()
            except (ValueError, ImportError) as e:
                if requested_experts == "grouped_mm":
                    raise e
                applicable_experts = "eager"

        return applicable_experts

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        """Detect whether the class supports setting its attention implementation dynamically. Inspects the module
        source as a heuristic, which avoids maintaining yet another property flag. Instead, the flag is set dynamically
        on the first successful call.
        """
        # Early return if there is a cached value
        cached_value = getattr(cls, "_can_set_attn_implementation_cached_value", None)
        if isinstance(cached_value, bool):
            return cached_value

        class_module = sys.modules.get(cls.__module__)
        # Missing module entry (e.g. cleared by a test) or custom model in a jupyter notebook / repl -> do not allow to set it
        if class_module is None:
            return False
        try:
            code = inspect.getsource(class_module)
        except (OSError, TypeError):
            return False
        # Heuristic: if we find an `*Attention*(nn.Module)` class, check whether the interface is used
        if re.search(r"^class \w*Attention\w*\(nn\.Module\):", code, re.MULTILINE):
            can_set = "ALL_ATTENTION_FUNCTIONS.get_interface(" in code
        # If no attention layer, assume `True`. Most probably a multimodal model or inherits from existing models
        else:
            can_set = True
        # Successful read of source code -> cache the result
        cls._can_set_attn_implementation_cached_value = can_set
        return cls._can_set_attn_implementation_cached_value

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Detect whether the class supports setting its experts implementation dynamically. Inspects the module source
        as a heuristic, which avoids maintaining yet another property flag. Instead, the flag is set dynamically
        on the first successful call.
        """
        # Early return if there is a cached value
        cached_value = getattr(cls, "_can_set_experts_implementation_cached_value", None)
        if isinstance(cached_value, bool):
            return cached_value

        class_module = sys.modules.get(cls.__module__)
        # Missing module entry (e.g. cleared by a test) or custom model in a jupyter notebook / repl -> do not allow to set it
        if class_module is None:
            return False
        try:
            code = inspect.getsource(class_module)
        except (OSError, TypeError):
            return False
        # Heuristic: if the `@use_experts_implementation` decorator is used, then we can set it
        can_set = "@use_experts_implementation" in code
        cls._can_set_experts_implementation_cached_value = can_set
        return can_set

    def set_attn_implementation(self, attn_implementation: str | dict, allow_all_kernels: bool = False):
        """
        Set the requested `attn_implementation` for this model.

        Args:
            attn_implementation (`str` or `dict`):
                The attention implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value (to itself and all the submodels nested in it). A value can
                itself be a `dict` in the same format, to target the sub_configs of that submodel (as when initializing a
                config with a nested `attn_implementation` dict).
            allow_all_kernels (`bool`, optional):
                Whether to load kernels from unverified hub repos, if `attn_implementation` is a custom kernel outside
                of the `kernels-community` hub repository.
        """
        requested_implementation = (
            attn_implementation
            if not isinstance(attn_implementation, dict)
            else attn_implementation.get("", self.config._attn_implementation)
        )

        if requested_implementation != self.config._attn_implementation:
            # In this case, raise
            if not self._can_set_attn_implementation():
                logger.warning(
                    f"{self.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                    "does not follow the functional approach based on AttentionInterface "
                    "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                )
            else:
                requested_implementation = self._check_and_adjust_attn_implementation(
                    requested_implementation, is_init_check=False, allow_all_kernels=allow_all_kernels
                )
                # Apply the change (on the internal attr, to avoid setting it recursively)
                self.config._attn_implementation_internal = requested_implementation

        # Map each sub-config (at any depth, keyed by identity as configs are not hashable) to the implementation requested
        # for it and its dotted key. A value requested for a sub-config applies to its whole sub-tree, unless it is itself
        # a `dict` (same format as for the sub-config's own `sub_configs`, mirroring how `attn_implementation` is resolved
        # when initializing a config)
        subconfig_implementations = {}
        top_implementation = attn_implementation if isinstance(attn_implementation, dict) else requested_implementation
        subconfigs = deque([(self.config, None, top_implementation)])
        while subconfigs:
            config, key, implementation = subconfigs.popleft()
            for subconfig_key in config.sub_configs:
                if (subconfig := getattr(config, subconfig_key)) is not None:
                    sub_key = subconfig_key if key is None else f"{key}.{subconfig_key}"
                    sub_implementation = implementation
                    if isinstance(implementation, dict):
                        sub_implementation = implementation.get(subconfig_key, subconfig._attn_implementation)
                    own_implementation = sub_implementation
                    if isinstance(sub_implementation, dict):
                        own_implementation = sub_implementation.get("", subconfig._attn_implementation)
                    subconfig_implementations[id(subconfig)] = (subconfig, sub_key, own_implementation)
                    subconfigs.append((subconfig, sub_key, sub_implementation))

        # Apply it to all submodels as well. The top-level config was already handled above, and it may be shared by
        # submodels (e.g. ForCausalLM has a Model inside, which is the same "actual model"), so no need to check it again
        handled_config_ids = {id(self.config)}
        for submodule in self.modules():
            # We found a submodel with a config that was not handled yet (several submodels may share the same config)
            if isinstance(submodule, PreTrainedModel) and id(submodule.config) not in handled_config_ids:
                # In this case, warn and skip
                if not submodule._can_set_attn_implementation():
                    logger.warning(
                        f"{submodule.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                        "does not follow the functional approach based on AttentionInterface "
                        "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                    )
                # Set the attn on the submodule
                else:
                    # Default to the top-level implementation if the config is not a registered sub-config
                    _, _, sub_implementation = subconfig_implementations.get(
                        id(submodule.config), (None, None, requested_implementation)
                    )
                    # Check the module can use correctly, otherwise we raise an error if requested attention can't be set for submodule
                    sub_implementation = submodule.get_correct_attn_implementation(sub_implementation)
                    submodule.config._attn_implementation_internal = sub_implementation

                # Still add it as "changed" even if it was skipped, as we would otherwise try to set it in the dark afterwards
                handled_config_ids.add(id(submodule.config))

        # We need this as some old and badly designed models use subconfigs without declaring the corresponding modules as PreTrainedModel
        for subconfig_id, (subconfig, subconfig_key, sub_implementation) in subconfig_implementations.items():
            # This means we did not perform any check above for this particular subconfig -> set it in the dark if it is registered
            if (
                subconfig_id not in handled_config_ids
                # If it's already the same, then no need to enter here and raise warnings
                and sub_implementation != subconfig._attn_implementation
            ):
                if sub_implementation not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
                    raise ValueError(
                        f'Specified `attn_implementation="{sub_implementation}"` is not supported for {subconfig_key}. '
                        'The only possible arguments are "eager" (manual attention implementation)'
                        f"or one of the following: {list(ALL_ATTENTION_FUNCTIONS.valid_keys())}"
                    )
                subconfig._attn_implementation_internal = sub_implementation
                logger.warning(
                    f"We set the attention implementation for the sub-config `{subconfig_key}` to `{sub_implementation}` "
                    "without finding the associated sub-model. For this reason we could not check if the model supports it. "
                    "You may encounter undefined behavior."
                )

    def get_experts_implementation(self) -> dict[str, str | None]:
        """
        Return the experts implementation of this model and its submodels, as a `dict` in the form accepted by
        `set_experts_implementation` (`""` for this model, and one entry per sub_config). This is the counterpart of
        `set_experts_implementation`, e.g. to snapshot the current implementation and later restore it.
        """
        experts_implementation = {"": self.config._experts_implementation}
        for subconfig_key in self.config.sub_configs:
            subconfig = getattr(self.config, subconfig_key, None)
            if subconfig is not None:
                experts_implementation[subconfig_key] = subconfig._experts_implementation
        return experts_implementation

    def set_experts_implementation(self, experts_implementation: str | dict):
        """
        Set the requested `experts_implementation` for this model.

        Args:
            experts_implementation (`str` or `dict`):
                The experts implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value.
        """
        requested_implementation = (
            experts_implementation
            if not isinstance(experts_implementation, dict)
            else experts_implementation.get("", self.config._experts_implementation)
        )

        # MegaMoE is locked at load time: its TP plan is baked into `model.tp_plan` by `update_parallel_plan` (and isn't
        # re-evaluated) and `setup_megamoe_weights` mutates the expert weights into UTCCP layout on first forward.
        # Either side of a switch would silently produce garbage, so reject it with a clear pointer.
        current = self.config._experts_implementation
        if "deepgemm_megamoe" in (current, requested_implementation) and current != requested_implementation:
            raise RuntimeError(
                f"Cannot switch experts implementation from {current!r} to {requested_implementation!r} "
                "at runtime: `deepgemm_megamoe` is a load-time choice. Reload via "
                "`from_pretrained(..., experts_implementation=...)` to switch."
            )

        # Check the requested implementation is supported
        if requested_implementation != self.config._experts_implementation:
            requested_implementation = self._check_and_adjust_experts_implementation(requested_implementation)

            # Modify the implementation of the top level config
            if self._can_set_experts_implementation():
                # Apply the change (on the internal attr, to avoid setting it recursively)
                self.config._experts_implementation_internal = requested_implementation

        # Apply it to all submodels as well
        for submodule in self.modules():
            # We found a submodel (which is not self) with a different config (otherwise, it may be the same "actual model",
            # e.g. ForCausalLM has a Model inside, but no need to check it again)
            if (
                submodule is not self
                and isinstance(submodule, PreTrainedModel)
                and submodule.config.__class__ != self.config.__class__
                and submodule._can_set_experts_implementation()  # avoids bugs when text_model has MoEs but encoder no
            ):
                # Set the experts on the submodule
                sub_implementation = requested_implementation
                if isinstance(experts_implementation, dict):
                    for subconfig_key in self.config.sub_configs:
                        # We need to check for exact object match here, with `is`
                        if getattr(self.config, subconfig_key) is submodule.config:
                            sub_implementation = experts_implementation.get(
                                subconfig_key, submodule.config._experts_implementation
                            )
                            break
                # Check the module can use correctly, otherwise we raise an error if requested experts can't be set for submodule
                sub_implementation = submodule.get_correct_experts_implementation(sub_implementation)
                submodule.config._experts_implementation_internal = sub_implementation

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()
        found_embeddings = False

        for module in self.modules():
            if not (isinstance(module, PreTrainedModel) and hasattr(module, "get_input_embeddings")):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                continue

            if input_embeddings is None or not hasattr(input_embeddings, "register_forward_hook"):
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(input_embeddings.register_forward_hook(make_inputs_require_grads))
            found_embeddings = True

        self._require_grads_hooks = hooks
        if hooks:
            # for BC
            self._require_grads_hook = hooks[0]
        if not found_embeddings:
            logger.warning_once(
                f"{self.__class__.__name__} does not expose input embeddings. Gradients cannot flow back to the token "
                "embeddings when using adapters or gradient checkpointing. Override `get_input_embeddings` to fully "
                "support those features, or set `_input_embed_layer` to the attribute name that holds the embeddings."
            )

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        hooks = getattr(self, "_require_grads_hooks", None)
        if not hooks:
            return

        for hook in hooks:
            hook.remove()

        self._require_grads_hooks = []
        if hasattr(self, "_require_grads_hook"):
            del self._require_grads_hook

    def get_encoder(self, modality: str | None = None):
        """
        Best-effort lookup of the *encoder* module. If provided with `modality` argument,
        it looks for a modality-specific encoder in multimodal models (e.g. "image_encoder")
        By default the function returns model's text encoder if any, and otherwise returns `self`.

        Possible `modality` values are "image", "video" and "audio".
        """
        # NOTE: new models need to use existing names for layers if possible, so this list doesn't grow infinitely
        if modality in ["image", "video"]:
            possible_module_names = ["vision_tower", "visual", "vision_model", "vision_encoder", "image_tower"]
        elif modality == "audio":
            possible_module_names = ["audio_tower", "audio_encoder", "speech_encoder"]
        elif modality is None:
            possible_module_names = ["text_encoder", "encoder"]
        else:
            raise ValueError(f'Unnrecognized modality, has to be "image", "video" or "audio" but found {modality}')

        for name in possible_module_names:
            if hasattr(self, name):
                return getattr(self, name)

        if self.base_model is not self and hasattr(self.base_model, "get_encoder"):
            base_encoder = self.base_model.get_encoder(modality=modality)
            # Base model will always have attr `get_encoder` if inherited from `PreTrainedModel`
            # But it doesn't mean that the model has an encoder module, and we need to return `self`
            if base_encoder != self.base_model:
                return base_encoder

        # If this is a base transformer model (no encoder/model attributes), return self
        return self

    def set_encoder(self, encoder, modality: str | None = None):
        """
        Symmetric setter. Mirrors the lookup logic used in `get_encoder`.
        """

        # NOTE: new models need to use existing names for layers if possible, so this list doesn't grow infinitely
        if modality in ["image", "video"]:
            possible_module_names = ["vision_tower", "visual", "vision_model", "vision_encoder", "image_tower"]
        elif modality == "audio":
            possible_module_names = ["audio_tower", "audio_encoder"]
        elif modality is None:
            possible_module_names = ["text_encoder", "encoder"]
        else:
            raise ValueError(f'Unnrecognized modality, has to be "image", "video" or "audio" but found {modality}')

        for name in possible_module_names:
            if hasattr(self, name):
                setattr(self, name, encoder)
                return

        if self.base_model is not self:
            if hasattr(self.base_model, "set_encoder"):
                self.base_model.set_encoder(encoder, modality=modality)
            else:
                self.model = encoder

    def get_decoder(self):
        """
        Best-effort lookup of the *decoder* module.

        Order of attempts (covers ~85 % of current usages):

        1. `self.decoder/self.language_model/self.text_model`
        2. `self.base_model`                  (many wrappers store the decoder here)
        3. `self.base_model.get_decoder()`    (nested wrappers)
        4. fallback: raise for the few exotic models that need a bespoke rule
        """
        possible_module_names = ["language_model", "text_model", "decoder", "text_decoder"]
        for name in possible_module_names:
            if hasattr(self, name):
                return getattr(self, name)

        if self.base_model is not self and hasattr(self.base_model, "get_decoder"):
            return self.base_model.get_decoder()

        # If this is a base transformer model (no decoder/model attributes), return self
        # This handles cases like MistralModel which is itself the decoder
        return self

    def set_decoder(self, decoder):
        """
        Symmetric setter. Mirrors the lookup logic used in `get_decoder`.
        """

        possible_module_names = ["language_model", "text_model", "decoder"]
        for name in possible_module_names:
            if hasattr(self, name):
                setattr(self, name, decoder)
                return

        if self.base_model is not self:
            if hasattr(self.base_model, "set_decoder"):
                self.base_model.set_decoder(decoder)
            else:
                self.model = decoder

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.constant_(param, 0.0)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            # And the potential buffers for the BatchNorms
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)
        # This matches all the usual RotaryEmbeddings modules
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            # Default, vision axial and NTK-alpha rope are defined in modeling files, only one can be defined at a time!
            rope_init_fn_with_self = {
                "axial": getattr(module, "compute_axial_rope_parameters", None),
                "ntk_alpha": getattr(module, "compute_ntk_alpha_rope_parameters", None),
                "default": getattr(module, "compute_default_rope_parameters", None),
                **ROPE_INIT_FUNCTIONS,
            }
            rope_fn = rope_init_fn_with_self[module.rope_type]
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)

    def _initialize_weights(self, module, is_custom_code: bool = False):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        # This check is for remote code that does NOT use either `torch.init` or `transformers.initialization` in `_init_weights`
        # which allow to check the flag directly on param. As they don't and write the params in-place, params would be reinitialized
        # otherwise
        if (
            is_custom_code
            and all(getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False))
            and all(
                getattr(buffer, "_is_hf_initialized", False)
                for buffer in module.buffers(recurse=False)
                if buffer is not None
            )
        ):
            module._is_hf_initialized = True
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    def initialize_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.
        """
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(module: nn.Module, fn: Callable[[nn.Module, bool], None], is_custom_code: bool):
                for child in module.children():
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(child, PreTrainedModel):
                        smart_apply(child, child._initialize_weights, is_custom_code)
                    else:
                        smart_apply(child, fn, is_custom_code)
                fn(module, is_custom_code)
                return module

            setattr(torch.nn.Module, "smart_apply", smart_apply)

        # Let the magic happen with this simple call
        smart_apply_fn = getattr(self, "smart_apply")
        # `getattr(self, ...)` returns a bound method, so `self` is already provided as the receiver.
        smart_apply_fn(self._initialize_weights, True)

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        r"""
        Return the expanded tied weight keys (in case they contain modules or regex patterns) for only the current
        model, or recursively for all submodels if `all_submodels=True` (i.e. it will re-check the config values for all
        submodels).

        For almost all models, we only require to tie the embeddings, so the model has an internal property
        `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`. In this case, the mapping is already
        "expanded", i.e. it already contains full parameters, and this function will simply return a copy of the property.
        For more complex patterns, e.g. for `DFineForObjectDetection`, we have the following attribute
        ```
        _tied_weights_keys = {
            r"bbox_embed.(?![0])\d+": "bbox_embed.0",
            r"class_embed.(?![0])\d+": "class_embed.0",
            "model.decoder.class_embed": "class_embed",
            "model.decoder.bbox_embed": "bbox_embed",
        }
        ```
        In this case, the function looks up all the model's parameters and buffers, and matches all the params,
        returning the following:
        ```
        {
            'bbox_embed.1.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'bbox_embed.1.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            'bbox_embed.1.layers.1.bias': 'bbox_embed.0.layers.1.bias',
            'bbox_embed.1.layers.1.weight': 'bbox_embed.0.layers.1.weight',
            'bbox_embed.1.layers.2.bias': 'bbox_embed.0.layers.2.bias',
            'bbox_embed.1.layers.2.weight': 'bbox_embed.0.layers.2.weight',
            'bbox_embed.2.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'bbox_embed.2.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            ...
            'class_embed.1.bias': 'class_embed.0.bias',
            'class_embed.1.weight': 'class_embed.0.weight',
            'class_embed.2.bias': 'class_embed.0.bias',
            'class_embed.2.weight': 'class_embed.0.weight',
            ...
            'model.decoder.class_embed.0.bias': 'class_embed.0.bias',
            'model.decoder.class_embed.0.weight': 'class_embed.0.weight',
            'model.decoder.class_embed.1.bias': 'class_embed.0.bias',
            'model.decoder.class_embed.1.weight': 'class_embed.0.weight',
            ...
            'model.decoder.bbox_embed.0.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'model.decoder.bbox_embed.0.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            'model.decoder.bbox_embed.0.layers.1.bias': 'bbox_embed.0.layers.1.bias',
            'model.decoder.bbox_embed.0.layers.1.weight': 'bbox_embed.0.layers.1.weight',
            ...
        }
        ```
        i.e. all the parameters matching the regex and modules patterns in `_tied_weights_keys`
        """
        if all_submodels:
            expanded_tied_weights = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if isinstance(submodule, PreTrainedModel):
                    # Will dynamically check the config if it has changed
                    submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                    if prefix != "":
                        submodel_tied_weights = {
                            f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                        }
                    expanded_tied_weights.update(submodel_tied_weights)
            return expanded_tied_weights

        tied_mapping = self._tied_weights_keys
        # If the config does not specify any tying, return empty dict
        # NOTE: not all modules have `tie_word_embeddings` attr, for example vision-only
        # modules do not have any word embeddings!
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        if not tie_word_embeddings:
            return {}
        # If None, return empty dict
        elif tied_mapping is None:
            return {}
        # Short-cut for the most common cases: if the tied weights mapping only contains already expanded params,
        # return it directly (the regex matches names containing only letters, numbers, dots, and underscores to make
        # sure it does not contain a regex pattern, and finishing by "bias" or "weight" to make sure it's not a module)
        common_case_regex = re.compile(r"^[A-Za-z0-9_\.]+(weight)|(bias)$")
        if all(common_case_regex.match(k) for k in tied_mapping.keys() | tied_mapping.values()):
            return tied_mapping.copy()

        # We need to expand the regex patterns or the modules into proper parameters
        expanded_tied_weights = {}
        all_param_names = {k for k, _ in self.named_parameters(remove_duplicate=False)} | {
            k for k, _ in self.named_buffers(remove_duplicate=False)
        }
        for target_name, source_name in tied_mapping.items():
            target_name = "^" + target_name
            source_name = "^" + source_name

            source_params = sorted(filter(lambda x: re.search(source_name, x), all_param_names))
            target_params = sorted(filter(lambda x: re.search(target_name, x), all_param_names))
            if (
                not len(source_params) > 0
                or not len(target_params) > 0
                or len(target_params) % len(source_params) != 0
            ):
                raise ValueError(
                    f"There is an issue with your definition of `tie_weights_keys` for {source_name}:{target_name}. "
                    f"We found {source_params} to tie into {target_params}"
                )
            # we cycle source as it should be dispatch in many target if regex
            for target_n, source_n in zip(target_params, cycle(source_params)):
                # If the source is already registered as a target, use the original corresponding source. This should never
                # happen in general, but some models such as `d_fine` have complicated regex patterns, so it end up being
                # the case for simplicity of the regexes. Fix it silently here
                if source_n in expanded_tied_weights.keys():
                    # Use original source instead of having keys both as source and targets
                    expanded_tied_weights[target_n] = expanded_tied_weights[source_n]
                # Usual case, everything is already correct
                else:
                    expanded_tied_weights[target_n] = source_n

        return expanded_tied_weights

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        """
        Tie the model weights. If `recompute_mapping=False` (default when called internally), it will rely on the
        `model.all_tied_weights_keys` attribute, containing the `{target: source}` mapping for the tied params.
        If `recompute_mapping=True`, it will re-check all internal submodels and their config to determine the params
        that need to be tied. This is the default when `model.tie_weights()` is called on its own, outside of
        `__init__`, and `from_pretrained`, in case the config values were changed somewhere.

        Note that during `from_pretrained`, tying is *symmetric*: if the mapping says "tie target -> source" but
        `source` is missing in the checkpoint while `target` exists, we *swap* source and target so we can still
        tie everything to the parameter that actually exists.
        """
        # In this case, the keys stored in `all_tied_weights_keys` are already correct
        if not recompute_mapping:
            tied_keys = self.all_tied_weights_keys
        else:
            tied_keys = self.get_expanded_tied_weights_keys(all_submodels=True)

        tied_keys = list(tied_keys.items())
        for i, (target_param_name, source_param_name) in enumerate(tied_keys):
            # This is `from_pretrained` -> let's check symmetrically in case the source key is not present
            if missing_keys is not None:
                remove_from_missing = True
                source_is_there = source_param_name not in missing_keys
                target_is_there = target_param_name not in missing_keys
                # Both are already present -> it means the config is wrong and do not reflect the actual
                # checkpoint -> let's raise a warning and NOT tie them
                if source_is_there and target_is_there:
                    source_param = self.get_parameter(source_param_name)
                    target_param = self.get_parameter(target_param_name)

                    # Skip check if both are disk offloaded. Tied tensors always
                    # share the same offload device as per `infer_auto_device_map`
                    if source_param.device.type == "meta" and target_param.device.type == "meta":
                        continue

                    # If both are present, check if the weights are exactly similar, and only tie in this case
                    # This check is important, as torch `.bin` checkpoints always contain both keys, referencing the same storage
                    if not torch.equal(source_param, target_param):
                        logger.warning(
                            f"The tied weights mapping and config for this model specifies to tie {source_param_name} to "
                            f"{target_param_name}, but both are present in the checkpoints with different values, so we will NOT "
                            "tie them. You should update the config with `tie_word_embeddings=False` to silence this warning."
                        )
                        # Remove from internal attribute to correctly reflect actual tied weights
                        self.all_tied_weights_keys.pop(target_param_name)
                        # Skip to next iteration
                        continue
                # We're missing the source but we have the target -> we swap them, tying the parameter that exists
                elif not source_is_there and target_is_there:
                    target_param_name, source_param_name = source_param_name, target_param_name
                # Both are missing -> check other keys in case more than 2 keys are tied to the same weight
                elif not source_is_there and not target_is_there:
                    for target_backup, source_backup in tied_keys[i + 1 :]:
                        # In case of more than 2 keys tied to the same weight, they are guaranteed to all have
                        # the same source thanks to `get_expanded_tied_weights_keys` so this check is enough
                        if source_backup == source_param_name:
                            target_backup_is_there = target_backup not in missing_keys
                            # If the target is present, we found the correct weight to tie into (we know the source is missing)
                            # Note here that we do not tie the missing source right now as well, as it will be done anyway when
                            # the pair (target_backup, source_backup) becomes the main pair (target_param_name, source_param_name)
                            if target_backup_is_there:
                                source_param_name = target_backup
                                break
                    # If we did not break from the loop, it was impossible to find a source key -> let's raise
                    else:
                        # TODO Cyril: here ideally we want to raise instead of warning, but will break our CI as we have
                        # tests loading model from empty dicts to perform init checks - since we don't raise, add a flag
                        # to NOT remove from missing keys as it's actually still missing
                        remove_from_missing = False
                        logger.warning(
                            f"This checkpoint seem corrupted. The tied weights mapping for this model specifies to tie "
                            f"{source_param_name} to {target_param_name}, but both are absent from the checkpoint, "
                            "and we could not find another related tied weight for those keys"
                        )

            # Perform the actual tying
            source_param = self.get_parameter_or_buffer(source_param_name)
            if "." in target_param_name:
                parent_name, name = target_param_name.rsplit(".", 1)
                parent = self.get_submodule(parent_name)
            else:
                name = target_param_name
                parent = self
            # Tie the weights
            setattr(parent, name, source_param)
            self._adjust_bias(parent, source_param)
            # Remove from missing if necessary
            if missing_keys is not None and remove_from_missing:
                missing_keys.discard(target_param_name)

    def _adjust_bias(self, output_embeddings, input_embeddings):
        if getattr(output_embeddings, "bias", None) is not None and hasattr(output_embeddings, "weight"):
            weight_shape = output_embeddings.weight.shape
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, weight_shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically reusing the same old embeddings with new weight values, gathering is required
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model_embeds.weight, modifier_rank=None):
                vocab_size = model_embeds.weight.shape[0]
        else:
            vocab_size = model_embeds.weight.shape[0]

        # Update base model and current model config.
        self.config.get_text_config().vocab_size = vocab_size
        self.vocab_size = vocab_size

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing
        )
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None:
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            old_embeddings.num_embeddings = new_num_tokens  # maybe weights are tied which doesn't update attr
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_embeddings)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new embeddings  (in particular added tokens). The new embeddings will be initialized
            # from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters([old_embeddings.weight], modifier_rank=None):
                    self._init_added_embeddings_weights_with_mean(
                        old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
                    )
            else:
                self._init_added_embeddings_weights_with_mean(
                    old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
                )

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                old_embeddings.weight = new_embeddings.weight
                old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]

                # If the new number of tokens is smaller than the original `padding_idx`, the `padding_idx`
                # will be set to `None` in the resized embeddings.
                if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                    old_embeddings.padding_idx = None
        else:
            old_embeddings.weight.data = new_embeddings.weight.data
            old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
            if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: int | None = None,
        transposed: bool = False,
        mean_resizing: bool = True,
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """

        if new_num_tokens is None:
            return old_lm_head

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            old_lm_head.out_features = new_num_tokens  # maybe weights are tied which doesn't update attr
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_lm_head)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new lm_head weights (in particular added tokens). The new lm_head weights
            # will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                params = [old_lm_head.weight]
                if has_new_lm_head_bias:
                    params += [old_lm_head.bias]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                    self._init_added_lm_head_weights_with_mean(
                        old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                    )
                    if has_new_lm_head_bias:
                        self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

            else:
                self._init_added_lm_head_weights_with_mean(
                    old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                )
                if has_new_lm_head_bias:
                    self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        setattr(new_lm_head, "_is_hf_initialized", True)
        return new_lm_head

    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
    ):
        old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
        mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
        old_centered_embeddings = old_embeddings_weight - mean_embeddings
        covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

        # Check if the covariance is positive definite.
        epsilon = 1e-9
        is_covariance_psd = constraints.positive_definite.check(epsilon * covariance).all()
        if is_covariance_psd:
            # If covariances is positive definite, a distribution can be created. and we can sample new weights from it.
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embeddings, covariance_matrix=epsilon * covariance
            )
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = distribution.sample(
                sample_shape=(added_num_tokens,)
            ).to(old_embeddings.weight.dtype)
        else:
            # Otherwise, just initialize with the mean. because distribution will not be created.
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = (
                mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
            )

    def _init_added_lm_head_weights_with_mean(
        self,
        old_lm_head,
        new_lm_head,
        old_lm_head_dim,
        old_num_tokens,
        added_num_tokens,
        transposed: bool = False,
    ):
        if transposed:
            # Transpose to the desired shape for the function.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

        # The same initialization logic as Embeddings.
        self._init_added_embeddings_weights_with_mean(old_lm_head, new_lm_head, old_num_tokens, added_num_tokens)

        if transposed:
            # Transpose again to the correct shape.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

    def _init_added_lm_head_bias_with_mean(self, old_lm_head, new_lm_head, added_num_tokens):
        bias_mean = torch.mean(old_lm_head.bias.data, axis=0, dtype=torch.float32)
        bias_std = torch.std(old_lm_head.bias.data, axis=0).to(torch.float32)
        new_lm_head.bias.data[-1 * added_num_tokens :].normal_(mean=bias_mean, std=1e-9 * bias_std)

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> nn.Embedding | tuple[nn.Embedding]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        Initialize and tie the weights if needed. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # If we are initializing on meta device, there is no point in trying to run inits
        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            # Initialize weights
            self.initialize_weights()
        # Tie weights needs to be called here, but it can use the pre-computed `all_tied_weights_keys`
        self.tie_weights(recompute_mapping=False)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
    ):
        """
        Activates gradient checkpointing for the current model.

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            every_n_layers (`int`, *optional*, defaults to 1):
                Checkpoint only every `every_n_layers`-th decoder layer, leaving the rest to keep their activations.
                `1` checkpoints every layer, which is the usual all-or-nothing behavior. Larger values trade memory
                back for speed, which is worth it whenever the memory freed by full checkpointing is going unused.
            offload (`bool`, *optional*, defaults to `False`):
                Hold the activations saved for the recompute in pinned host memory rather than on the accelerator.
                This frees `layers x sequence x hidden` bytes of device memory, which is what dominates at long
                sequence lengths, and costs a device-to-host copy in the forward and a host-to-device copy in the
                backward. Both copies run on the compute stream, so the step gets slower.
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        if offload:
            # `get_current_accelerator()` is None when no accelerator is available, in which case the
            # activations already live on the host and there is nothing to copy off a device.
            device_type = (get_current_accelerator() or torch.device("cpu")).type

            def checkpoint_func(function, *args, **kwargs):
                with save_on_cpu(pin_memory=True, device_type=device_type):
                    return checkpoint(function, *args, **kwargs)
        else:
            checkpoint_func = checkpoint

        gradient_checkpointing_func = functools.partial(checkpoint_func, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=gradient_checkpointing_func,
                every_n_layers=every_n_layers,
            )
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        needs_embedding_grads = self.main_input_name == "input_ids"
        # we use that also to detect whether or not we have to raise if embeddings are missing (the submodel might not have embeddings at all)
        enable_input_grads = needs_embedding_grads or getattr(self, "_hf_peft_config_loaded", False)
        if enable_input_grads:
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable = checkpoint,
        every_n_layers: int = 1,
    ):
        # Imported here rather than at module scope: `modeling_layers` imports from this module.
        from .modeling_layers import GradientCheckpointingLayer

        is_gradient_checkpointing_set = False
        layer_index = 0

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                setattr(module, "_gradient_checkpointing_func", gradient_checkpointing_func)
                # Only the repeated per-layer blocks are counted, so `every_n_layers` means what it says even when
                # other modules also carry a `gradient_checkpointing` flag.
                if enable and isinstance(module, GradientCheckpointingLayer):
                    setattr(module, "gradient_checkpointing", layer_index % every_n_layers == 0)
                    layer_index += 1
                else:
                    setattr(module, "gradient_checkpointing", enable)
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warning(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        state_dict: dict | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "50GB",
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        save_original_format: bool = True,
        distributed_checkpoint: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            variant (`str`, *optional*):
                If specified, weights are saved in the format model.<variant>.safetensors.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be prepended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            save_original_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with the previous versions of `transformers` you can save the checkpoint with
                its reverse mapping. The reverse mapping needs to exists even if the model was loaded from a None legacy
                checkpoint.
            distributed_checkpoint (`bool`, *optional*, defaults to `False`):
                When saving an FSDP- or TP-sharded model, write safetensors with distributed checkpointing (DCP) instead of
                gathering weights to CPU first. Every rank must call this method.
                When `False`, weights are gathered to CPU on rank 0 via `gather_full_state_dict` before writing.
                It is only intended to save and resume training, set it to `False` if you want full `save_pretrained`
                features. With `push_to_hub=True`, rank 0 consolidates the checkpoint in a `consolidated/` subdirectory
                and uploads that directory.
                Requires `torch>=2.7`.
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None and isinstance(hf_quantizer, HfQuantizer) and hf_quantizer.is_serializable()
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        # we need to check against tp_size, not tp_plan, as tp_plan is substituted to the class one
        if self._tp_size is not None and not is_huggingface_hub_greater_or_equal("0.31.4"):
            raise ImportError(
                "Saving a model with tensor parallelism requires `huggingface_hub` version 0.31.4 or higher."
            )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        save_directory_path = os.fspath(save_directory)

        save_on_this_rank = self.should_save_on_this_rank(is_main_process)
        hub_kwargs = {}
        if push_to_hub and save_on_this_rank:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory_path.split(os.path.sep)[-1])
            create_pr = kwargs.pop("create_pr", False)
            repo_id = hf_api().create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)
            hub_kwargs = {
                "repo_id": repo_id,
                "commit_message": commit_message,
                "create_pr": create_pr,
            }

        metadata = {}
        if hf_quantizer is not None:
            state_dict, metadata = hf_quantizer.get_state_dict_and_metadata(self)
        metadata["format"] = "pt"

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)
        distributed_config = getattr(self.config, "distributed_config", None)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = model_to_save.dtype
        model_to_save.config.dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        # When using FSDP2, unwrapping is a noop, so the model name doesn't change back to the original model name
        model_to_save.config.architectures = [model_to_save.__class__.__name__.removeprefix("FSDP")]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if save_on_this_rank and self.is_remote_code():
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if save_on_this_rank:
            if not _hf_peft_config_loaded:
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict(state_dict=state_dict)

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be prepended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        if distributed_checkpoint:
            self.save_distributed_checkpoint(
                model_to_save,
                save_directory,
                push_to_hub=push_to_hub,
                save_on_this_rank=save_on_this_rank,
                token=token,
                **hub_kwargs,
            )
            return

        # Get the model state_dict
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # if any model parameters are offloaded, we need to know it for later
        is_offloaded = False
        if hasattr(self, "hf_device_map") and (
            "cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values()
        ):
            is_offloaded = True
            warnings.warn(
                "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory "
                "exceeds the `shard_size` (50GB default)"
            )

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None and len(self._keys_to_ignore_on_save) > 0:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict:
                    del state_dict[ignore_key]

        # If model was sharded with TP/FSDP, gather full tensors for saving
        state_dict = self.gather_sharded_state_dict_for_save(
            model_to_save,
            state_dict,
            distributed_config,
            save_on_this_rank=save_on_this_rank,
        )

        # Remove tied weights as safetensors do not handle them
        state_dict = remove_tied_weights_from_state_dict(state_dict, model_to_save)

        # Revert all renaming and/or weight operations. In general, due to potential many-weights-to-one conversion patterns,
        # we need to revert the whole state_dict at once to make sure all weights are available. For offloaded models though,
        # some weights are on meta device, so we need to first load them back into cpu then convert (and we do it later inside a
        # given shard, to avoid blowing cpu memory since offloading means constrained resources)
        if save_original_format and not is_offloaded and not _hf_peft_config_loaded:
            state_dict = revert_weight_conversion(model_to_save, state_dict)

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors
                and save_on_this_rank
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # The weight_map may change compared to state_dict_split.tensor_to_filename due to revert weight conversions
        weight_map = None
        if state_dict_split.is_sharded:
            # For offloaded weights, we will convert each shard later, so the weight names will change and we will fill
            # the weight_map as we get them
            weight_map = (
                state_dict_split.tensor_to_filename
                if not (is_offloaded and save_original_format and not _hf_peft_config_loaded)
                else {}
            )

        # Save the model
        if save_on_this_rank:
            for shard_file, tensor_names in logging.tqdm(
                state_dict_split.filename_to_tensors.items(), desc="Writing model shards"
            ):
                filename = os.path.join(save_directory, shard_file)
                shard_state_dict = {}
                for tensor_name in tensor_names:
                    # Get the tensor, and remove it from state_dict to avoid keeping the ref
                    tensor = state_dict.pop(tensor_name)

                    # If the param was offloaded, we need to load it back from disk to resave it. It's a strange pattern,
                    # but it would otherwise not be contained in the saved shard if we were to simply move the file
                    # or something
                    if is_offloaded and tensor.device.type == "meta":
                        tensor = load_offloaded_parameter(model_to_save, tensor_name)

                    # only do contiguous after it's permuted correctly in case of TP
                    shard_state_dict[tensor_name] = tensor.contiguous()

                # As explained above, for offloaded scenarios, weight format could not be reverted before due to meta weights,
                # so do it now after they were loaded onto cpu. For one-weight-to-many operations, it may be an issue, but usually the shards
                # contain all the necessary params, except if we are quite unlucky on the sharding. The failure surface is (very few models
                # with one-weight-to-many + offloading to disk + unlucky sharding), so it will almost never happen
                if is_offloaded and save_original_format and not _hf_peft_config_loaded:
                    try:
                        shard_state_dict = revert_weight_conversion(model_to_save, shard_state_dict)
                        # Save the weight_map, since some names etc may have changed due to conversion compared to initial `state_dict_split`
                        if state_dict_split.is_sharded:
                            weight_map.update({k: os.path.basename(shard_file) for k in shard_state_dict.keys()})  # ty: ignore[unresolved-attribute]
                    except Exception:
                        raise RuntimeError(
                            "We could not revert some weight conversions because of offlading, and several weights needed for a single "
                            "conversion operation living in different shard files. Try reducing `max_shard_size` a bit, or worst case "
                            "set `save_original_format=False`."
                        )

                # TODO: it would be very nice to do the writing concurrently, but safetensors never releases the GIL,
                # so it's not possible for now....
                # Write the shard to disk
                safe_save_file(shard_state_dict, filename, metadata=metadata)
                # Cleanup the data before next loop (important with offloading, so we don't blowup cpu RAM)
                del shard_state_dict

            index = None
            if state_dict_split.is_sharded:
                index = {
                    "metadata": {"total_parameters": self.num_parameters(), **state_dict_split.metadata},
                    "weight_map": weight_map,
                }

            if index is None:
                path_to_weights = os.path.join(save_directory, weights_name)
                logger.info(f"Model weights saved in {path_to_weights}")
            else:
                save_index_file = SAFE_WEIGHTS_INDEX_NAME
                save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
                logger.info(
                    f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )

        if push_to_hub and save_on_this_rank:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(repo_id, self.model_tags, token=token)

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

        self.barrier_after_gathered_checkpoint_save(distributed_config)

    def push_to_hub(self, *args, **kwargs):
        tags = self.model_tags if self.model_tags is not None else []

        tags_kwargs = kwargs.get("tags", [])
        if isinstance(tags_kwargs, str):
            tags_kwargs = [tags_kwargs]

        for tag in tags_kwargs:
            if tag not in tags:
                tags.append(tag)

        if tags:
            kwargs["tags"] = tags
        return super().push_to_hub(*args, **kwargs)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum(param.nelement() * param.element_size() for param in self.parameters())
        if return_buffers:
            mem_bufs = sum(buf.nelement() * buf.element_size() for buf in self.buffers())
            mem = mem + mem_bufs
        return mem

    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute,
            # it's necessary to manually call the `cuda` method on HQQLinear layers.
            super().cuda(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if len(args) > 0:
                        device = args[0]
                    else:
                        device = kwargs.get("device", "cuda")
                    module.cuda(device)
            return self

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
        return super().cuda(*args, **kwargs)

    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute, we must
            # explicitly move the parameters to the target device for each HQQLinear layer after `to`.
            super().to(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if "device" in kwargs:
                        device = kwargs["device"]
                    else:
                        device = args[0]
                    if "dtype" in kwargs:
                        dtype = kwargs["dtype"]
                    elif dtype_present_in_args:
                        dtype = arg
                    else:
                        dtype = None
                    # Due to the current messy implementation of HQQLinear, updating `compute_dtype`
                    # followed by calling the `cuda` method achieves the intended behavior of `to`,
                    # even when the target device is CPU.
                    if dtype is not None:
                        module.compute_dtype = dtype
                    module.cuda(device)
            return self

        if dtype_present_in_args and getattr(self, "quantization_method", None) == QuantizationMethod.QUARK:
            raise ValueError("Casting a Quark quantized model to a new `dtype` is not supported.")

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False) and not is_bitsandbytes_available("0.48"):
                raise ValueError(
                    "You need to install `pip install bitsandbytes>=0.48.0` if you want to move a 8-bit model across devices using to()."
                )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def get_init_context(
        cls, dtype: torch.dtype, is_quantized: bool, _is_ds_init_called: bool, allow_all_kernels: bool | None
    ):
        # Need to instantiate with correct dtype
        init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
        # Needed as we cannot forward the `allow_all_kernels` arg in the model's __init__
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # We cannot initialize the model on meta device with deepspeed when not quantized
            if not is_quantized and not _is_ds_init_called:
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                init_contexts.extend(
                    [
                        init.no_init_weights(),
                        deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                        set_zero3_state(),
                    ]
                )
            elif is_quantized:
                init_contexts.extend([torch.device("meta"), set_quantized_state()])
        else:
            # meta_device_safe_creation_ops patches torch.linspace to default to CPU
            # so that custom models calling .item() during __init__ (e.g. drop-path
            # schedules) don't crash on meta tensors.
            init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])

        return init_contexts

    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        """Create the dtype_plan describing modules/parameters that should use the `keep_in_fp32` flag."""
        dtype_plan = {}

        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay in bf16 in fp16
        # See https://github.com/huggingface/transformers/issues/20287 for details.
        if self._keep_in_fp32_modules is not None and dtype == torch.float16:
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules, torch.float32))

        # The _keep_in_fp32_modules_strict was introduced to always force upcast to fp32, for both fp16 and bf16
        if self._keep_in_fp32_modules_strict is not None and dtype in (torch.float16, torch.bfloat16):
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules_strict, torch.float32))

        return dtype_plan

    def set_use_kernels(self, use_kernels, kernel_config: KernelConfig | None = None, mode: "Mode | None" = None):
        """
        Set whether or not to use the `kernels` library to kernelize some layers of the model.
        Args:
            use_kernels (`bool`):
                Whether or not to use the `kernels` library to kernelize some layers of the model.
            kernel_config (`KernelConfig`, *optional*):
                The kernel configuration to use to kernelize the model. If `None`, the default kernel mapping will be used.
            mode (`Mode`, *optional*):
                The mode that should be applied during `kernelize`. Optional, defaults to either training or inference mode
                based on the internal `training` flag.
        """
        if use_kernels:
            if not is_kernels_available():
                raise ValueError(
                    "Kernels are not available. "
                    f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
                    f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
                )
            from .integrations.hub_kernels import register_kernel_mapping_transformers

            register_kernel_mapping_transformers()

            if kernel_config is not None:
                if not isinstance(kernel_config, KernelConfig):
                    raise ValueError(
                        f"Expected `kernel_config` to be of type `KernelConfig` but got {type(kernel_config)}"
                    )

                # Since kernel_config is a correct value, set it as an attribute of the model so it can be used.
                self.kernel_config = kernel_config

                # This will make sure the mapping is valid, and the layers are registered in the model
                kernel_config.sanitize_kernel_mapping(self)

                # This will create a compatible mapping for the model with the kernels library
                kernel_config.create_compatible_mapping(self)

            kernelize(self, mode=mode)
        else:
            self._use_kernels = False

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PreTrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        fusion_config: dict[str, bool | dict[str, Any]] | None = None,
        disable_mmap: bool | None = None,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`. Directories saved with
                      `save_pretrained(..., distributed_checkpoint=True)` are detected and loaded with distributed
                      checkpointing (DCP), preserving the model's current mesh and placements. This requires
                      `torch>=2.7` and loads strictly: every model key must be present in the checkpoint.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PreTrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PreTrainedConfig`],
                    - a string or path valid as input to [`~PreTrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of
                    - `"eager"` (manual implementation of the attention)
                    - `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html))
                    - `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))
                    - `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper))
                    - `"flash_attention_4"` (using [Dao-AILab/flash-attention/flash_attn/cute](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)).
                By default, if available, SDPA will be used. The default is otherwise the manual `"eager"` implementation.

                Accept HF kernel references in the form:
                  <namespace>/<repo_name>[@<revision>][:<kernel_name>]

                - <namespace> and <repo_name> are any non-"/" and non-":" sequences.
                - "@<revision>" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123".
                - ":<kernel_name>" is optional and selects a function inside the kernel repo.
                - Both options can appear together and in this order only: @revision first, then :kernel_name.
                - We intentionally allow a leading "<wrapper>|" prefix (e.g., "flash|...") because the code
                  strips it before loading; '|' is not excluded in the character classes here.

                Examples that match:
                  "org/model"
                  "org/model@main"
                  "org/model:custom_kernel"
                  "org/model@v1.2.3:custom_kernel"
            experts_implementation (`str`, *optional*):
                The experts implementation to use in the model (if relevant). Can be any of:

                - `"eager"` (sequential implementation of the experts matrix multiplications).
                - `"batched_mm"` (using [`torch.bmm`](https://pytorch.org/docs/stable/generated/torch.bmm.html)).
                - `"grouped_mm"` (using [`torch.nn.functional.grouped_mm`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html)).

                By default, if the model supports it, `"grouped_mm"` will be used. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*, defaults to `"auto"`):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `dtype` or `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            distributed_config ([`~transformers.distributed.configuration_utils.DistributedConfig`], *optional*):
                Configuration for native distributed loading with tensor parallelism or FSDP2. Pass
                `DistributedConfig(tp_size=N)` to use a model's predefined tensor parallel plan,
                `DistributedConfig(tp_plan=...)` to specify a tensor parallel plan, or
                `DistributedConfig(fsdp_size=N)` for FSDP2. Requires `torchrun` and an initialized
                process group when `tp_size > 1` or `fsdp_size > 1`. Mutually exclusive with `device_map`.
            device_mesh (`torch.distributed.DeviceMesh`, *optional*):
                A torch device mesh. If not provided would default to world size. Used only for tensor parallel for now.
                If provided, it has to contain dimension named `"tp"` in case it's > 1 dimensional, this dimension will be used for tensor parallelism
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq).
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            disable_mmap (`bool`, *optional*):
                Whether to disable memory mapping when loading safetensors checkpoints. When `None` (default),
                it is auto-detected to `True` when the checkpoint lives on an `hf-mount` FUSE filesystem
                (used by HF Spaces/Endpoints), where mmap + parallel page-faults can deadlock. When `True`,
                files are read fully into memory and parsed with `safetensors.torch.load`. When `False`, the
                default memory-mapped loader is always used.
            fusion_config (`dict[str, bool | dict[str, Any]]`, *optional*):
                Optional fusion configuration applied before model instantiation. Each key enables a fusion family and
                its value can either be `True` to enable that fusion with default options or a dictionary of
                family-specific options. For example, `{"patch_embeddings": True}` enables patch embedding fusion.
                This should only be used as an inference optimization, as it can slightly change outputs. If omitted,
                `from_pretrained()` falls back to `config.fusion_config` when available. Refer to the fusion mapping
                guide in `docs/source/en/fusion_mapping.md` for more details.
            key_mapping (`dict[str, str], *optional*):
                A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
                architecture, but was not converted accordingly.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PreTrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        ```
        """
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        tqdm_class = kwargs.pop("tqdm_class", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        kwargs.pop("_commit_hash", None)  # BC: not used anymore, `revision` is resolved to a commit hash instead
        variant = kwargs.pop("variant", None)
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        distributed_config: DistributedConfig = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        allow_all_kernels = kwargs.pop("allow_all_kernels", False)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        key_mapping = kwargs.pop("key_mapping", None)

        # Not used anymore -- remove them from the kwargs
        for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
            _ = kwargs.pop(name, None)

        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        # Resolve the revision once and for all: config, weights, generation config and adapters are then all loaded
        # from the exact same repository state, without any further call to the Hub to revalidate a mutable revision.
        requested_revision = revision
        revision = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }

        if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
            raise ValueError(
                "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
            )

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        has_standalone_tp_args = tp_plan is not None or tp_size is not None

        if distributed_config is not None and has_standalone_tp_args:
            raise ValueError(
                "Pass either `distributed_config` or the standalone `tp_plan`/`tp_size` arguments, not both. "
                "Set tensor-parallel options on `DistributedConfig` when using it."
            )

        if tp_plan is not None:
            warnings.warn(
                "Passing `tp_plan` directly to `from_pretrained` is deprecated and will be removed in v5.18. "
                "Pass it in `distributed_config=DistributedConfig(tp_plan=...)` instead.",
                FutureWarning,
                stacklevel=2,
            )

        if has_standalone_tp_args:
            # For backwards compatibility, we still support passing `tp_plan` and `tp_size` directly to `from_pretrained`.
            distributed_config = DistributedConfig(tp_plan=tp_plan, tp_size=tp_size)

        if distributed_config is not None:
            distributed_config, device_map, mesh_manager = cls.prepare_distribute_model(
                distributed_config, device_map=device_map
            )
            device_mesh = mesh_manager.get_mesh(("pp", "fsdp", "tp")) if mesh_manager is not None else None

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if adapter_kwargs is None:
            adapter_kwargs = {}

        adapter_repo_id = pretrained_model_name_or_path
        _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs = maybe_load_adapters(
            pretrained_model_name_or_path,
            download_kwargs,
            **adapter_kwargs,
        )
        if pretrained_model_name_or_path != adapter_repo_id:
            # We were pointed at an adapter, and now load the base model it refers to: the revision we resolved
            # above belongs to the adapter repository, so resolve the base model's own.
            revision = resolve_revision(
                pretrained_model_name_or_path,
                requested_revision,
                token=token,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            download_kwargs["revision"] = revision
        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config_class = cls.config_class
            if config_class is None:
                raise ValueError(
                    f"{cls.__name__} does not define `config_class`; pass an explicit config to `from_pretrained`."
                )
            config, model_kwargs = config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **download_kwargs,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        if distributed_config is not None:
            config.distributed_config = distributed_config

        # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
        # to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, device_map, weights_only, user_agent, gguf_file=gguf_file
        )

        if kernel_config is not None and not use_kernels:
            logger.warning_once(
                "A kernel_config was provided but use_kernels is False; setting use_kernels=True automatically. To suppress this warning, explicitly set use_kernels to True."
            )
            use_kernels = True

        distributed_checkpoint_dir = get_distributed_checkpoint_dir(
            pretrained_model_name_or_path, subfolder, gguf_file, state_dict
        )
        if distributed_checkpoint_dir is not None:
            checkpoint_files = _get_resolved_distributed_checkpoint_files(distributed_checkpoint_dir, hf_quantizer)
            sharded_metadata = None
        else:
            checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                variant=variant,
                gguf_file=gguf_file,
                use_safetensors=use_safetensors,
                download_kwargs=download_kwargs,
                user_agent=user_agent,
                is_remote_code=cls.is_remote_code(),
                transformers_explicit_filename=getattr(config, "transformers_weights", None),
                tqdm_class=tqdm_class,
            )

        is_quantized = hf_quantizer is not None

        if gguf_file:
            # Read before the dtype is settled: a GGUF's own float type is what `dtype="auto"` resolves to.
            hf_quantizer.read_header(checkpoint_files[0])

        # Find the correct dtype based on current state
        config, dtype = _get_dtype(
            dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only, hf_quantizer
        )

        config.name_or_path = pretrained_model_name_or_path

        # Overwrite `config.fusion_config` if it is provided.
        if fusion_config is not None:
            config.fusion_config = copy.deepcopy(fusion_config)

        # Register fusion patches
        fusion_config = getattr(config, "fusion_config", None)
        if fusion_config is not None:
            from .fusion_mapping import register_fusion_patches

            register_fusion_patches(cls, config, fusion_config)

        # Kernel patches: single-layer replacement (stateful __init__) then fusions.
        if kernel_config is not None and use_kernels:
            from .integrations.hub_kernels import register_kernel_replacements_and_fusions

            # For remote kernels, we need to apply the context manager
            allow_all_kernels_context = [allow_all_hub_kernels()] if allow_all_kernels else []
            with ContextManagers(allow_all_kernels_context):
                register_kernel_replacements_and_fusions(cls, config, kernel_config)

        model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            model = cls(config, *model_args, **model_kwargs)
            patch_output_recorders(model)

            if hf_quantizer is not None:  # replace module with quantized modules (does not touch weights)
                hf_quantizer.preprocess_model(
                    model=model,
                    dtype=dtype,
                    device_map=device_map,
                    checkpoint_files=checkpoint_files,
                    use_kernels=use_kernels,
                )

        if gguf_file:
            state_dict = hf_quantizer.get_state_dict(checkpoint_files[0], model)

        # Create the dtype_plan to potentially use the `keep_in_fp32` flags (this needs to be called on the already
        # instantiated model, as the flags can be modified by instances sometimes)
        dtype_plan = model._get_dtype_plan(dtype)

        # Obtain the weight conversion mapping for this model if any are registered and apply to all submodels recursively
        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)

        if distributed_config is not None:
            model = cls.maybe_distribute_model(model, distributed_config, mesh_manager)

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)

        # Finalize model weight initialization
        load_config = LoadStateDictConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            device_map=device_map,
            disk_offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=dtype,
            dtype_plan=dtype_plan,
            hf_quantizer=hf_quantizer,
            device_mesh=device_mesh,
            weights_only=weights_only,
            weight_mapping=weight_conversions,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs,
            disable_mmap=disable_mmap,
        )
        if distributed_checkpoint_dir is not None:
            loading_info = model._load_and_finalize_distributed_pretrained_model(
                distributed_checkpoint_dir, device_map, device_mesh
            )
            disk_offload_index = None
        else:
            loading_info, disk_offload_index = cls._load_pretrained_model(
                model, state_dict, checkpoint_files, load_config
            )
            loading_info = cls._finalize_model_loading(model, load_config, loading_info)
        model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        model.set_use_kernels(use_kernels, kernel_config)

        # If it is a model with generation capabilities, attempt to load generation files (generation config,
        # custom generate function)
        if model.can_generate() and hasattr(model, "adjust_generation_fn") and not gguf_file:
            model.adjust_generation_fn(
                generation_config,
                from_auto_class,
                from_pipeline,
                pretrained_model_name_or_path,
                **download_kwargs,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # If the device_map has more than 1 device or disk offloading: dispatch model with hooks
        if device_map is not None and (len(set(device_map.values())) > 1 or "disk" in set(device_map.values())):
            accelerate_dispatch(model, hf_quantizer, device_map, offload_folder, disk_offload_index, offload_buffers)

        if hf_quantizer is not None:
            model.hf_quantizer = hf_quantizer
            hf_quantizer.postprocess_model(
                model
            )  # usually a no-op but sometimes needed, e.g to remove the quant config when dequantizing

        if _adapter_model_path is not None:
            if token is not None:
                adapter_kwargs["token"] = token
            loading_info = model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                load_config=load_config,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            return model, loading_info.to_dict()
        return model

    @staticmethod
    def _load_pretrained_model(
        model: "PreTrainedModel",
        state_dict: dict | None,
        checkpoint_files: list[str] | None,
        load_config: LoadStateDictConfig,
        expected_keys: list[str] | None = None,
    ) -> tuple[LoadStateDictInfo, dict]:
        """Perform the actual loading of some checkpoints into a `model`, by reading them from disk and dispatching them accordingly."""
        hf_quantizer = load_config.hf_quantizer
        is_quantized = load_config.is_quantized
        is_hqq_or_quark = hf_quantizer is not None and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
        }

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

        if logger.level >= logging.WARNING:
            verify_tp_plan(expected_keys, getattr(model, "_tp_plan", None))

        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        # Prepare parameters offloading if needed
        if load_config.device_map is not None and "disk" in load_config.device_map.values():
            disk_offload_index = accelerate_disk_offload(
                model,
                load_config.disk_offload_folder,
                checkpoint_files,
                load_config.device_map,
                load_config.sharded_metadata,
                load_config.weight_mapping,
            )

        # Warmup cuda to load the weights much faster on devices
        if load_config.device_map is not None and not is_hqq_or_quark:
            expanded_device_map = expand_device_map(load_config.device_map, expected_keys)
            caching_allocator_warmup(model, expanded_device_map, load_config.hf_quantizer)

        error_msgs = []

        if is_deepspeed_zero3_enabled() and not is_quantized:
            if state_dict is None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(
                        load_state_dict(
                            ckpt_file,
                            map_location="cpu",
                            weights_only=load_config.weights_only,
                            disable_mmap=load_config.disable_mmap,
                        )
                    )
                state_dict = merged_state_dict
            error_msgs, missing_keys = _load_state_dict_into_zero3_model(model, state_dict, load_config)
            # This is not true but for now we assume only best-case scenario with deepspeed, i.e. perfectly matching checkpoints
            loading_info = LoadStateDictInfo(
                missing_keys=missing_keys,
                error_msgs=error_msgs,
                unexpected_keys=set(),
                mismatched_keys=set(),
                conversion_errors={},
                skipped_pp_keys=set(),
            )
        else:
            all_pointer = set()
            if state_dict is not None:
                merged_state_dict = state_dict
            elif checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors") and state_dict is None:
                merged_state_dict = {}
                for file in checkpoint_files:
                    if load_config.disable_mmap or _is_on_hf_mount(file):
                        with open(file, "rb") as _fh:
                            merged_state_dict.update(_safe_load_bytes(_fh.read()))
                        continue
                    is_mps = load_config.device_map is not None and any(
                        (d.type if isinstance(d, torch.device) else d) == "mps"
                        for d in load_config.device_map.values()
                    )
                    # Use pread on MPS (mmap incompatible) and Windows (mmap reserves
                    # copy-on-write commit charge for the entire file, exhausting memory
                    # for large multi-shard checkpoints).
                    if is_mps:
                        backend, device = "pread", "mps"
                    elif sys.platform == "win32":
                        backend, device = "pread", "cpu"
                    else:
                        backend, device = "mmap", "cpu"
                    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
                    all_pointer.add(file_pointer)
                    for k in file_pointer.keys():
                        merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet
            # Checkpoints are .bin
            elif checkpoint_files is not None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(load_state_dict(ckpt_file, disable_mmap=load_config.disable_mmap))
            else:
                raise ValueError("Neither a state dict nor checkpoint files were found.")

            loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=merged_state_dict,
                load_config=load_config,
                disk_offload_index=disk_offload_index,
            )

            # finally close all opened file pointers
            for k in all_pointer:
                k.__exit__(None, None, None)

        return loading_info, disk_offload_index

    @staticmethod
    def _finalize_model_loading(
        model, load_config: LoadStateDictConfig, loading_info: LoadStateDictInfo
    ) -> LoadStateDictInfo:
        """Perform all post processing operations after having loaded some checkpoints into a model, such as moving
        missing keys from meta device to their expected device, reinitializing missing weights according to proper
        distributions, tying the weights and logging the loading report."""
        try:
            # Marks tied weights as `_is_hf_initialized` to avoid initializing them (it's very important for efficiency)
            model.mark_tied_weights_as_initialized(loading_info)

            # Move missing (and potentially mismatched) keys and non-persistent buffers back to their expected device from
            # meta device (because they were not moved when loading the weights as they were not in the loaded state dict)
            model._move_missing_keys_from_meta_to_device(
                loading_info.missing_and_mismatched(),
                load_config.device_map,
                load_config.device_mesh,
                load_config.hf_quantizer,
            )

            # Correctly initialize the missing (and potentially mismatched) keys (all parameters without the `_is_hf_initialized` flag)
            model._initialize_missing_keys(load_config.is_quantized)

            # Tie the weights
            model.tie_weights(missing_keys=loading_info.missing_keys, recompute_mapping=False)

            # Adjust missing and unexpected keys
            model._adjust_missing_and_unexpected_keys(loading_info)
        finally:
            log_state_dict_report(
                model=model,
                pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
                ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
                loading_info=loading_info,
                logger=logger,
            )

        return loading_info

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name.removeprefix(_prefix)
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_tracing(input_ids):
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            # NOTE: `sep_token_id` is not used in all models and it can be absent in the config
            sep_token_id = getattr(self.config, "sep_token_id", None)
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (sep_token_id is not None and sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)

    @property
    def supports_tp_plan(self):
        """
        Returns whether the model has a tensor parallelism plan.
        """
        # Check if model has a TP plan
        if self._tp_plan:
            return True
        # Check if base model has a TP plan
        if self.base_model._tp_plan:
            return True
        # Check if config has TP plan
        if self.config.base_model_tp_plan:
            return True
        return False

    @property
    def tp_size(self):
        """
        Returns the model's tensor parallelism degree.
        """
        # if None, the model didn't undergo tensor parallel sharding
        return self._tp_size

    @property
    def fsdp_size(self):
        """
        Returns the model's FSDP sharding degree.
        """
        # if None, the model didn't undergo FSDP sharding
        return self._fsdp_size

    @property
    def supports_pp_plan(self):
        # Check if model has a PP plan
        if self._pp_plan:
            return True
        # Check if base model has PP plan
        if self.base_model._pp_plan:
            return True
        # Check if config has PP plan
        if self.config.base_model_pp_plan:
            return True
        return False

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            logger.warning_once(
                f"`loss_type={loss_type}` was set in the config but it is unrecognized. "
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def use_kernels(self) -> bool:
        return getattr(self, "_use_kernels", False)

    @use_kernels.setter
    def use_kernels(self, value: bool) -> None:
        # Avoid re-kernelizing if already enabled
        if bool(value) and getattr(self, "_use_kernels", False):
            return

        if value:
            self.set_use_kernels(True)
        else:
            if getattr(self, "_use_kernels", False):
                logger.warning_once(
                    "Disabling kernels at runtime is a no-op as there is no 'unkernelize' routine; keeping current kernels active."
                )
            self._use_kernels = False

    def _default_compile_config(self) -> CompileConfig:
        """Build the default `CompileConfig` for `get_compiled_call`.

        Inductor + `reduce-overhead` (the `CompileConfig` defaults) target CUDA.
        torch_tpu registers its own TorchDynamo backend named `"tpu"`; route
        `device.type == "tpu"` through it with static shapes to match the
        common StaticCache + fixed-prefill usage. Neuron registers a `"neuron"`
        backend and is handled the same way."""
        if self.device.type in ("tpu", "neuron"):
            return CompileConfig(backend=self.device.type, dynamic=False, mode="default")
        return CompileConfig()

    def get_compiled_call(self, compile_config: CompileConfig | None) -> Callable:
        """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
        non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
        want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
        (where we want the speed-ups of compiled version with static shapes)."""
        # Only reset it if not present or different from previous config
        if "llama4" in self.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
            return self.__call__
        compile_config = compile_config or self._default_compile_config()
        default_config = getattr(self.generation_config, "compile_config", None) or self._default_compile_config()
        if (
            not hasattr(self, "_compiled_call")
            or getattr(self, "_last_compile_config", default_config) != compile_config
        ):
            self._last_compile_config = compile_config
            self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._compiled_call

    @classmethod
    def is_backend_compatible(cls):
        return cls._supports_attention_backend

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
        device_mesh: "DeviceMeshLike | None",
        hf_quantizer: HfQuantizer | None,
    ) -> None:
        """Move missing params/buffers off meta to their target device.

        Loaded weights are handled earlier in `convert_and_load_state_dict_in_model`
        via `DtensorShardOperation` and `set_param_for_module`. This only
        materializes keys that were not loaded (or mismatched) so
        `_initialize_missing_keys` can run proper init on them.
        """
        is_quantized = hf_quantizer is not None
        # This is the only case where we do not initialize the model on meta device, so we don't have to do anything here
        if is_deepspeed_zero3_enabled() and not is_quantized:
            return

        # In this case we need to move everything back
        if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
            for key, param in self.named_parameters():
                value = torch.zeros_like(param, device="cpu")
                _load_parameter_into_model(self, key, value)
            for key, buffer in self.named_buffers():
                value = torch.zeros_like(buffer, device="cpu")
                _load_parameter_into_model(self, key, value)
            return

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            # For TP, we may need to shard the param
            if is_dtensor(param):
                local = torch.empty(param._local_tensor.shape, dtype=param.dtype, device=param_device)
                value = torch.nn.Parameter(
                    _dtensor_from_local_like(local, param),
                    requires_grad=param.requires_grad,
                )
            _load_parameter_into_model(self, key, value)
        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """
        Initialize the missing keys (keys that are part of the model parameters, but were NOT found in the loaded state dicts), according to
        `_initialize_weights`. Indeed, since the corresponding weights are missing from the state dict, they will not be replaced and need to
        be initialized correctly (i.e. weight initialization distribution).

        Also marks non-missing params/buffers with `_is_hf_initialized` and propagates this flag to modules,
        so that `_initialize_weights` can skip fully-initialized modules entirely.
        """
        if is_fsdp_enabled() and not is_local_dist_rank_0():
            # Handle FSDP edge case when using cpu ram efficient loading to ensure it is marked as initialized
            # since it will get its weights broadcasted from rank0
            # We actually need to do that only because we want to re-initialize non-persistent buffers with correct values.
            # Everything else in the state_dict will be gathered from rank0, so we don't need re-initialization.
            # We could simply early return after buffer inits if we had a way to init only the non-persistent buffers
            for key in self.state_dict():
                try:
                    param_or_buffer = self.get_parameter_or_buffer(key)
                    param_or_buffer._is_hf_initialized = True
                except AttributeError:
                    pass  # may happen when handling pre-quantized weights
            self._is_hf_initialized = True

        # This will only initialize submodules that are not marked as initialized by the line above.
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            # keep_vars=True as we need the original tensors, so that the "_is_hf_initialized" is present on them
            not_initialized_parameters = list(
                {v for v in self.state_dict(keep_vars=True).values() if not getattr(v, "_is_hf_initialized", False)}
            )
            with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                self.initialize_weights()
        else:
            self.initialize_weights()

    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors. This is performed in-place.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = {r"rotary_emb\.inv_freq"} if has_inv_freq_buffers else set()
        # Same idea for `position_ids`: used to be a persistent buffer, now `persistent=False` in most models.
        has_position_ids_buffers = any(buffer.endswith("position_ids") for buffer, _ in self.named_buffers())
        if has_position_ids_buffers:
            additional_unexpected_patterns.add(r"(^|\.)position_ids$")

        missing_patterns = self._keys_to_ignore_on_load_missing or set()
        unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            loading_info.missing_keys = {
                key for key in loading_info.missing_keys if ignore_missing_regex.search(key) is None
            }

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            loading_info.unexpected_keys = {
                key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
            }

    def mark_tied_weights_as_initialized(self, loading_info):
        """Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
        later as they will be tied (overwritten) anyway.
        This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
        running inits on them is very costly."""
        for tied_param in getattr(self, "all_tied_weights_keys", {}).keys():
            param = self.get_parameter(tied_param)
            setattr(param, "_is_hf_initialized", True)

        # Some custom code models define module tying (not parameter tying) in their __init__. When modules themselves are shared,
        # weights inside both modules appear in the `state_dict` but only one will appear in the safetensors checkpoints
        # as they are inherently tied because the 2 modules are the same object. In this case, once we load a parameter
        # inside one of the 2 modules, the other will also automatically be loaded and will have the `_is_hf_initialized`
        # flag (because we call `setattr` with the loaded param on the module, which is the same object), but its counterpart
        # will still appear as a missing key as we never get it out of the set (because it appears in the state_dict as well).
        # So we remove it now - otherwise it's considered missing and will be wrongly reinitialized
        # Note: this is never an issue in main Transformers, as we never do module-tying, only parameter-tying, and we know
        # which params are supposed to be tied to which other params
        if self.is_custom_code():
            # Remove those that are already initialized, but appear as missing due to module tying (only if they are not known
            # tied weights, i.e. we did not explicitly mark them as initialized just above)
            loading_info.missing_keys = {
                key
                for key in loading_info.missing_keys
                if key in self.all_tied_weights_keys
                or not getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False)
            }

    def get_parameter_or_buffer(self, target: str):
        """
        Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
        `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
        it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
        """
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")

    def named_non_persistent_buffers(
        self, recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Similar to `named_buffers`, but only yield non-persistent ones. It is handy as it's not perfectly straightforward
        to know if they are persistent or not"""
        for name, tensor in self.named_buffers(recurse=recurse, remove_duplicate=remove_duplicate):
            # We have to grab the parent here, as the attribute `_non_persistent_buffers_set` is on the immediate
            # parent only
            parent, buf_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.get_submodule(parent)
            if buf_name in parent._non_persistent_buffers_set:
                yield name, tensor

    def train(self, mode: bool = True):
        changed_mode = self.training != mode
        out = super().train(mode)
        # Avoid recasting kernels if not necessary
        if self.use_kernels and changed_mode:
            self.set_use_kernels(True)
        return out

    def eval(self):
        return self.train(False)

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

class Qwen2_5_VLPreTrainedModel:
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


