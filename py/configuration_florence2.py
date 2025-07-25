import warnings
""" Florence-2 configuration"""

from typing import Optional

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Florence2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2VisionModel`]. It is used to instantiate a Florence2VisionModel
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the 
    defaults will yield a similar configuration to that of the Florence2VisionModel architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout rate of the drop path layer.
        patch_size (`List[int]`, *optional*, defaults to [7, 3, 3, 3]):
            The patch size of the image.
        patch_stride (`List[int]`, *optional*, defaults to [4, 2, 2, 2]):
            The patch stride of the image.
        patch_padding (`List[int]`, *optional*, defaults to [3, 1, 1, 1]):
            The patch padding of the image.
        patch_prenorm (`List[bool]`, *optional*, defaults to [false, true, true, true]):
            Whether to apply layer normalization before the patch embedding layer.
        enable_checkpoint (`bool`, *optional*, defaults to False):
            Whether to enable checkpointing.
        dim_embed (`List[int]`, *optional*, defaults to [256, 512, 1024, 2048]):
            The dimension of the embedding layer.
        num_heads (`List[int]`, *optional*, defaults to [8, 16, 32, 64]):
            The number of attention heads.
        num_groups (`List[int]`, *optional*, defaults to [8, 16, 32, 64]):
            The number of groups.
        depths (`List[int]`, *optional*, defaults to [1, 1, 9, 1]):
            The depth of the model.
        window_size (`int`, *optional*, defaults to 12):
            The window size of the model.
        projection_dim (`int`, *optional*, defaults to 1024):
            The dimension of the projection layer.
        visual_temporal_embedding (`dict`, *optional*):
            The configuration of the visual temporal embedding.
        image_pos_embed (`dict`, *optional*):
            The configuration of the image position embedding.
        image_feature_source (`List[str]`, *optional*, defaults to ["spatial_avg_pool", "temporal_avg_pool"]):
            The source of the image feature.
    Example:

    ```python
    >>> from transformers import Florence2VisionConfig, Florence2VisionModel

    >>> # Initializing a Florence2 Vision style configuration
    >>> configuration = Florence2VisionConfig()

    >>> # Initializing a model (with random weights)
    >>> model = Florence2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2_vision"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        drop_path_rate=0.1,
        patch_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        patch_prenorm=[False, True, True, True],
        enable_checkpoint=False,
        dim_embed=[256, 512, 1024, 2048],
        num_heads=[8, 16, 32, 64],
        num_groups=[8, 16, 32, 64],
        depths=[1, 1, 9, 1],
        window_size=12,
        projection_dim=1024,
        visual_temporal_embedding=None,
        image_pos_embed=None,
        image_feature_source=["spatial_avg_pool", "temporal_avg_pool"],
        **kwargs,
    ):
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_prenorm = patch_prenorm
        self.enable_checkpoint = enable_checkpoint
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.depths = depths
        self.window_size = window_size
        self.projection_dim = projection_dim
        self.visual_temporal_embedding = visual_temporal_embedding
        self.image_pos_embed = image_pos_embed
        self.image_feature_source = image_feature_source

        super().__init__(**kwargs)



class Florence2LanguageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2LanguagePreTrainedModel`]. It is used to instantiate a BART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BART
    [facebook/bart-large](https://huggingface.co/facebook/bart-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 51289):
            Vocabulary size of the Florence2Language model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Florence2LanguageModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels (`int`, *optional*, defaults to 3):
            The number of labels to use in [`Florence2LanguageForSequenceClassification`].
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Example:

    ```python
    >>> from transformers import Florence2LanguageConfig, Florence2LanguageModel

    >>> # Initializing a Florence2 Language style configuration
    >>> configuration = Florence2LanguageConfig()

    >>> # Initializing a model (with random weights)
    >>> model = Florence2LangaugeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2_language"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=51289,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )

class Florence2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2ForConditionalGeneration`]. It is used to instantiate an
    Florence-2 model according to the specified arguments, defining the model architecture. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Florence2VisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. 
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        vocab_size (`int`, *optional*, defaults to 51289):
            Vocabulary size of the Florence2model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~Florence2ForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 1024):
            Dimension of the multimodal projection space.

    Example:

    ```python
    >>> from transformers import Florence2ForConditionalGeneration, Florence2Config, CLIPVisionConfig, BartConfig

    >>> # Initializing a clip-like vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Bart config
    >>> text_config = BartConfig()

    >>> # Initializing a Florence-2 configuration
    >>> configuration = Florence2Config(vision_config, text_config)

    >>> # Initializing a model from the florence-2 configuration
    >>> model = Florence2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        vocab_size=51289,
        projection_dim=1024,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        if vision_config is not None:
            vision_config = PretrainedConfig(**vision_config)
        self.vision_config = vision_config
        self.vocab_size = self.vocab_size

        self.text_config = text_config
        if text_config is not None:
            self.text_config = Florence2LanguageConfig(**text_config)


        super().__init__(**kwargs)

