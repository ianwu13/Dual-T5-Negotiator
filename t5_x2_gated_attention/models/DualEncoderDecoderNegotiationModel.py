import warnings
from typing import Optional, Tuple, Union

import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, replace_return_docstrings
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, EncoderDecoderConfig, T5ForConditionalGeneration


logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)


class MLPFeatureCombinationModule(nn.Module):

    def __init__(self, om_output_size, rg_output_size, self_output_size):
        self.mlp_projection = nn.Linear(om_output_size+rg_output_size, self_output_size)

    def forward(self, x_m, x_n):
        x = torch.cat(x_m, x_n)
        return self.mlp_projection(x)


# TODO: Create leaky cross gated attention archiecture
class LeakyGatedCrossAttentionModule(nn.Module):

    def __init__(self, om_output_size, rg_output_size, self_output_size):
        pass

    def forward(self, x_m, x_n):
        return torch.cat(x_m, x_n)


class DualEncoderDecoderConfig(PretrainedConfig):
    model_type = "dual-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "om_encoder" in kwargs and "om_decoder" in kwargs and "rg_encoder" in kwargs and "rg_decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"

        om_encoder_config = kwargs.pop("om_encoder")
        om_encoder_model_type = om_encoder_config.pop("model_type")
        om_decoder_config = kwargs.pop("om_decoder")
        om_decoder_model_type = om_decoder_config.pop("model_type")

        self.om_encoder = AutoConfig.for_model(om_encoder_model_type, **om_encoder_config)
        self.om_decoder = AutoConfig.for_model(om_decoder_model_type, **om_decoder_config)

        rg_encoder_config = kwargs.pop("rg_encoder")
        rg_encoder_model_type = rg_encoder_config.pop("model_type")
        rg_decoder_config = kwargs.pop("rg_decoder")
        rg_decoder_model_type = rg_decoder_config.pop("model_type")

        self.rg_encoder = AutoConfig.for_model(rg_encoder_model_type, **rg_encoder_config)
        self.rg_decoder = AutoConfig.for_model(rg_decoder_model_type, **rg_decoder_config)

        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, 
        om_encoder_config: PretrainedConfig, 
        om_decoder_config: PretrainedConfig, 
        rg_encoder_config: PretrainedConfig, 
        rg_decoder_config: PretrainedConfig, 
        **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for opponent model and response generation decoder_configs")
        om_decoder_config.is_decoder = True
        om_decoder_config.add_cross_attention = True

        rg_decoder_config.is_decoder = True
        rg_decoder_config.add_cross_attention = True

        return cls(
            om_encoder=om_encoder_config.to_dict(),
            om_decoder=om_decoder_config.to_dict(), 
            rg_encoder=rg_encoder_config.to_dict(),
            rg_decoder=rg_decoder_config.to_dict(), 
            **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["om_encoder"] = self.om_encoder.to_dict()
        output["om_decoder"] = self.om_decoder.to_dict()
        output["rg_encoder"] = self.rg_encoder.to_dict()
        output["rg_decoder"] = self.rg_decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


'''
# https://huggingface.co/docs/transformers/main_classes/trainer#:~:text=The%20Trainer%20class%20is

The Trainer class is optimized for 🤗 Transformers models and can have surprising behaviors when you use it on other models. When using it on your own model, make sure:

    your model always return tuples or subclasses of ModelOutput.
    your model can compute the loss if a labels argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
    your model can accept multiple label arguments (use the label_names in your TrainingArguments to indicate their name to the Trainer) but none of them should be named "label".
'''
class DualEncoderDecoderNegotiationModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = DualEncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        om_encoder: Optional[PreTrainedModel] = None,
        om_decoder: Optional[PreTrainedModel] = None,
        rg_encoder: Optional[PreTrainedModel] = None,
        rg_decoder: Optional[PreTrainedModel] = None,
        embedding_combiner: Optional[PreTrainedModel] = None
    ):
        if config is None and (om_encoder is None or om_decoder is None or rg_encoder is None or rg_decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
            
        if config is None:
            config = DualEncoderDecoderConfig.from_encoder_decoder_configs(om_encoder.config, om_decoder.config, rg_encoder.config, rg_decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.om_decoder.cross_attention_hidden_size is not None:
            if config.om_decoder.cross_attention_hidden_size != config.om_encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the om_decoder's configuration, it has to be equal"
                    f" to the om_encoder's `hidden_size`. Got {config.om_decoder.cross_attention_hidden_size} for"
                    f" `config.om_decoder.cross_attention_hidden_size` and {config.om_encoder.hidden_size} for"
                    " `config.om_encoder.hidden_size`."
                )
        if config.rg_decoder.cross_attention_hidden_size is not None:
            if config.rg_decoder.cross_attention_hidden_size != config.rg_encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the rg_decoder's configuration, it has to be equal"
                    f" to the rg_encoder's `hidden_size`. Got {config.rg_decoder.cross_attention_hidden_size} for"
                    f" `config.rg_decoder.cross_attention_hidden_size` and {config.rg_encoder.hidden_size} for"
                    " `config.rg_encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if om_encoder is None:
            om_encoder = AutoModel.from_config(config.om_encoder)
        if rg_encoder is None:
            rg_encoder = AutoModel.from_config(config.rg_encoder)

        if om_decoder is None:
            om_decoder = AutoModelForCausalLM.from_config(config.om_decoder)
        if rg_decoder is None:
            rg_decoder = AutoModelForCausalLM.from_config(config.rg_decoder)

        self.om_encoder = om_encoder
        self.om_decoder = om_decoder
        self.rg_encoder = rg_encoder
        self.rg_decoder = rg_decoder

        if self.om_encoder.config.to_dict() != self.config.om_encoder.to_dict():
            logger.warning(
                f"Config of the om_encoder: {self.om_encoder.__class__} is overwritten by shared om_encoder config:"
                f" {self.config.om_encoder}"
            )
        if self.om_decoder.config.to_dict() != self.config.om_decoder.to_dict():
            logger.warning(
                f"Config of the om_decoder: {self.om_decoder.__class__} is overwritten by shared om_decoder config:"
                f" {self.config.om_decoder}"
            )
            
        if self.rg_encoder.config.to_dict() != self.config.rg_encoder.to_dict():
            logger.warning(
                f"Config of the rg_encoder: {self.rg_encoder.__class__} is overwritten by shared rg_encoder config:"
                f" {self.config.rg_encoder}"
            )
        if self.rg_decoder.config.to_dict() != self.config.rg_decoder.to_dict():
            logger.warning(
                f"Config of the rg_decoder: {self.rg_decoder.__class__} is overwritten by shared rg_decoder config:"
                f" {self.config.rg_decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.om_encoder.config = self.om_config.encoder
        self.om_decoder.config = self.om_config.decoder
        self.rg_encoder.config = self.rg_config.encoder
        self.rg_decoder.config = self.rg_config.decoder

        '''
        # HELPFUL CODE TO GET DECODER?ENCODER INPUT AND OUTPUT SHAPES
        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
                                                INPUT SIZE                          OUTPUT SIZE
        '''
        if embedding_combiner is not None:
            self.embedding_combiner = embedding_combiner
        else:
            # Default to using "MLPFeatureCombinationModule" as combination mechanism
            self.embedding_combiner = MLPFeatureCombinationModule(self.om_encoder.config.hidden_size, self.rg_encoder.config.hidden_size, self.rg_decoder.config.hidden_size)
            # self.embedding_combiner = LeakyGatedCrossAttentionModule(self.om_encoder.config.hidden_size, self.rg_encoder.config.hidden_size, self.rg_decoder.config.hidden_size)

        if self.om_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The om_encoder {self.om_encoder} should not have a LM Head. Please use a model without LM Head"
            )
        if self.rg_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The rg_encoder {self.rg_encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    @classmethod
    def from_finetuned_models(
        self,
        finetuned_om: Optional[T5ForConditionalGeneration] = None,
        finetuned_rgm: Optional[T5ForConditionalGeneration] = None,
        finetuned_om_pth: Optional[str] = None,
        finetuned_rgm_pth: Optional[str] = None,
        **kwargs
    ):
        # Opponent model loading
        if (finetuned_om is None) and (finetuned_om_pth is None):
            print('Instantiated model object and saved model path provided for opponent model. Defaulting to already instantiated version')
            finetuned_om_pth = None
        elif finetuned_om is None:
            finetuned_om = T5ForConditionalGeneration.from_pretrained(finetuned_om_pth)

            token_embeddings_size = kwargs.pop("token_embeddings_size", False)
            if token_embeddings_size:
                finetuned_om.resize_token_embeddings(len(token_embeddings_size))
            eos_token_id = kwargs.pop("eos_token_id", False)
            if eos_token_id:
                finetuned_om.config.eos_token_id = eos_token_id

        # Response generation model loading
        if (finetuned_rgm is None) and (finetuned_rgm_pth is None):
            print('Instantiated model object and saved model path provided for response generation model. Defaulting to already instantiated version')
            finetuned_rgm_pth = None
        elif finetuned_rgm is None:
            finetuned_rgm = T5ForConditionalGeneration.from_pretrained(finetuned_rgm_pth)

            token_embeddings_size = kwargs.pop("token_embeddings_size", False)
            if token_embeddings_size:
                finetuned_rgm.resize_token_embeddings(len(token_embeddings_size))
            eos_token_id = kwargs.pop("eos_token_id", False)
            if eos_token_id:
                finetuned_rgm.config.eos_token_id = eos_token_id


        om_encoder = finetuned_om.get_encoder()
        om_decoder = finetuned_om.get_decoder()
        rg_encoder = finetuned_rgm.get_encoder()
        rg_decoder = finetuned_rgm.get_decoder()

        # instantiate config with corresponding kwargs
        config = DualEncoderDecoderConfig.from_encoder_decoder_configs(om_encoder.config, om_decoder.config, rg_encoder.config, rg_decoder.config, **kwargs)

        return self(config=config, om_encoder=om_encoder, om_decoder=om_decoder, rg_encoder=rg_encoder, rg_decoder=rg_decoder)

    # TODO: HERE ***********************************
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        om_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        rg_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Use "kwargs_encoder" and "kwargs_decoder" for both om and rg enc/dec
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # OM encoder embeddings
        if om_encoder_outputs is None:
            om_encoder_outputs = self.om_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(om_encoder_outputs, tuple):
            om_encoder_outputs = BaseModelOutput(*om_encoder_outputs)
        om_encoder_hidden_states = om_encoder_outputs[0]

        # RG encoder embeddings
        if rg_encoder_outputs is None:
            rg_encoder_outputs = self.rg_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(rg_encoder_outputs, tuple):
            rg_encoder_outputs = BaseModelOutput(*rg_encoder_outputs)
        rg_encoder_hidden_states = rg_encoder_outputs[0]
        
        # Combine RG and OM embeddings
        rg_encoder_hidden_states = self.embedding_combiner(om_encoder_hidden_states, rg_encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        om_decoder_outputs = self.om_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=om_encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        rg_decoder_outputs = self.rg_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=rg_encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        om_loss = None
        rg_loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            loss_fct = CrossEntropyLoss()

            om_logits = om_decoder_outputs.logits if return_dict else om_decoder_outputs[0]
            om_loss = loss_fct(om_logits.reshape(-1, self.om_decoder.config.vocab_size), labels.view(-1))

            rg_logits = rg_decoder_outputs.logits if return_dict else rg_decoder_outputs[0]
            rg_loss = loss_fct(rg_logits.reshape(-1, self.rg_decoder.config.vocab_size), labels.view(-1))

        # maybe weighted sum? - e.g. loss = lambda*loss + (1-lambda)*om_loss
        if (rg_loss is not None) and (om_loss is not None):
            loss = loss + om_loss
        else:
            loss = None

        if not return_dict:
            if loss is not None:
                return (loss,) + om_decoder_outputs + om_encoder_outputs + rg_decoder_outputs + rg_encoder_outputs
            else:
                return om_decoder_outputs + om_encoder_outputs + rg_decoder_outputs + rg_encoder_outputs

        '''
        # TODO: HERE
        om_encoder_hidden_states
        rg_encoder_hidden_states IS COMBINED EMBEDDINGS
        '''

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            om_decoder_base_model_prefix = self.om_decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.om_encoder, self.om_decoder._modules[om_decoder_base_model_prefix], self.om_decoder.base_model_prefix
            )

            rg_decoder_base_model_prefix = self.rg_decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.rg_encoder, self.rg_decoder._modules[rg_decoder_base_model_prefix], self.rg_decoder.base_model_prefix
            )

    def _set_gradient_checkpointing(self, module, value=False):
        # call both encoder and decoder function on gradient checkpointing
        self.om_encoder._set_gradient_checkpointing(module, value=value)
        self.om_decoder._set_gradient_checkpointing(module, value=value)
        self.rg_encoder._set_gradient_checkpointing(module, value=value)
        self.rg_decoder._set_gradient_checkpointing(module, value=value)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # TODO/CHECK: May not be proper output
    def _reorder_cache(self, past, beam_idx):
        # ORIGINAL CODE
        # return self.decoder._reorder_cache(past, beam_idx)
        
        # apply decoder cache reordering here
        return self.om_decoder._reorder_cache(past, beam_idx), self.rg_decoder._reorder_cache(past, beam_idx)

    #############################################
    # Encoder/decoder utility functions
    def get_om_encoder(self):
        return self.om_encoder

    def get_rg_encoder(self):
        return self.rg_encoder

    def get_om_decoder(self):
        return self.om_decoder

    def get_rg_decoder(self):
        return self.rg_decoder

    def get_om_input_embeddings(self):
        return self.om_encoder.get_input_embeddings()

    def get_rg_input_embeddings(self):
        return self.rg_encoder.get_input_embeddings()

    def get_om_output_embeddings(self):
        return self.om_decoder.get_output_embeddings()

    def get_rg_output_embeddings(self):
        return self.rg_decoder.get_output_embeddings()

    def set_om_output_embeddings(self, new_embeddings):
        return self.om_decoder.set_output_embeddings(new_embeddings)

    def set_rg_output_embeddings(self, new_embeddings):
        return self.rg_decoder.set_output_embeddings(new_embeddings)
