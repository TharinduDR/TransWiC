from transformers import ElectraForPreTraining, ElectraForMaskedLM, PreTrainedModel


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()