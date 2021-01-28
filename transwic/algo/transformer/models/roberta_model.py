import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers import BertPreTrainedModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from transwic.algo.transformer.models.model_util import get_pooled_entity_output, get_first_entity_output, \
    get_last_entity_output


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, merge_n):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj = nn.Linear(config.hidden_size * merge_n, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# class RobertaForSequenceClassification(BertPreTrainedModel):
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, weight=None, merge_type=None, merge_n=1):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        if merge_type is not None and "-pool" in merge_type:
            self.pool = nn.AdaptiveAvgPool1d(config.hidden_size)

        self.classifier = RobertaClassificationHead(config, merge_n)
        self.weight = weight

        self.merge_type = merge_type
        self.merge_n = merge_n

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_positions=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # # sequence_output = outputs[0]
        # sequence_output = outputs[1]

        # if entity positions are given, get embeddings at the given positions
        if entity_positions is not None:
            indices = [i for i in range(0, entity_positions.shape[0])]
            tensor_indices = torch.tensor(indices, dtype=torch.long)

            if self.merge_type is not None:
                if "concat" in self.merge_type:
                    list_pooled_output = []
                    for i in range(0, entity_positions.shape[1]):
                        temp_pooled_output = outputs[0][tensor_indices, entity_positions[:, i], :]
                        list_pooled_output.append(temp_pooled_output)
                    pooled_output = torch.cat(list_pooled_output, 1)

                elif "add" in self.merge_type or "avg" in self.merge_type:
                    pooled_output = outputs[0][tensor_indices, entity_positions[:, 0], :]
                    for i in range(1, entity_positions.shape[1]):
                        temp_pooled_output = outputs[0][tensor_indices, entity_positions[:, i], :]
                        pooled_output = pooled_output.add(temp_pooled_output)
                    if "avg" in self.merge_type:
                        pooled_output = torch.div(pooled_output, entity_positions.shape[1])

                elif "entity-pool" in self.merge_type:
                    if entity_positions.shape[1] % 2 != 0:
                        raise ValueError("begin or end of the entity is missing!")
                    pooled_output = get_pooled_entity_output(outputs[0], entity_positions, self.pool)

                elif "entity-first" in self.merge_type:
                    pooled_output = get_first_entity_output(outputs[0], entity_positions, tensor_indices)

                elif "entity-last" in self.merge_type:
                    pooled_output = get_last_entity_output(outputs[0], entity_positions, tensor_indices)

                else:  # If merge type is unkown
                    raise KeyError(f"Unknown merge type found - {self.merge_type}")

                if "cls-" in self.merge_type:
                    pooled_output = torch.cat((outputs[1], pooled_output), 1)

            else:  # if no merge type defined
                raise KeyError("No merge type defined.")

        # if no entity positions are given, get the embedding of CLS token
        else:
            pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
