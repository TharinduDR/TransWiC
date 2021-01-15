# Created by Hansi at 1/15/2021
import torch


def get_pooled_entity_output(all_embeddings, entity_positions, pool):
    """
    Get pooled entity sub-token embeddings
    """
    list_outputs = []
    for i in range(all_embeddings.shape[0]):
        temp_input = all_embeddings[i]
        temp_positions = entity_positions[i]
        pooled_entity_embeddings = []
        for j in range(0, temp_positions.shape[0], 2):
            entity_embeddings = []
            # consider embeddings of entity sub tokens
            for r in range(temp_positions[j]+1, temp_positions[j + 1]):
                temp_embedding = temp_input[r, :]
                entity_embeddings.append(temp_embedding)
            merge = torch.cat(entity_embeddings, 0)
            merge = (merge.unsqueeze(0)).unsqueeze(0)
            entity_tensor = pool(merge)
            pooled_entity_embeddings.append(entity_tensor[0, 0])
        output = torch.cat(pooled_entity_embeddings, 0)
        list_outputs.append(output)

    return torch.stack(list_outputs, dim=0)


def get_first_entity_output(all_embeddings, entity_positions, tensor_indices):
    """
    Get entity first sub-token embeddings
    """
    list_outputs = []
    for i in range(0, entity_positions.shape[1]):
        first_token_positions = torch.add(entity_positions[:, i], -1)
        temp_output = all_embeddings[tensor_indices, first_token_positions, :]
        list_outputs.append(temp_output)
    return torch.cat(list_outputs, 1)


def get_last_entity_output(all_embeddings, entity_positions, tensor_indices):
    """
    Get entity last sub-token embeddings
    """
    list_outputs = []
    for i in range(0, entity_positions.shape[1]):
        last_token_positions = torch.add(entity_positions[:, i], -1)
        temp_output = all_embeddings[tensor_indices, last_token_positions, :]
        list_outputs.append(temp_output)
    return torch.cat(list_outputs, 1)
