# Created by Hansi at 1/15/2021
import torch


def get_pooled_entity_output(all_embeddings, entity_positions, pool):
    """
    Get pooled entity sub-token embeddings

    :param all_embeddings: tensor of all embeddings
        if batch_size=8, max_seq_length=120, token_embedding_dimension=768, size of all_embeddings=([8,120,178])
    :param entity_positions: tensor of entity positions
        per instance both begin and end tag positions of each entity need to be provided.
        [[begin tag position of entity1, end tag position of entity1, begin tag position of entity2, end tag position of
         entity2,...]...]
    :param pool:
    :return: tensor of pooled entity embeddings
    """
    list_outputs = []
    for i in range(all_embeddings.shape[0]):  # iterate through each instance of the batch
        temp_input = all_embeddings[i]
        temp_positions = entity_positions[i]  # entity positions of ith instance in the batch
        pooled_entity_embeddings = []
        for j in range(0, temp_positions.shape[0], 2):  # iterate through all begin tag entity positions
            entity_embeddings = []
            # consider embeddings of entity sub tokens
            # iterate through all positions from begin tag position+1 to end tag position-1
            for r in range(temp_positions[j]+1, temp_positions[j + 1]):
                temp_embedding = temp_input[r, :]
                entity_embeddings.append(temp_embedding)
            merge = torch.cat(entity_embeddings, 0)
            merge = (merge.unsqueeze(0)).unsqueeze(0)  # convert tensor of size ([n]) to ([1,1,n])
            entity_tensor = pool(merge)
            pooled_entity_embeddings.append(entity_tensor[0, 0])
        output = torch.cat(pooled_entity_embeddings, 0)
        list_outputs.append(output)

    return torch.stack(list_outputs, dim=0)


def get_first_entity_output(all_embeddings, entity_positions, tensor_indices):
    """
    Get entity first sub-token embeddings

    :param all_embeddings: tensor of all embeddings
        if batch_size=8, max_seq_length=120, token_embedding_dimension=768, size of all_embeddings=([8,120,178])
    :param entity_positions: tensor of entity positions
        if batch_size=8, number of targeted entities=2, size of entity_positions=([8,2])
        per instance begin tag position of each entity need to be provided.
        [[begin tag position of entity1, begin tag position of entity2]...]
    :param tensor_indices: tensor of batch indices
        if batch_size=8, tensor([0,1,2,3,4,5,6,7])
    :return: tensor of embeddings at entity first sub-token
        if more than one entity  is targeted per instance, concatenated embedding will be returned.
        if batch_size=8, number of targeted entities=2, token_embedding_dimension=768, size of output=([8,1536])
    """
    list_outputs = []
    for i in range(0, entity_positions.shape[1]):
        first_token_positions = torch.add(entity_positions[:, i], 1)
        temp_output = all_embeddings[tensor_indices, first_token_positions, :]
        list_outputs.append(temp_output)
    return torch.cat(list_outputs, 1)


def get_last_entity_output(all_embeddings, entity_positions, tensor_indices):
    """
    Get entity last sub-token embeddings

    :param all_embeddings: tensor of all embeddings
        if batch_size=8, max_seq_length=120, token_embedding_dimension=768, size of all_embeddings=([8,120,178])
    :param entity_positions: tensor of entity positions
        if batch_size=8, number of targeted entities=2, size of entity_positions=([8,2])
    :param tensor_indices: tensor of batch indices
        if batch_size=8, tensor([0,1,2,3,4,5,6,7])
    :return: tensor of embeddings at entity last sub-token
        if more than one entity  is targeted per instance, concatenated embedding will be returned.
        if batch_size=8, number of targeted entities=2, token_embedding_dimension=768, size of output=([8,1536])
    """
    list_outputs = []
    for i in range(0, entity_positions.shape[1]):
        last_token_positions = torch.add(entity_positions[:, i], -1)
        temp_output = all_embeddings[tensor_indices, last_token_positions, :]
        list_outputs.append(temp_output)
    return torch.cat(list_outputs, 1)


def process_embeddings(outputs, entity_positions, merge_type, pool):
    if merge_type == 'cls':
        processed_output = outputs[1]
    else:
        if entity_positions is None:
            raise ValueError('Required entity positions are not provided!')
        else:
            indices = [i for i in range(0, entity_positions.shape[0])]
            tensor_indices = torch.tensor(indices, dtype=torch.long)

            # exact match is not used, because combined merge_types (e.g. cls-concat) are possible.
            if "concat" in merge_type:
                list_processed_output = []
                for i in range(0, entity_positions.shape[1]):
                    temp_output = outputs[0][tensor_indices, entity_positions[:, i], :]
                    list_processed_output.append(temp_output)
                processed_output = torch.cat(list_processed_output, 1)

            elif "add" in merge_type or "avg" in merge_type:
                processed_output = outputs[0][tensor_indices, entity_positions[:, 0], :]
                for i in range(1, entity_positions.shape[1]):
                    temp_output = outputs[0][tensor_indices, entity_positions[:, i], :]
                    processed_output = processed_output.add(temp_output)
                if "avg" in merge_type:
                    processed_output = torch.div(processed_output, entity_positions.shape[1])

            elif "entity-pool" in merge_type:
                if entity_positions.shape[1] % 2 != 0:
                    raise ValueError("begin or end of the entity is missing!")
                processed_output = get_pooled_entity_output(outputs[0], entity_positions, pool)

            elif "entity-first" in merge_type:
                processed_output = get_first_entity_output(outputs[0], entity_positions, tensor_indices)

            elif "entity-last" in merge_type:
                processed_output = get_last_entity_output(outputs[0], entity_positions, tensor_indices)

            else:  # If merge type is unknown
                raise KeyError(f"Unknown merge type found - {merge_type}")

            if "cls-" in merge_type:
                processed_output = torch.cat((outputs[1], processed_output), 1)
    return processed_output

