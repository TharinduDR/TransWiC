# Created by Hansi at 1/15/2021


def validate_transformer_config(transformer_config, has_text_b=False):
    transformer_config['merge_n'] = 1

    if "entity-pool" in transformer_config['merge_type']:
        transformer_config['special_tags'] = ["<begin>", "<end>"]
    elif "entity-first" in transformer_config['merge_type']:
        transformer_config['special_tags'] = ["<begin>"]
    elif "entity-last" in transformer_config['merge_type']:
        transformer_config['special_tags'] = ["<end>"]
    print(f"updated special_tags to {transformer_config['special_tags']}")

    if "entity" in transformer_config['merge_type']:
        if has_text_b:
            transformer_config['merge_n'] = 2
    elif "concat" in transformer_config['merge_type']:
        special_tag_count = len(transformer_config['special_tags'])
        if has_text_b:
            transformer_config['merge_n'] = 2*special_tag_count
        else:
            transformer_config['merge_n'] = special_tag_count

    if "cls-" in transformer_config['merge_type']:
        transformer_config['merge_n'] += 1
    print(f"Added merge_n - {transformer_config['merge_n']}")

    return transformer_config
