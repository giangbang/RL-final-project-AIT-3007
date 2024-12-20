from tensordict.tensordict import TensorDict
from typing import List
import torch
def process_episodes(batch):
    batch_blue = {
        'o' : batch['o_b'],
        'a' : batch['a_b'],
        'r' : batch['r_b'],
        'd' : batch['d_b'],
        's' : batch['s'],
    }
    batch_red = {
        'o' : batch['o_r'],
        'a' : batch['a_r'],
        'r' : batch['r_r'],
        'd' : batch['d_r'],
        's' : batch['s'],
    }
    return batch_blue, batch_red

def collate_episodes(episodes: List[TensorDict], max_length:int):
    """
    Collate a list of episodes into batched TensorDict with padding.

    Args:
        episodes (List[TensorDict]): List of episodes to collate.
        device (torch.device): Device to place tensors on.

    Returns:
        TensorDict: Batched episodes with padding and masks.
    """
    # Determine the maximum episode length in the batch
    max_length = max_length

    batched = {}
    masks = torch.zeros(len(episodes), max_length, dtype=torch.bool)

    for key in episodes[0].keys():
        # Collect all steps for this key
        key_data = [episode[key] for episode in episodes]
        # Determine the shape excluding time
        shape = key_data[0].shape[1:]
        # Initialize batched tensor with padding
        if key_data[0].dtype == torch.bool:
            pad_value = False
        elif key_data[0].dtype == torch.float32:
            pad_value = 0.0
        else:
            pad_value = 0

        batched[key] = torch.zeros(len(episodes), max_length, *shape)
        batched[key].fill_(pad_value)

        for i, episode in enumerate(key_data):
            length = episode.shape[0]
            batched[key][i, :length] = episode
            # masks[i, :length] = 1  # Mark valid steps

    batched_tensordict = TensorDict(batched, batch_size=[len(episodes)])
    # batched_tensordict.set("mask", masks)
    return batched_tensordict
