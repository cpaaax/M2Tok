"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# from prismatic.vla.action_tokenizer import ActionTokenizer
from .data_transform import RLDSBatchTransform
# from .load_data.hdf5_vla_dataset import HDF5VLADataset
from .load_data.hdf5_vla_dataset_read_all_data import HDF5VLADataset

import numpy as np
from transformers import AutoProcessor


from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100



def pad_sequences_to_length(sequences, target_length, batch_first=True, padding_value=0):
    # sequences: A list containing 1D tensors of varying lengths
    # target_length: The target length to pad to

    padded_sequences = []
    for seq in sequences:
        # Pad the sequence if it is shorter than the target length
        if len(seq) < target_length:
            padded_seq = torch.cat([seq, torch.full((target_length - len(seq),), padding_value)])
        else:
            # Truncate the sequence if it exceeds the target length
            padded_seq = seq[:target_length]

        padded_sequences.append(padded_seq)

    padded_sequences = pad_sequence(padded_sequences, batch_first=batch_first, padding_value=padding_value)

    return padded_sequences

@dataclass
class PaddedCollatorForImageActionPrediction:

    state_max: float = 3.0
    state_min: float = -2.5
    state_vocab_size: int=256
    state_token_start_idx: int=None
    pad_token_id: int=None
    target_length: int=None

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        input_text_ids = [torch.from_numpy(instance["input_text_ids"]).long() for instance in instances]
        pixel_values_steps = torch.stack([torch.stack(instance["pixel_values_steps"]) for instance in instances])

        action_steps = torch.stack([torch.from_numpy(instance["action_steps_ids"]) for instance in instances])


        # process states
        states = []
        bins = np.linspace(self.state_min, self.state_max, self.state_vocab_size)
        for instance in instances:
            state = np.clip(instance["state_ids"], self.state_min, self.state_max)
            discretized_state = np.digitize(state, bins)+self.state_token_start_idx-1
            states.append(torch.from_numpy(discretized_state))
        state_ids = torch.stack(states)

        input_text_ids = pad_sequences_to_length(input_text_ids, self.target_length,
                                                 batch_first=True, padding_value=self.pad_token_id)

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        output = dict(
            pixel_values_steps=pixel_values_steps,
            input_text_ids=input_text_ids,
            state_ids=state_ids,
            action_steps_ids=action_steps,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output



def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    # image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    # prompt_builder_fn: Type[PromptBuilder],
    # default_image_resolution: Tuple[int, int, int],
    act_token_start_idx: int,
    state_token_start_idx: int,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    img_history_size: int = 1,
    action_chunk_size: int = 1,
    image_size: int = 224,
    state_dim: int = 14,
    state_max: float=3.0,
    state_min: float=-2.5,
    state_vocab_size: int=256,
    target_length: int=30,
    instruction_path: str = None

) -> Tuple[Dataset, PaddedCollatorForImageActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    batch_transform = RLDSBatchTransform(
        # action_tokenizer,
        tokenizer, image_size=image_size, predict_stop_token=predict_stop_token,
        act_token_start_idx=act_token_start_idx, img_history_size=img_history_size, action_chunk_size=action_chunk_size
    )
    collator = PaddedCollatorForImageActionPrediction(
        state_min=state_min, state_max=state_max, state_vocab_size=state_vocab_size,
        state_token_start_idx=state_token_start_idx, pad_token_id=tokenizer.pad_token_id,
        target_length=target_length
    )


    dataset = HDF5VLADataset(
        data_root_dir,
        action_chunk_size,
        img_history_size,
        state_dim,
        batch_transform,
        instruction_path
        # dataset_num
        # resize_resolution=default_image_resolution[1:],
    )

    return dataset, collator