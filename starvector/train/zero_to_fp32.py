#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import glob
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

# Constants from DeepSpeed (hardcoded to remove DeepSpeed dependency)
DS_VERSION = "version"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"
FP32_FLAT_GROUPS = "fp32_flat_groups"
ZERO_STAGE = "zero_stage"
PARTITION_COUNT = "partition_count"
PARAM_SHAPES = "param_shapes"
BUFFER_NAMES = "buffer_names"
FROZEN_PARAM_SHAPES = "frozen_param_shapes"
FROZEN_PARAM_FRAGMENTS = "frozen_param_fragments"

debug = 0

# load to cpu
device = torch.device('cpu')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_optim_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")


def get_model_state_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_model_states.pt")


@dataclass
class zero_model_state:
    buffers: dict
    param_shapes: dict
    shared_params: list
    ds_version: int
    frozen_param_shapes: dict
    frozen_param_fragments: dict


def parse_model_states(files):
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location=device)

        if BUFFER_NAMES not in state_dict:
            raise ValueError(f"{file} is not a model state checkpoint")
        buffer_names = state_dict[BUFFER_NAMES]
        if debug:
            print("Found buffers:", buffer_names)

        # recover just the buffers while restoring them to fp32 if they were saved in fp16
        buffers = {k: v.float() for k, v in state_dict["module"].items() if k in buffer_names}
        param_shapes = state_dict[PARAM_SHAPES]

        # collect parameters that are included in param_shapes
        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)

        # update with frozen parameters
        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            if debug:
                print(f"Found frozen_param_shapes: {frozen_param_shapes}")
            param_names += list(frozen_param_shapes.keys())

        # handle shared params
        shared_params = [[k, v] for k, v in state_dict["shared_params"].items()]

        ds_version = state_dict.get(DS_VERSION, None)

        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)

        z_model_state = zero_model_state(buffers=buffers,
                                         param_shapes=param_shapes,
                                         shared_params=shared_params,
                                         ds_version=ds_version,
                                         frozen_param_shapes=frozen_param_shapes,
                                         frozen_param_fragments=frozen_param_fragments)
        zero_model_states.append(z_model_state)

    return zero_model_states


def parse_optim_states(files, ds_checkpoint_dir):
    total_files = len(files)
    state_dicts = []
    for f in files:
        state_dict = torch.load(f, map_location=device)
        # immediately discard the potentially huge 2 optimizer states as we only care for fp32 master weights
        # and also handle the case where it was already removed by another helper script
        state_dict["optimizer_state_dict"].pop("optimizer_state_dict", None)
        state_dicts.append(state_dict)

    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]

    # For ZeRO-2 each param group can have different partition_count as data parallelism for expert
    # parameters can be different from data parallelism for non-expert parameters. So we can just
    # use the max of the partition_count to get the dp world_size.
    if isinstance(world_size, list):
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    # the groups are named differently in each stage
    if zero_stage <= 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    if zero_stage <= 2:
        fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
    elif zero_stage == 3:
        # if there is more than one param group, there will be multiple flattened tensors - one
        # flattened tensor per group - for simplicity merge them into a single tensor
        fp32_flat_groups = [
            torch.cat(state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key], 0) for i in range(len(state_dicts))
        ]

    return zero_stage, world_size, fp32_flat_groups


def _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir):
    """
    Returns fp32 state_dict reconstructed from ds checkpoint

    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)

    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    model_files = get_model_state_files(ds_checkpoint_dir)

    zero_model_states = parse_model_states(model_files)
    print(f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}')

    if zero_stage <= 2:
        return _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states)
    elif zero_stage == 3:
        return _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states)


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _zero2_merge_frozen_params(state_dict, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        state_dict[name] = zero_model_states[0].frozen_param_fragments[name]

        if debug:
            print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes

    # Reconstruction protocol

    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i] for sd in fp32_flat_groups]
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)

    # params
    total_numel = 0
    total_params = 0
    for shapes, full_single_fp32_vector in zip(param_shapes, merged_single_partition_of_fp32_groups):
        offset = 0
        avail_numel = full_single_fp32_vector.numel()
        for name, shape in shapes.items():
            unpartitioned_numel = shape.numel() if hasattr(shape, 'numel') else math.prod(shape)
            total_numel += unpartitioned_numel
            total_params += 1

            if debug:
                print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")
            state_dict[name] = full_single_fp32_vector.narrow(0, offset, unpartitioned_numel).view(shape)
            offset += unpartitioned_numel

        # Z2 started to align to 2*world_size to improve nccl performance
        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)

        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)

        # Sanity check
        if offset != avail_numel:
            raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    _zero2_merge_frozen_params(state_dict, zero_model_states)

    _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        if debug:
            partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)
            print(
                f"Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = fp32_flat_groups[0].numel() * world_size
    
    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}

    # params
    offset = 0
    total_numel = 0
    total_params = 0
    for name, shape in param_shapes.items():
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1

        partitioned_numel, _ = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel}"
            )

        state_dict[name] = torch.cat(
            tuple(fp32_flat_groups[i].narrow(0, offset, partitioned_numel) for i in range(world_size)),
            0).narrow(0, 0, unpartitioned_numel).view(shape)
        offset += partitioned_numel

    offset *= world_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=None):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated state_dict

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint.

    Returns:
        - pytorch ``state_dict``
    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    return _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir)


def convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file, tag=None):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder.
        - ``output_file``: path to the pytorch fp32 state_dict output file
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint.
    """
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    print(f"Saving fp32 state dict to {output_file}")
    torch.save(state_dict, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument(
        "output_file",
        type=str,
        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")
    parser.add_argument("-t",
                        "--tag",
                        type=str,
                        default=None,
                        help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    global debug
    debug = args.debug

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_dir, args.output_file, tag=args.tag)


if __name__ == "__main__":
    main()
