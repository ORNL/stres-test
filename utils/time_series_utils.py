#
# This source file is adapted from OpenTSLM open-source project; SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md); SPDX-License-Identifier: MIT
#

import torch
import torch.nn.functional as F

MAX_VALUE = 50_000
MIN_VALUE = -MAX_VALUE



def pad_time_series(batch, max_length=None):
    """
    Pad time series to the same length (either max in batch or specified max);
    References https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/model/llm/OpenTSLMFlamingo.py
    """
    time_series = [item["time_series"] for item in batch]
    # Determine target length (either specified or max in batch)
    if max_length is None:
        max_length = max(ts.shape[1] for ts in time_series)
    padded_series = []
    for ts in time_series:
        current_length = ts.shape[1]
        if current_length < max_length:
            # Pad with zeros to reach max_length; Ensure padding has the same number of dimensions as the time series
            padding_shape = list(ts.shape)
            padding_shape[1] = max_length - current_length
            padding = torch.zeros(padding_shape, device=ts.device, dtype=ts.dtype)
            padded = torch.cat([ts, padding], dim=1)
        else: # If already at or exceeding max_length, truncate
            padded = ts[:, :max_length]
        padded_series.append(padded)
    return torch.stack(padded_series)


# def extend_time_series_to_match_patch_size_and_aggregate(batch, *, patch_size: int = 4):
#     """Pad variable-length series so each sample length is a multiple of *patch_size*."""
#     for element in batch:
#         # 1) pull out the list of (1D) time‑series
#         ts_list = element["time_series"]
#         # for ts in ts_list:
#         #     print("extend_time_series_to_match_patch_size_and_aggregate: DEBUGGING -- ts.shape == ", ts.shape, flush=True)
#         if ts_list is None:
#             ts_list = []
#         # 2) convert each to a torch.Tensor (float)
#         ts_tensors = [torch.as_tensor(ts, dtype=torch.float32) for ts in ts_list]
#         # 3) find the longest series length
#         max_len = max([0]+[ts.size(0) for ts in ts_tensors])
#         # 4) round up to nearest multiple of patch_size
#         padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size
#         # 5) pad (or trim) each series to padded_len
#         padded = []
#         for ts in ts_tensors:
#             L = ts.size(0)
#             if L < padded_len:
#                 pad_amt = padded_len - L
#                 ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
#             else:
#                 ts = ts[:padded_len]
#             padded.append(ts)
#         # 6) stack into a single 2D tensor: (num_series, padded_len)
#         # for ts in padded:
#         #     print("extend_time_series_to_match_patch_size_and_aggregate: DEBUGGING -- post-padded ts.size() == ", ts.size(), flush=True)
#         element["time_series"] = torch.stack(padded, dim=0)
#     return batch
def extend_time_series_to_match_patch_size_and_aggregate(batch, *, patch_size: int=4, normalize: bool=False):
    """
    Pad variable-length series so each sample length is a multiple of *patch_size*.
    Optionally normalize each time series to have zero mean and unit variance.
    *** ADAPTED FROM https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/time_series_datasets/util.py
    """
    for element in batch:
        # 1) pull out the list of (1D) time‑series
        ts_list = element["time_series"]
        # 2) convert each to a torch.Tensor (float)
        ts_tensors = [torch.as_tensor(ts, dtype=torch.float32) for ts in ts_list]
        # 3) normalize each time series if requested
        if normalize:
            normalized_tensors = []
            for ts in ts_tensors:
                mean = ts.mean()
                std = ts.std()
                if std > 1e-8:  # Avoid division by zero
                    ts_normalized = (ts - mean) / std
                else:
                    ts_normalized = ts - mean
                normalized_tensors.append(ts_normalized)
            ts_tensors = normalized_tensors
        # 4) find the longest series length
        max_len = max([ts.size(0) for ts in ts_tensors])
        # 5) round up to nearest multiple of patch_size
        padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size
        # 6) pad (or trim) each series to padded_len
        padded = []
        for ts in ts_tensors:
            L = ts.size(0)
            if L < padded_len:
                pad_amt = padded_len - L
                ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
            else:
                ts = ts[:padded_len]
            padded.append(ts)
        # 7) stack into a single 2D tensor: (num_series, padded_len)
        element["time_series"] = torch.stack(padded, dim=0)
    return batch