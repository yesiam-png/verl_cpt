from typing import List, Union, Any
import bisect
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


def series_to_item(ls: Any) -> Any:
    import numpy as _np
    import pandas as _pd
    while isinstance(ls, (_pd.Series, _np.ndarray)) and len(ls) == 1:
        ls = ls[0]
    return ls


def tokenize_response(raw: Any, tokenizer: PreTrainedTokenizer, max_length: int, truncation: str):
    item = series_to_item(raw)
    response_str = str(item) + tokenizer.eos_token
    out = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)
    ids = out["input_ids"][0]
    attn = out["attention_mask"][0]
    L = ids.size(0)
    pad_id = tokenizer.pad_token_id
    if L < max_length:
        pad_len = max_length - L
        ids = torch.cat([ids, ids.new_full((pad_len,), pad_id)])
        attn = torch.cat([attn, attn.new_zeros((pad_len,))])
    elif L > max_length:
        if truncation == "left":
            ids = ids[-max_length:]
            attn = attn[-max_length:]
        elif truncation == "right":
            ids = ids[:max_length]
            attn = attn[:max_length]
        else:
            raise NotImplementedError(
                f"Sequence length {L} > max_length {max_length}; set truncation='left' or 'right'"
            )
    position_ids = compute_position_id_with_mask(attn)
    loss_mask = attn.clone()
    loss_mask[min(L, loss_mask.size(0)) - 1] = 0
    return {"input_ids": ids, "attention_mask": attn, "position_ids": position_ids, "loss_mask": loss_mask}


class CPTDataset(Dataset):
    """
    Random-access Parquet dataset with __len__ and __getitem__.
    Caches ParquetFile objects in memory and metadata indices in init.
    """
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: Union[str, PreTrainedTokenizer],
        config: dict,
    ):
        response_key = config.get("response_key", "response")
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            from verl.utils import hf_tokenizer
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.response_key = response_key
        self.max_length = max_length
        self.truncation = truncation

        # cache ParquetFile objects
        self._pqfiles = [pq.ParquetFile(p) for p in parquet_files]

        # build row-group index
        self.cumulative_file_rows = [0]
        self.file_group_prefix = []
        for pqf in self._pqfiles:
            counts = [pqf.metadata.row_group(i).num_rows for i in range(pqf.num_row_groups)]
            prefix = []
            total = 0
            for c in counts:
                total += c
                prefix.append(total)
            self.file_group_prefix.append(prefix)
            self.cumulative_file_rows.append(self.cumulative_file_rows[-1] + total)
        self.total_rows = self.cumulative_file_rows[-1]

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total_rows
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"Index {idx} out-of-bounds for dataset of size {self.total_rows}")
        # locate file and row-group
        file_idx = bisect.bisect_right(self.cumulative_file_rows, idx) - 1
        idx_in_file = idx - self.cumulative_file_rows[file_idx]
        gr_prefix = self.file_group_prefix[file_idx]
        gr = bisect.bisect_right(gr_prefix, idx_in_file)
        prev = gr_prefix[gr - 1] if gr > 0 else 0
        pos_in_group = idx_in_file - prev

        # read only needed row-group
        pqf = self._pqfiles[file_idx]
        table = pqf.read_row_group(gr, columns=[self.response_key])
        raw = table.column(self.response_key).to_pylist()[pos_in_group]
        return tokenize_response(raw, self.tokenizer, self.max_length, self.truncation)