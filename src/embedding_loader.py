import os
import numpy as np
import torch


def load_npz_embeddings(npz_path: str, device: torch.device, dtype: torch.dtype):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Embedding file not found: {npz_path}")
    data = np.load(npz_path)
    for required_key in ("prompt_embeds", "pooled_prompt_embeds", "text_ids"):
        if required_key not in data:
            raise KeyError(
                f"Embedding file {npz_path} missing required key: {required_key}"
            )

    # Convert numpy arrays to torch tensors with the correct device/dtype
    prompt_embeds = torch.from_numpy(data["prompt_embeds"]).to(device=device, dtype=dtype)
    pooled_prompt_embeds = torch.from_numpy(data["pooled_prompt_embeds"]).to(
        device=device, dtype=dtype
    )
    text_ids_np = data["text_ids"]
    # Ensure shape is [token_n, id_dim] (drop batch dim if present)
    if text_ids_np.ndim == 3 and text_ids_np.shape[0] == 1:
        text_ids_np = text_ids_np[0]
    # Text ids must be integer indices
    text_ids = torch.as_tensor(text_ids_np, device=device, dtype=torch.long)

    return prompt_embeds, pooled_prompt_embeds, text_ids


