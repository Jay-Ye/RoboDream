# T5-11B instruction embeddings: cached lookup (DROID instructions), optional live encoder, default fallback.

import gzip
import pickle
from typing import Optional

import numpy as np
import torch

from .vendor import log

T5_MAX_LENGTH = 512
T5_CHANNELS = 1024


def load_t5_cache(path: str) -> dict:
    '''
    :param path (str): droid_t5_text_embeddings.pkl(.gz): dict mapping instruction -> (L, 1024)
        float16 array (true sequence length L, padding stripped).
    :return dict mapping instruction -> (L, 1024) float16 tensor.
    '''
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        data = pickle.load(f)
    cache = dict()
    for (prompt, emb) in data.items():
        if not isinstance(prompt, str) or emb is None:
            continue
        emb = np.asarray(emb)
        if emb.ndim == 3:
            emb = emb[0]
        cache[prompt] = torch.from_numpy(emb[:T5_MAX_LENGTH]).to(torch.float16)
    log.info(f'Loaded {len(cache)} cached T5 instruction embeddings from {path}')
    return cache


class T5Encoder:
    '''
    Live T5-11B encoder (same tokenization / zero padding as the cached embeddings).
    '''

    def __init__(self, model_path: str, device: str = 'cuda'):
        from transformers import T5EncoderModel, T5TokenizerFast
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, prompts: list[str], max_length: int = T5_MAX_LENGTH) -> tuple:
        '''
        :return (emb, lengths): (B, max_length, 1024) float tensor zeroed beyond each length, (B,) lengths.
        '''
        batch = self.tokenizer.batch_encode_plus(
            prompts, return_tensors='pt', truncation=True, padding='max_length',
            max_length=max_length, return_length=True, return_offsets_mapping=False)
        input_ids = batch.input_ids.to(self.device)
        attn_mask = batch.attention_mask.to(self.device)
        emb = self.model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()
        for b in range(emb.shape[0]):
            emb[b, lengths[b]:] = 0
        return (emb, lengths)


class TextEmbedder:
    '''
    Maps instructions to (512, 1024) cross-attention embeddings: cached T5 embeddings first, then
    the live T5 encoder if one is configured, else the default-prompt embedding (this is what the
    released inference server did for instructions outside the DROID cache).
    '''

    def __init__(self, cache_path: Optional[str] = None, default_path: Optional[str] = None,
                 t5_path: Optional[str] = None, device: str = 'cuda'):
        self.cache = load_t5_cache(cache_path) if cache_path else dict()
        self.default = None
        if default_path:
            self.default = torch.load(default_path, map_location='cpu', weights_only=True)['t5_text_embeddings']
        self.encoder = T5Encoder(t5_path, device) if t5_path else None

    def __call__(self, prompts: list[str], device='cuda') -> torch.Tensor:
        '''
        :return (B, 512, 1024) bfloat16 tensor.
        '''
        out = torch.zeros((len(prompts), T5_MAX_LENGTH, T5_CHANNELS), dtype=torch.bfloat16, device=device)
        for (b, prompt) in enumerate(prompts):
            if prompt in self.cache:
                emb = self.cache[prompt]
                out[b, :emb.shape[0]] = emb.to(device=device, dtype=torch.bfloat16)
            elif self.encoder is not None:
                (emb, length) = self.encoder.encode([prompt])
                L = int(length[0])
                out[b, :L] = emb[0, :L].to(torch.float16).to(dtype=torch.bfloat16)
                self.cache[prompt] = emb[0, :L].detach().to('cpu', torch.float16)
            elif self.default is not None:
                log.warning(f'Instruction not in the T5 cache, using the default embedding: {prompt!r}')
                out[b] = self.default.to(device=device, dtype=torch.bfloat16)
            else:
                raise KeyError(f'Instruction not in the T5 cache and no encoder / default given: {prompt!r}')
        return out
