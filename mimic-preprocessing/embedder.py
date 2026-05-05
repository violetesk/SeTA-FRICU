"""Local embedding client for transformer-based text embedding.

Provides the EmbeddingClient class, a reusable wrapper around
HuggingFace AutoModel + AutoTokenizer that computes L2-normalized
embeddings using last-token pooling. Designed for use with models
like Qwen3-Embedding-8B with flash attention 2.

Usage:
    from preprocessing.embedder import EmbeddingClient

    client = EmbeddingClient(model_name_or_path="/path/to/Qwen3-Embedding-8B")
    vector = client.embed("Patient presented with fever and chills.")
    batch_vectors = client.embed_batch(["text1", "text2"], batch_size=32)
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, model_name_or_path: str = "Qwen3-Embedding-8B", device: str = None):
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading local model: {model_name_or_path} to {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                padding_side='left',
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16, 
                device_map=self.device
            )
            self.model.eval()
            
            self.hidden_size = self.model.config.hidden_size
            logger.info(f"Model loaded successfully! Output dimension: {self.hidden_size}")

        except Exception as e:
            logger.critical(f"Failed to load model. Error: {e}")
            raise e

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
            return last_hidden_states[batch_indices, sequence_lengths]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> Optional[List[List[float]]]:
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            try:
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=8192,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.extend(embeddings.float().cpu().numpy().tolist())
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM triggered! Current batch_size={batch_size}. Consider reducing batch_size.")
                    return None
                else:
                    logger.error(f"Inference error: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unknown error: {e}")
                return None
                
        return all_embeddings

    def embed(self, text: str) -> Optional[List[float]]:
        results = self.embed_batch([text], batch_size=1)
        if results:
            return results[0]
        return None
