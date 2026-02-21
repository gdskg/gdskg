import os
import sys
from pathlib import Path
from typing import List
import numpy as np

class ONNXEmbedder:
    """
    Downloads and uses Xenova/all-MiniLM-L6-v2 via ONNX and tokenizers
    to produce 384-dimensional embeddings.
    """
    
    MODEL_ID = "Xenova/all-MiniLM-L6-v2"
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(Path.home() / ".gdskg" / "models")
            
        self.model_dir = Path(model_dir) / self.MODEL_ID.replace("/", "--")
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        self.onnx_path = self.model_dir / "model.onnx"
        
        self.tokenizer = None
        self.session = None
        
    def _download_if_needed(self):
        """Downloads the ONNX model and tokenizer from HuggingFace."""
        if self.tokenizer_path.exists() and self.onnx_path.exists():
            return
            
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model {self.MODEL_ID} to {self.model_dir}...", file=sys.stderr)
        
        from huggingface_hub import hf_hub_download
        
        hf_hub_download(repo_id=self.MODEL_ID, filename="tokenizer.json", local_dir=str(self.model_dir))
        hf_hub_download(repo_id=self.MODEL_ID, filename="onnx/model_quantized.onnx", local_dir=str(self.model_dir))
        
        # Rename the downloaded model
        quantized_path = self.model_dir / "onnx" / "model_quantized.onnx"
        import shutil
        shutil.copy(quantized_path, self.onnx_path)
        
        print("Download complete.", file=sys.stderr)

    def _load(self):
        """Loads tokenizer and ONNX session into memory."""
        if self.tokenizer is not None and self.session is not None:
            return
            
        self._download_if_needed()
        
        from tokenizers import Tokenizer
        import onnxruntime as ort
        
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        # Ensure correct truncation
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        # Silence ORT warnings optionally
        options = ort.SessionOptions()
        options.log_severity_level = 3
        
        self.session = ort.InferenceSession(str(self.onnx_path), options)
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings and mean-pool the results.
        Returns a numpy array of shape (len(texts), 384).
        """
        if not texts:
            return np.array([])
            
        self._load()
        
        encoded = self.tokenizer.encode_batch(texts)
        
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        
        # Get ONNX model inputs
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        # Run inference
        ort_outs = self.session.run(None, ort_inputs)
        
        # The first output is the last_hidden_state (batch_size, seq_len, hidden_size)
        last_hidden_state = ort_outs[0]
        
        # Mean Pooling
        # Expand attention mask to match hidden state shape
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        
        # Zero out padding tokens
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        pooled_embeddings = sum_embeddings / sum_mask
        
        # L2 normalize
        norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
        normalized_embeddings = pooled_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized_embeddings
