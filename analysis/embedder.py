import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

class ONNXEmbedder:
    """
    Downloads and uses Xenova/all-MiniLM-L6-v2 via ONNX and tokenizers
    to produce 384-dimensional embeddings.
    """
    
    MODEL_ID = "Xenova/all-MiniLM-L6-v2"
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the ONNXEmbedder with the specified model directory.
        
        Args:
            model_dir (str, optional): The directory to store and load the model from.
        """
        if model_dir is None:
            model_dir = str(Path.home() / ".gdskg" / "models")
            
        self.model_dir = Path(model_dir) / self.MODEL_ID.replace("/", "--")
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        self.onnx_path = self.model_dir / "model.onnx"
        
        self.tokenizer = None
        self.session = None
        
    def _download_if_needed(self):
        """
        Download the ONNX model and tokenizer from HuggingFace if they are not already present.

        Returns:
            None
        """
        if self.tokenizer_path.exists() and self.onnx_path.exists():
            return
            
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model {self.MODEL_ID} to {self.model_dir}...", file=sys.stderr)
        
        from huggingface_hub import hf_hub_download
        
        hf_hub_download(repo_id=self.MODEL_ID, filename="tokenizer.json", local_dir=str(self.model_dir))
        hf_hub_download(repo_id=self.MODEL_ID, filename="onnx/model_quantized.onnx", local_dir=str(self.model_dir))
        
        quantized_path = self.model_dir / "onnx" / "model_quantized.onnx"
        import shutil
        shutil.copy(quantized_path, self.onnx_path)
        
        print("Download complete.", file=sys.stderr)

    def _load(self):
        """
        Load the tokenizer and ONNX inference session into memory.

        Returns:
            None
        """
        if self.tokenizer is not None and self.session is not None:
            return
            
        self._download_if_needed()
        
        from tokenizers import Tokenizer
        import onnxruntime as ort
        
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        options = ort.SessionOptions()
        options.log_severity_level = 3
        
        self.session = ort.InferenceSession(str(self.onnx_path), options)
        
    def _chunk_tokens(self, ids: List[int], attention_mask: List[int], type_ids: List[int], max_length: int = 256, overlap: int = 50) -> List[Dict[str, List[int]]]:
        """
        Split token lists into overlapping chunks to handle long text.

        Args:
            ids (List[int]): The token identifiers.
            attention_mask (List[int]): The attention mask for the tokens.
            type_ids (List[int]): The token type identifiers.
            max_length (int): The maximum length of each chunk.
            overlap (int): The number of tokens to overlap between chunks.

        Returns:
            List[Dict[str, List[int]]]: A list of dictionary objects, each containing token IDs, masks, and types for a chunk.
        """
        chunks = []
        length = len(ids)
        
        if length <= max_length:
            return [{"ids": ids, "attention_mask": attention_mask, "type_ids": type_ids}]
            
        start = 0
        while start < length:
            end = min(start + max_length, length)
            
            chunk_ids = ids[start:end]
            chunk_mask = attention_mask[start:end]
            chunk_type = type_ids[start:end]
            
            if len(chunk_ids) < max_length:
                pad_len = max_length - len(chunk_ids)
                chunk_ids.extend([0] * pad_len)
                chunk_mask.extend([0] * pad_len)
                chunk_type.extend([0] * pad_len)
                
            chunks.append({
                "ids": chunk_ids,
                "attention_mask": chunk_mask,
                "type_ids": chunk_type
            })
            
            start += (max_length - overlap)
            
        return chunks
        
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of strings and mean-pool the results.

        Args:
            texts (List[str]): A list of strings to generate embeddings for.

        Returns:
            List[np.ndarray]: A list of numpy arrays, one for each input text. 
            Each array contains the normalized embeddings for the chunks of that text.
        """
        if not texts:
            return []
            
        self._load()
        
        self.tokenizer.no_padding()
        encoded = self.tokenizer.encode_batch(texts)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        all_text_embeddings = []
        
        for e in encoded:
            chunks = self._chunk_tokens(e.ids, e.attention_mask, e.type_ids)
            
            input_ids = np.array([c["ids"] for c in chunks], dtype=np.int64)
            attention_mask = np.array([c["attention_mask"] for c in chunks], dtype=np.int64)
            token_type_ids = np.array([c["type_ids"] for c in chunks], dtype=np.int64)
            
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            ort_outs = self.session.run(None, ort_inputs)
            last_hidden_state = ort_outs[0]
            
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            
            pooled_embeddings = sum_embeddings / sum_mask
            
            norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
            normalized_embeddings = pooled_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            
            all_text_embeddings.append(normalized_embeddings)
            
        return all_text_embeddings
