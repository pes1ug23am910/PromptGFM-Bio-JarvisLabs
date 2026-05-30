"""
Biomedical prompt encoder using pretrained language models.

Uses BioBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
to encode disease descriptions and phenotype lists into embedding vectors.
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptEncoder(nn.Module):
    """
    Biomedical prompt encoder using BioBERT.
    
    Encodes disease descriptions and phenotype information into dense embedding vectors
    that can be used to condition graph neural networks.
    
    Args:
        model_name: Pretrained model name from HuggingFace
        pooling: Pooling strategy - 'cls', 'mean', or 'max'
        max_length: Maximum sequence length for tokenization
        freeze: Whether to freeze BERT parameters
        dropout_prob: Dropout probability for the output layer
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        pooling: str = 'cls',
        max_length: int = 128,
        freeze: bool = False,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        logger.info(f"Initializing PromptEncoder with model: {model_name}")
        
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.freeze = freeze
        
        # Load tokenizer and model
        # Note: resume_download was removed in transformers>=4.45; local_files_only
        # is set via TRANSFORMERS_OFFLINE env var by the notebook before training starts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze BERT parameters if requested
        if freeze:
            logger.info("Freezing BERT parameters")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Optional projection and dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        logger.info(f"PromptEncoder initialized - hidden_size: {self.hidden_size}, "
                   f"pooling: {pooling}, max_length: {max_length}")
    
    def create_prompt(
        self,
        disease_name: str,
        phenotypes: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a structured prompt from disease information.
        
        Args:
            disease_name: Name of the disease
            phenotypes: List of associated phenotype terms
            description: Optional disease description
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Disease: {disease_name}."]
        
        if phenotypes and len(phenotypes) > 0:
            # Limit phenotypes to avoid exceeding max_length
            phenotype_str = ", ".join(phenotypes[:10])  # Take first 10
            prompt_parts.append(f"Phenotypes: {phenotype_str}.")
        
        if description:
            prompt_parts.append(f"Description: {description}.")
        
        prompt_parts.append("Associated genes:")
        
        return " ".join(prompt_parts)
    
    def forward(
        self,
        texts: Union[str, List[str]],
        return_attention_mask: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode text prompts into embedding vectors.
        
        Args:
            texts: Single text string or list of text strings
            return_attention_mask: Whether to return attention masks
            
        Returns:
            embeddings: [batch_size, hidden_size] tensor
            attention_mask (optional): [batch_size, seq_len] tensor
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass through BERT
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool the outputs
        embeddings = self._pool_outputs(
            outputs.last_hidden_state,
            attention_mask
        )
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings
    
    def _pool_outputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token-level hidden states into a single vector per sequence.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled: [batch_size, hidden_size]
        """
        if self.pooling == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'mean':
            # Mean pooling over non-padding tokens
            # Expand attention_mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        
        elif self.pooling == 'max':
            # Max pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.clone()
            hidden_states[mask_expanded == 0] = -1e9  # Set padding to very negative
            return torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def encode_batch(
        self,
        disease_names: List[str],
        phenotypes_list: Optional[List[List[str]]] = None,
        descriptions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode a batch of disease information.
        
        Args:
            disease_names: List of disease names
            phenotypes_list: List of phenotype lists (one per disease)
            descriptions: List of disease descriptions
            
        Returns:
            embeddings: [batch_size, hidden_size] tensor
        """
        batch_size = len(disease_names)
        
        # Create prompts for each disease
        prompts = []
        for i in range(batch_size):
            phenotypes = phenotypes_list[i] if phenotypes_list else None
            description = descriptions[i] if descriptions else None
            prompt = self.create_prompt(disease_names[i], phenotypes, description)
            prompts.append(prompt)
        
        # Encode all prompts
        return self.forward(prompts)
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of output embeddings."""
        return self.hidden_size
    
    def unfreeze(self):
        """Unfreeze all parameters for fine-tuning."""
        logger.info("Unfreezing BERT parameters")
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze = False


def test_prompt_encoder():
    """Test the prompt encoder with sample data."""
    logger.info("Testing PromptEncoder...")
    
    # Create encoder
    encoder = PromptEncoder(
        pooling='cls',
        max_length=128,
        freeze=False
    )
    
    # Test single prompt
    disease_name = "Angelman syndrome"
    phenotypes = ["seizures", "developmental delay", "speech impairment", "ataxia"]
    prompt = encoder.create_prompt(disease_name, phenotypes)
    logger.info(f"Sample prompt: {prompt}")
    
    # Test encoding
    embeddings = encoder([prompt])
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Embedding dim: {encoder.embedding_dim}")
    
    # Test batch encoding
    disease_names = ["Angelman syndrome", "Rett syndrome", "Fragile X syndrome"]
    phenotypes_list = [
        ["seizures", "developmental delay", "speech impairment"],
        ["hand stereotypies", "regression", "seizures"],
        ["intellectual disability", "anxiety", "autism"]
    ]
    
    batch_embeddings = encoder.encode_batch(disease_names, phenotypes_list)
    logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Test different pooling strategies
    for pooling in ['cls', 'mean', 'max']:
        encoder_test = PromptEncoder(pooling=pooling, max_length=64)
        emb = encoder_test([prompt])
        logger.info(f"Pooling={pooling}: shape {emb.shape}")
    
    logger.info("✓ PromptEncoder tests passed!")
    return encoder


if __name__ == "__main__":
    test_prompt_encoder()
