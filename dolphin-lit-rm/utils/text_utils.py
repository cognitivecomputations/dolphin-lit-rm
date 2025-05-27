import tiktoken
from loguru import logger

# Global tokenizer instance cache
_tokenizers = {}

def get_tokenizer(tokenizer_name: str = "gpt-4"):
    """
    Initializes and returns a tiktoken tokenizer.
    Caches tokenizer instances for efficiency.
    """
    if tokenizer_name not in _tokenizers:
        try:
            _tokenizers[tokenizer_name] = tiktoken.encoding_for_model(tokenizer_name)
        except KeyError:
            logger.warning(f"Model {tokenizer_name} not found for tiktoken. Trying cl100k_base.")
            try:
                _tokenizers[tokenizer_name] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.error(f"Failed to initialize tiktoken tokenizer {tokenizer_name} or cl100k_base: {e}")
                raise
    return _tokenizers[tokenizer_name]

def count_tokens(text: str, tokenizer_name: str = "gpt-4") -> int:
    """Counts the number of tokens in a text string using tiktoken."""
    if not text:
        return 0
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text, disallowed_special=()))

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Placeholder for more sophisticated cleaning if needed
    # e.g., unicode normalization, whitespace stripping
    if text is None:
        return ""
    text = text.strip()
    # Add more cleaning rules here, e.g., removing excessive newlines
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

# Add other text utilities as needed, e.g., regex cleaning, sentence splitting (though spacy is used for segmentation)