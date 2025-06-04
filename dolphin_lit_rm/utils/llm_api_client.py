import time
import requests
from typing import Dict, Any, Optional
from loguru import logger
import os

class LLMAPIClient:
    def __init__(
        self,
        api_base_url: str,
        api_key: Optional[str] = None,
        default_model_name: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ):
        self.api_base_url = api_base_url.rstrip('/')
        if api_key and api_key.startswith("ENV:"):
            env_var_name = api_key.split("ENV:")[1]
            self.api_key = os.getenv(env_var_name)
            if not self.api_key:
                logger.warning(f"Environment variable {env_var_name} for API key not found.")
        else:
            self.api_key = api_key
        
        self.default_model_name = default_model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def make_request(
        self,
        prompt: Optional[str] = None, # For completion
        messages: Optional[list] = None, # For chat completion
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        stop_sequences: Optional[list] = None,
        **kwargs # Other API specific params
    ) -> Dict[str, Any]:
        """
        Makes a request to an OpenAI-compatible completions or chat completions endpoint.
        Prioritizes `messages` for chat, falls back to `prompt` for completion if `messages` is None.
        """
        if not messages and not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        actual_model_name = model_name or self.default_model_name
        if not actual_model_name:
            raise ValueError("Model name must be provided either at init or per request.")

        payload = {
            "model": actual_model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # Determine endpoint and add prompt/messages
        if messages:
            # Prefer chat completions endpoint if messages are provided
            endpoint = f"{self.api_base_url}/chat/completions"
            if "qwen3" in actual_model_name.lower():
                messages = [{"role": "system", "content": "/no_think"}] + messages
            payload["messages"] = messages
        elif prompt:
            # Fallback to completions endpoint if only prompt is provided
            endpoint = f"{self.api_base_url}/completions"
            payload["prompt"] = prompt
        else: # Should be caught by earlier check
             raise ValueError("Either 'prompt' or 'messages' must be provided.")

        

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                
                response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"API request to {endpoint} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt == self.max_retries:
                    logger.error(f"API request failed after {self.max_retries + 1} attempts.")
                    raise
                time.sleep(2**attempt)  # Exponential backoff
        return {} # Should not be reached due to raise in loop

    def get_completion(self, response_json: Dict[str, Any], is_chat: bool) -> Optional[str]:
        """Extracts the text completion from the API response."""
        try:
            if is_chat: # Chat completion
                if response_json.get("choices") and response_json["choices"][0].get("message"):
                    return response_json["choices"][0]["message"].get("content", "").strip()
            else: # Legacy completion
                if response_json.get("choices"):
                    return response_json["choices"][0].get("text", "").strip()
            logger.warning(f"Could not extract completion from response: {response_json}")
            return None
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Error parsing completion from response: {e}. Response: {response_json}")
            return None