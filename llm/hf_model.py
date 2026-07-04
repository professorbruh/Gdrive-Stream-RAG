"""
HuggingFace LLM wrapper supporting both local inference (with 4-bit
quantization for GPU) and the HuggingFace Inference API.

Local mode:  Uses transformers + bitsandbytes for 4-bit quantized inference
             on your local GPU (12GB VRAM). Mistral-7B fits in ~6GB.

API mode:    Calls the HuggingFace Inference API (free tier available).
             No GPU required, but rate-limited.
"""

import config


class HuggingFaceModel:
    """
    Unified LLM interface for text generation.

    Automatically selects local GPU inference or API based on config.
    """

    def __init__(
        self,
        model_name: str = None,
        mode: str = None,
        hf_token: str = None,
    ):
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.mode = mode if mode is not None else config.LLM_MODE
        self.hf_token = hf_token or config.HF_TOKEN

        if self.mode == "hf_api":
            self._init_api()
        elif self.mode == "remote":
            self._init_remote()
        else:
            self._init_local()

    def _init_local(self):
        """Initializes the model locally with 4-bit quantization."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(f"  Loading local model: {self.model_name}")
        print(f"  4-bit quantization: {config.LLM_LOAD_IN_4BIT}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "token": self.hf_token,
            "device_map": "cuda",
            "torch_dtype": torch.float16,
        }

        if config.LLM_LOAD_IN_4BIT:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        print(f"  ✓ Model loaded on {self.model.device}")
        self._generate = self._generate_local

    def _init_api(self):
        """Initializes the HuggingFace Inference API client."""
        from huggingface_hub import InferenceClient

        print(f"  Using HF Inference API: {self.model_name}")
        self.client = InferenceClient(
            model=self.model_name,
            token=self.hf_token,
        )
        print(f"  Initialized HuggingFace API client")
        self._generate = self._generate_api

    def _init_remote(self):
        """Initializes the model in remote client mode."""
        print(f"  Initialized Remote LLM client -> {config.LLM_REMOTE_URL}")
        self._generate = self._generate_remote

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generates text from a prompt. Delegates to local or API backend."""
        max_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS
        return self._generate(prompt, max_tokens)

    def _generate_local(self, prompt: str, max_new_tokens: int) -> str:
        """Generates text using the local model."""
        import torch

        # Format as instruction for Mistral
        messages = [{"role": "user", "content": prompt}]

        # Use chat template if available, otherwise raw prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def _generate_api(self, prompt: str, max_new_tokens: int) -> str:
        """Generates text using the HuggingFace Inference API."""
        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
        )
        return response.strip()

    def _generate_remote(self, prompt: str, max_new_tokens: int) -> str:
        """Generates text by sending a POST request to a local GPU server (e.g. from AWS)."""
        import requests
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        headers = {}
        if config.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
        
        try:
            response = requests.post(config.LLM_REMOTE_URL, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json().get("text", "")
        except Exception as e:
            print(f"Error calling remote LLM server at {config.LLM_REMOTE_URL}: {e}")
            return (
                f"🤖 **Uh oh!** The Oracle Cloud server tried to call your local GPU at `{config.LLM_REMOTE_URL}`, "
                f"but it didn't pick up the phone! 📱💥\n\n"
                f"Did you turn off your PC, close the Ngrok tunnel, or are you just playing video games instead of hosting the LLM? "
                f"Admin pls!"
            )
    def ping(self) -> bool:
        """Checks if the LLM backend is responsive."""
        if self.mode == "local" or self.mode == "hf_api":
            return True
            
        import requests
        try:
            health_url = config.LLM_REMOTE_URL.replace("/generate", "/health")
            resp = requests.get(
                health_url, 
                timeout=5, 
                headers={"ngrok-skip-browser-warning": "true"}
            )
            return resp.status_code == 200
        except Exception:
            return False
