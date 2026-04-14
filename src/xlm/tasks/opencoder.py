from typing import Any, Dict
from transformers import AutoTokenizer
import re

def extract_code_block(response: str):
        """
        Extracts the content inside a markdown-style Python code block:
        
        Returns:
            prefix: everything before ```python
            code_block: content inside the code block
            suffix: everything after ```
        """
        # Split by ```python and ``` to extract the code block
        parts = re.split(r'```python\s*|\s*```', response)
        
        if len(parts) < 2:
            return "", response, ""  # No code block found
        
        prefix = parts[0]
        code_block = '```' + parts[1] + '\n```' if len(parts) > 1 else ""
        suffix = parts[2] if len(parts) > 2 else ""
        
        return prefix, code_block, suffix

def opencoder_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: Any,
    prompt_key: str,
    response_key: str,
) -> Dict[str, Any]:
    """Tokenize prompt and response."""
    prompt = example[prompt_key]
    response = example[response_key]
    if not isinstance(prompt, str):
        prompt_chat = list(prompt)
    else:
        prompt_chat = [{"role": "user", "content": prompt}]

    # string
    prompt_chat_str = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, tokenize=False
    )
    response_chat_str = response + tokenizer.eos_token

    prompt_ids_output = tokenizer.encode(prompt_chat_str, add_special_tokens=False)
    response_ids_output = tokenizer.encode(response_chat_str, add_special_tokens=False)
    prefix, code_block, suffix = extract_code_block(response_chat_str)

    return {
        "prompt_ids": prompt_ids_output,
        "token_ids": response_ids_output,
        "prefix": prefix,
        "middle": code_block,
        "suffix": suffix,
    }


def get_tokenizer(pretrained_model_name_or_path: str):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path,trust_remote_code=True)