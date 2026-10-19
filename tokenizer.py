#!/usr/bin/env python3
"""
Prompt formatting for the model families evaluated in the paper.

This module provides a unified interface for creating properly formatted prompts
for the two model families used here (Llama-3.1 and Qwen2.5) while ensuring
all functions return strings consistently.
"""

from typing import Optional, List
from transformers import PreTrainedTokenizer, AutoTokenizer


def detect_model_family(tokenizer: PreTrainedTokenizer) -> str:
    """
    Detect the model family from the tokenizer.
    
    Args:
        tokenizer: The tokenizer to analyze
        
    Returns:
        str: Model family ('llama', 'qwen', 'unknown')
    """
    model_name = getattr(tokenizer, "name_or_path", "") or ""
    model_name_lower = model_name.lower()

    if any(name in model_name_lower for name in ["llama", "meta-llama"]):
        return "llama"
    elif any(name in model_name_lower for name in ["qwen"]):
        return "qwen"
    else:
        return "unknown"


def create_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_chat_format: bool = True,
) -> str:
    """
    Create a properly formatted chat prompt for any model family.
    
    Args:
        tokenizer: The tokenizer for the model
        question: The user's question/instruction
        answer: The model's answer (optional, for training data)
        system_prompt: System prompt (optional)
        use_chat_format: Whether to use chat format (default: True)
        
    Returns:
        str: Formatted prompt string
    """
    if not use_chat_format:
        # Base model format
        if answer is not None:
            return f"Input: {question}\nResponse: {answer}"
        else:
            return f"Input: {question}\nResponse:"

    model_family = detect_model_family(tokenizer)

    if model_family == "llama":
        return _extract_llama_template(
            tokenizer=tokenizer,
            question=question,
            answer=answer,
            system_prompt=system_prompt,
        )
    elif model_family == "qwen":
        return _extract_qwen_template(
            tokenizer=tokenizer,
            question=question,
            answer=answer,
            system_prompt=system_prompt,
        )
    else:
        # Fallback to simple format for unknown models
        print(f"Warning: Unknown model family '{model_family}', using fallback format")
        if answer is not None:
            return f"Question: {question}\nAnswer: {answer}"
        else:
            return f"Question: {question}\nAnswer:"


def create_prompt_pairs(
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    positive_answers: List[str],
    negative_answers: List[str],
    system_prompt: Optional[str] = None,
    use_chat_format: bool = True,
) -> List[tuple[str, str]]:
    """
    Create pairs of positive and negative prompts for training.
    
    Args:
        tokenizer: The tokenizer for the model
        questions: List of questions/instructions
        positive_answers: List of positive answers
        negative_answers: List of negative answers
        system_prompt: System prompt (optional)
        use_chat_format: Whether to use chat format
        
    Returns:
        List of (positive_prompt, negative_prompt) tuples
    """
    pairs = []

    for question, pos_answer, neg_answer in zip(
        questions, positive_answers, negative_answers
    ):
        pos_prompt = create_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            answer=pos_answer,
            system_prompt=system_prompt,
            use_chat_format=use_chat_format,
        )

        neg_prompt = create_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            answer=neg_answer,
            system_prompt=system_prompt,
            use_chat_format=use_chat_format,
        )

        pairs.append((pos_prompt, neg_prompt))

    return pairs


def extract_chat_template(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_chat_format: bool = True,
) -> str:
    """
    Extract the chat template string without tokenizing it.
    This is an alias for create_chat_prompt for consistency.
    
    Args:
        tokenizer: The tokenizer for the model
        question: The user's question/instruction
        answer: The model's answer (optional, for training data)
        system_prompt: System prompt (optional)
        use_chat_format: Whether to use chat format (default: True)
        
    Returns:
        str: The formatted chat template string
    """
    return create_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        answer=answer,
        system_prompt=system_prompt,
        use_chat_format=use_chat_format,
    )


def tokenize_chat_template(
    tokenizer: PreTrainedTokenizer,
    template_string: str,
    add_special_tokens: bool = False,
) -> List[int]:
    """
    Tokenize a chat template string.
    
    Args:
        tokenizer: The tokenizer for the model
        template_string: The formatted chat template string
        add_special_tokens: Whether to add special tokens (default: False)
        
    Returns:
        List[int]: Token IDs
    """
    return tokenizer.encode(template_string, add_special_tokens=add_special_tokens)


def extract_and_tokenize_chat_template(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_chat_format: bool = True,
    add_special_tokens: bool = False,
) -> tuple[str, List[int]]:
    """
    Extract chat template and tokenize it in one step.
    
    Args:
        tokenizer: The tokenizer for the model
        question: The user's question/instruction
        answer: The model's answer (optional, for training data)
        system_prompt: System prompt (optional)
        use_chat_format: Whether to use chat format (default: True)
        add_special_tokens: Whether to add special tokens (default: False)
        
    Returns:
        tuple[str, List[int]]: (template_string, token_ids)
    """
    template_string = extract_chat_template(
        tokenizer=tokenizer,
        question=question,
        answer=answer,
        system_prompt=system_prompt,
        use_chat_format=use_chat_format,
    )

    token_ids = tokenize_chat_template(
        tokenizer=tokenizer,
        template_string=template_string,
        add_special_tokens=add_special_tokens,
    )

    return template_string, token_ids


def _extract_llama_template(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Extract Llama-3-formatted template string."""
    pieces = []

    # pieces.append(
    #     f"<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023.\n\nYou are a helpful assistant.<|eot_id|>"
    # )
    pieces.append(
        f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
    )
    pieces.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n{question.strip()}<|eot_id|>"
    )
    if answer is not None:
        pieces.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}")
    else:
        pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(pieces)


def _extract_qwen_template(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Extract Qwen-formatted template string."""
    bos = getattr(tokenizer, "bos_token", None) or ""
    pieces = [bos]
    # if system_prompt:
    #     pieces.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n")
    pieces.append(f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n")
    pieces.append(f"<|im_start|>user\n{question.strip()}<|im_end|>\n")

    if answer is not None:
        pieces.append(f"<|im_start|>assistant\n{answer.strip()}")
    else:
        pieces.append("<|im_start|>assistant\n")

    return "".join(pieces)
