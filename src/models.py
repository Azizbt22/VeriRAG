# src/models.py

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from langchain_community.llms import HuggingFacePipeline


def get_llm(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Load a Hugging Face causal LLM and wrap it as a LangChain LLM.

    Designed to work on Colab / Kaggle with GPU.
    LoRA adapters can be loaded on top later.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)
