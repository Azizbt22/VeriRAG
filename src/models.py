from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def get_llm(model_name="meta-llama/Llama-3.2-3B-Instruct", temperature=0.2, max_new_tokens=512):
    print(f"Loading: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True if temperature > 0 else False, top_p=0.9, repetition_penalty=1.1)
    print(f"✓ Loaded: {model_name}")
    class LLMWrapper:
        def __init__(self, pipe):
            self.pipe = pipe
        def invoke(self, prompt):
            result = self.pipe(prompt, return_full_text=False)
            class Response:
                def __init__(self, text):
                    self.content = text
            return Response(result[0]['generated_text'])
    return LLMWrapper(pipe)
