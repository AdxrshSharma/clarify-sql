"""Model backends: all expose predict(question) -> (raw_text, tokens_in, tokens_out)."""
import os


class HFBackend:
    """Local Hugging Face model, optionally with a LoRA adapter (base vs fine-tuned)."""

    def __init__(self, model_name, adapter=None, load_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        kwargs = dict(torch_dtype=dtype, device_map="auto")
        if load_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def predict(self, question):
        from common import build_messages

        text = self.tok.apply_chat_template(build_messages(question), tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True), inputs["input_ids"].shape[1], len(gen)


class AnthropicBackend:
    """Prompted API model (needs ANTHROPIC_API_KEY). Same system prompt as the local models."""

    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def predict(self, question):
        from common import build_messages

        msgs = build_messages(question)
        resp = self.client.messages.create(model=self.model, max_tokens=400,
                                           system=msgs[0]["content"], messages=msgs[1:])
        return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


class GoldBackend:
    """Oracle that returns the gold answer. Used only to sanity-check the evaluation harness (must score 100%)."""

    def __init__(self, records):
        self.answers = {r["question"]: r["target"] for r in records}

    def predict(self, question):
        return self.answers[question], 0, 0


def make_backend(name, model=None, adapter=None, records=None, load_4bit=True):
    if name == "hf":
        return HFBackend(model, adapter, load_4bit)
    if name == "anthropic":
        return AnthropicBackend(model)
    if name == "gold":
        return GoldBackend(records)
    raise ValueError(f"Unknown backend: {name}")
