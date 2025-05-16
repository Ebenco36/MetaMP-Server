from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Text2SQLModel:
    def __init__(self, checkpoint: str = "suriya7/t5-base-text-to-sql"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    def translate(self, nl_query: str, schema: str) -> str:
        prompt = f"-- SQL tables:\n{schema}\n-- Translate:\n{nl_query}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)