# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

class NeuralRetriever:
    def __init__(self, model_name):
        print(f"Initializing Neural Retriever with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto"
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model {model_name} loaded successfully!")

    def retrieve_answer(self, context, question, symbolic_guidance=None):
        """
        Generate an answer using symbolic guidance to influence neural attention.
        """
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        if symbolic_guidance:
            guidance_text = " ".join(symbolic_guidance)
            input_text = f"Guidance: {guidance_text}\n" + input_text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def encode(self, text):
        try:
            emb = self.encoder.encode(text, convert_to_tensor=True)
            if torch.cuda.is_available():
                emb = emb.to('cuda')
            return emb
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            raise

