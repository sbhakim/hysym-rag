# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

def is_causal_query(question):
    """
    Determines if the question likely requires causal reasoning.
    """
    causal_keywords = ['cause', 'result', 'lead to', 'because']
    return any(keyword in question.lower() for keyword in causal_keywords)

class NeuralRetriever:
    def __init__(self, model_name, use_quantization=False):
        print(f"Initializing Neural Retriever with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_quantization:
            # Enable 8-bit quantization if supported (requires bitsandbytes)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto", load_in_8bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto"
            )
        # Load a SentenceTransformer for encoding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model {model_name} loaded successfully!")

    def retrieve_answer(self, context, question, symbolic_guidance=None):
        """
        Generate an answer using symbolic guidance and chain-of-thought prompting
        for causal queries.
        """
        if is_causal_query(question):
            cot_prompt = "Step-by-step causal reasoning:"
            input_text = f"{cot_prompt}\nContext: {context}\nQuestion: {question}\nAnswer:"
        else:
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        if symbolic_guidance:
            guidance_text = " ".join(symbolic_guidance)
            input_text = f"Guidance: {guidance_text}\n" + input_text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def retrieve_answers_batched(self, contexts, questions, symbolic_guidances=None):
        """
        Batch version of retrieve_answer.
        contexts: list of context strings
        questions: list of question strings
        symbolic_guidances: optional list of guidance lists (each guidance is a list of strings)
        """
        batch_inputs = []
        for i in range(len(questions)):
            guidance = symbolic_guidances[i] if symbolic_guidances and i < len(symbolic_guidances) else None
            if is_causal_query(questions[i]):
                cot_prompt = "Step-by-step causal reasoning:"
                input_text = f"{cot_prompt}\nContext: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:"
            else:
                input_text = f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:"
            if guidance:
                guidance_text = " ".join(guidance)
                input_text = f"Guidance: {guidance_text}\n" + input_text
            batch_inputs.append(input_text)
        inputs = self.tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        decoded = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return decoded

    def encode(self, text):
        """
        Returns the neural embedding for the provided text.
        """
        try:
            emb = self.encoder.encode(text, convert_to_tensor=True)
            if torch.cuda.is_available():
                emb = emb.to('cuda')
            return emb
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            raise
