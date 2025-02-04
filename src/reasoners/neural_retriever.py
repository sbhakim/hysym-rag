# src/reasoners/neural_retriever.py

from transformers import pipeline

class NeuralRetriever:
    def __init__(self):
        self.model = pipeline("question-answering", model="distilbert-base-uncased")

    def retrieve_answer(self, context, question):
        result = self.model(question=question, context=context)
        return result["answer"]
