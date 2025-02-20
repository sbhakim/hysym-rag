# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict

logger = logging.getLogger(__name__)


class NeuralRetriever:
    """Enhanced Neural Retriever with supporting facts awareness for HotpotQA."""

    def __init__(self,
                 model_name: str,
                 use_quantization: bool = False,
                 max_context_length: int = 2048,
                 chunk_size: int = 512,
                 overlap: int = 128):
        """
        Initialize the enhanced neural retriever.

        Args:
            model_name: Name of the language model
            use_quantization: Whether to use 8-bit quantization
            max_context_length: Maximum context length
            chunk_size: Size of context chunks
            overlap: Overlap between chunks
        """
        print(f"Initializing Neural Retriever with model: {model_name}...")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
                load_in_8bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto"
            )

        # Initialize sentence transformer for semantic search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Configuration parameters
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Performance tracking
        self.stats = defaultdict(list)

        print(f"Model {model_name} loaded successfully!")

    def retrieve_answer(self,
                        context: str,
                        question: str,
                        symbolic_guidance: Optional[List[Dict]] = None,
                        supporting_facts: Optional[List[Tuple[str, int]]] = None,
                        query_complexity: Optional[float] = None
                        ) -> str:
        """
        Retrieve answer with supporting facts awareness.

        Args:
            context: Input context
            question: Question to answer
            symbolic_guidance: Optional symbolic rules
            supporting_facts: Optional supporting fact indices
            query_complexity: Optional query complexity score

        Returns:
            Generated answer as a string.
        """
        try:
            # Validate inputs
            if not isinstance(context, str) or not isinstance(question, str):
                raise ValueError("Context and question must be strings")

            # Process context into chunks
            context_chunks = self._chunk_context(context)

            # Prioritize chunks with supporting facts if available
            if supporting_facts:
                context_chunks = self._prioritize_supporting_facts(
                    context_chunks,
                    supporting_facts
                )

            # Encode question safely
            question_embedding = self._encode_safely(question)

            # Get relevant chunks using semantic search and optional guidance
            relevant_chunks = self._get_relevant_chunks(
                question_embedding,
                context_chunks,
                symbolic_guidance
            )

            # Prepare prompt with enhanced context
            prompt = self._create_prompt(
                question,
                relevant_chunks,
                symbolic_guidance
            )

            # Generate answer with proper error handling
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                no_repeat_ngram_size=3
            )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return self._post_process_response(response)

        except Exception as e:
            logger.error(f"Error in retrieve_answer: {str(e)}")
            return "Error retrieving answer."

    def _encode_safely(self, text: str) -> torch.Tensor:
        """Safely encode text with error handling."""
        try:
            return self.encoder.encode(text, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def _generate_response(self, prompt: str) -> str:
        """Generate response with proper error handling."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                no_repeat_ngram_size=3
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Error generating response."

    def _post_process_response(self, response: str) -> str:
        """Clean and validate the generated response."""
        if "Question:" in response:
            response = response.split("Answer:")[-1]
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        return response

    def _chunk_context(self, context: str) -> List[Dict]:
        """
        Split context into overlapping chunks with metadata.
        """
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for idx, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if current_length + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    'text': chunk_text,
                    'embedding': self.encoder.encode(chunk_text, convert_to_tensor=True),
                    'sentences': current_chunk.copy(),
                    'start_idx': idx - len(current_chunk),
                    'end_idx': idx - 1
                }
                chunks.append(chunk_data)
                overlap_sentences = current_chunk[-max(1, self.overlap // sentence_tokens):]
                current_chunk = overlap_sentences
                current_length = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                'text': chunk_text,
                'embedding': self.encoder.encode(chunk_text, convert_to_tensor=True),
                'sentences': current_chunk,
                'start_idx': len(sentences) - len(current_chunk),
                'end_idx': len(sentences) - 1
            }
            chunks.append(chunk_data)
        return chunks

    def _prioritize_supporting_facts(self,
                                     chunks: List[Dict],
                                     supporting_facts: List[Tuple[str, int]]
                                     ) -> List[Dict]:
        """
        Prioritize chunks containing supporting facts.
        """
        supporting_indices = {idx for _, idx in supporting_facts}
        for chunk in chunks:
            support_count = sum(
                1 for idx in range(chunk['start_idx'], chunk['end_idx'] + 1)
                if idx in supporting_indices
            )
            chunk['support_score'] = support_count / len(chunk['sentences'])
        return sorted(chunks, key=lambda x: x['support_score'], reverse=True)

    def _get_relevant_chunks(self,
                             question_embedding: torch.Tensor,
                             chunks: List[Dict],
                             symbolic_guidance: Optional[List[Dict]] = None
                             ) -> List[Dict]:
        """
        Get relevant chunks using semantic search and symbolic guidance.
        """
        similarities = []
        for chunk in chunks:
            sim = util.cos_sim(question_embedding, chunk['embedding']).item()
            if 'support_score' in chunk:
                sim += chunk['support_score'] * 0.3
            if symbolic_guidance:
                sim += self._calculate_guidance_boost(chunk, symbolic_guidance)
            similarities.append(sim)
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:]
        return [chunks[i] for i in top_indices]

    def _calculate_guidance_boost(self,
                                  chunk: Dict,
                                  guidance: List[Dict]) -> float:
        """
        Calculate score boost based on symbolic guidance.
        """
        boost = 0.0
        chunk_text = chunk['text'].lower()
        for guide in guidance:
            if guide['statement'].lower() in chunk_text:
                boost += guide.get('confidence', 0.5) * 0.2
        return min(boost, 0.3)

    def _create_prompt(self,
                       question: str,
                       chunks: List[Dict],
                       symbolic_guidance: Optional[List[Dict]] = None) -> str:
        """
        Create enhanced prompt with relevant context and guidance.
        """
        context_parts = [c['text'] for c in chunks]
        context = "\n".join(context_parts)
        guidance_text = ""
        if symbolic_guidance:
            guidance_statements = [g['statement'] for g in symbolic_guidance if g.get('confidence', 0) > 0.5]
            if guidance_statements:
                guidance_text = "\nRelevant information:\n- " + "\n- ".join(guidance_statements)
        prompt = f"Context:{guidance_text}\n{context}\n\nQuestion: {question}\nAnswer:"
        return prompt

    # End of file
