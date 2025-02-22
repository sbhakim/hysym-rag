# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
from sentence_transformers import SentenceTransformer, util
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
from src.utils.progress import tqdm, ProgressManager
from src.utils.device_manager import DeviceManager
import time

logger = logging.getLogger(__name__)


class NeuralRetriever:
    """Enhanced Neural Retriever with supporting facts awareness for HotpotQA."""

    def __init__(self,
                 model_name: str,
                 use_quantization: bool = False,
                 max_context_length: int = 2048,
                 chunk_size: int = 512,
                 overlap: int = 128,
                 top_k: int = 3,
                 support_boost: float = 0.3,
                 guidance_boost_limit: float = 0.3,
                 guidance_boost_multiplier: float = 0.2,
                 guidance_statement_key: str = 'statement',
                 guidance_confidence_key: str = 'confidence',
                 device: Optional[torch.device] = None):

        print(f"Initializing Neural Retriever with model: {model_name}...")
        if device is None:
            device = DeviceManager.get_device()
        self.device = device

        ProgressManager.disable_progress()
        transformers_logging.set_verbosity_error()

        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            load_in_8bit=use_quantization
        )

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.support_boost = support_boost
        self.guidance_boost_limit = guidance_boost_limit
        self.guidance_boost_multiplier = guidance_boost_multiplier
        self.guidance_statement_key = guidance_statement_key
        self.guidance_confidence_key = guidance_confidence_key

        self.stats = defaultdict(list)

        print(f"Model {model_name} loaded successfully!")

    def _process_batch(self, inputs, disable_progress=True):
        return tqdm(inputs, disable=disable_progress)

    def retrieve_answer(self,
                        context: str,
                        question: str,
                        symbolic_guidance: Optional[List[Dict]] = None,
                        supporting_facts: Optional[List[Tuple[str, int]]] = None,
                        query_complexity: Optional[float] = None
                        ) -> str:

        start_time = time.time()
        try:
            if not isinstance(context, str) or not isinstance(question, str):
                raise ValueError("Context and question must be strings")

            if isinstance(symbolic_guidance, str):
                symbolic_guidance = [{"response": symbolic_guidance}]
            elif symbolic_guidance and not isinstance(symbolic_guidance, list):
                symbolic_guidance = [{"response": str(symbolic_guidance)}]
            if symbolic_guidance:
                symbolic_guidance = [
                    rule if isinstance(rule, dict) else {"response": str(rule)}
                    for rule in symbolic_guidance
                ]

            context_chunks = self._chunk_context(context)
            if not context_chunks:
                logger.warning("No context chunks created; falling back to full context.")
                context_chunks = [{
                    'text': context,
                    'embedding': self._encode_safely(context),
                    'sentences': context.split('.'),
                    'start_idx': 0,
                    'end_idx': len(context.split('.')) - 1
                }]

            if supporting_facts:
                context_chunks = self._prioritize_supporting_facts(
                    context_chunks,
                    supporting_facts
                )

            question_embedding = self._encode_safely(question)

            relevant_chunks = self._get_relevant_chunks(
                question_embedding,
                context_chunks,
                symbolic_guidance
            )
            if not relevant_chunks:
                logger.warning("No relevant chunks found; using full context as fallback.")
                relevant_chunks = [{
                    'text': context,
                    'embedding': self._encode_safely(context),
                    'sentences': context.split('.'),
                    'start_idx': 0,
                    'end_idx': len(context.split('.')) - 1
                }]

            prompt = self._create_prompt(
                question,
                relevant_chunks,
                symbolic_guidance
            )


            inputs_pt = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)


            with logging.getLogger('transformers').handlers[0].lock:
                try: # **NEW - Error handling for model.generate()**
                    outputs = self.model.generate(
                        **inputs_pt,
                        max_new_tokens=150,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.eos_token_id # **NEW - Pad token**
                    )
                except Exception as model_e:
                    logger.error(f"Error during model generation: {model_e}")
                    self.stats['errors'].append(str(model_e))
                    return "Error generating answer."


            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            result = self._post_process_response(response)
            self.stats['processing_times'].append(time.time() - start_time)
            return result

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU memory error in retrieve_answer: {str(e)}")
            self.stats['errors'].append(str(e))
            return "Error: GPU out of memory."
        except Exception as e:
            logger.error(f"Error in retrieve_answer: {str(e)}")
            self.stats['errors'].append(str(e))
            return "Error retrieving answer."


    def _encode_safely(self, text: str) -> torch.Tensor:
        try:
            with logging.getLogger('sentence_transformers').handlers[0].lock:
                return self.encoder.encode(text, convert_to_tensor=True)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU memory error during encoding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during encoding: {str(e)}")
            raise

    def _generate_response(self, prompt: str) -> str:
        try:
            inputs_pt = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)

            with logging.getLogger('transformers').handlers[0].lock:
                outputs = self.model.generate(
                    **inputs_pt,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id # **NEW - Pad token here as well, for consistency**
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU memory error during generation: {str(e)}")
            return "Error: GPU out of memory."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Error generating response."

    def _post_process_response(self, response: str) -> str:
        if "Question:" in response:
            response = response.split("Answer:")[-1]
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        return response

    def _chunk_context(self, context: str) -> List[Dict]:
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        if not sentences: # **NEW - Handle empty sentences list**
            return chunks


        try:
            for idx, sentence in enumerate(self._process_batch(sentences)):
                sentence_tokens = len(self.tokenizer.encode(sentence))
                if current_length + sentence_tokens > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    with logging.getLogger('sentence_transformers').handlers[0].lock:
                        chunk_data = {
                            'text': chunk_text,
                            'embedding': self.encoder.encode(chunk_text, convert_to_tensor=True),
                            'sentences': current_chunk.copy(),
                            'start_idx': idx - len(current_chunk),
                            'end_idx': idx - 1
                        }
                    chunks.append(chunk_data)

                    overlap_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk[-1:])
                    overlap_sentences = current_chunk[-1:]

                    while overlap_tokens < self.overlap and len(overlap_sentences) < len(current_chunk):
                        overlap_sentences.insert(0, current_chunk[-len(overlap_sentences)-1])
                        overlap_tokens += len(self.tokenizer.encode(overlap_sentences[0]))

                    current_chunk = overlap_sentences
                    current_length = overlap_tokens

                current_chunk.append(sentence)
                current_length += sentence_tokens

            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                with logging.getLogger('sentence_transformers').handlers[0].lock:
                    chunk_data = {
                        'text': chunk_text,
                        'embedding': self.encoder.encode(chunk_text, convert_to_tensor=True),
                        'sentences': current_chunk,
                        'start_idx': len(sentences) - len(current_chunk),
                        'end_idx': len(sentences) - 1
                    }
                chunks.append(chunk_data)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return chunks

    def _prioritize_supporting_facts(self,
                                     chunks: List[Dict],
                                     supporting_facts: List[Tuple[str, int]]
                                     ) -> List[Dict]:
        supporting_indices = {idx for _, idx in supporting_facts}
        for chunk in chunks:
            support_count = sum(
                1 for idx in range(chunk['start_idx'], chunk['end_idx'] + 1)
                if idx in supporting_indices
            )
            chunk['support_score'] = support_count / len(chunk['sentences']) if len(chunk['sentences']) > 0 else 0
        return sorted(chunks, key=lambda x: x.get('support_score', 0), reverse=True)

    def _get_relevant_chunks(self,
                             question_embedding: torch.Tensor,
                             chunks: List[Dict],
                             symbolic_guidance: Optional[List[Dict]] = None) -> List[Dict]:
        if not chunks:
            logger.warning("No chunks available for processing")
            return []

        try:
            scored_chunks = []
            question_emb = question_embedding.view(-1)
            for chunk in chunks:
                chunk_emb = chunk['embedding'].view(-1)
                sim = util.cos_sim(chunk_emb.unsqueeze(0), question_emb.unsqueeze(0)).item()

                if 'support_score' in chunk:
                    sim += chunk['support_score'] * self.support_boost

                if symbolic_guidance:
                    guidance_boost = self._calculate_guidance_boost(chunk, symbolic_guidance)
                    sim += guidance_boost

                scored_chunks.append((sim, chunk))

            sorted_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in sorted_chunks[:max(1, self.top_k)]]
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            return [chunks[0]] if chunks else [] # **Ensure at least one chunk as fallback**


    def _calculate_guidance_boost(self,
                                  chunk: Dict,
                                  guidance: List[Dict]) -> float:
        boost = 0.0
        chunk_text = chunk['text'].lower()
        for guide in guidance:
            if guide.get(self.guidance_statement_key, '').lower() in chunk_text:
                boost += guide.get(self.guidance_confidence_key, 0.5) * self.guidance_boost_multiplier
        return min(boost, self.guidance_boost_limit)

    def _create_prompt(self,
                       question: str,
                       chunks: List[Dict],
                       symbolic_guidance: Optional[List[Dict]] = None) -> str:
        context_parts = [c['text'] for c in chunks]
        context = "\n".join(context_parts)
        guidance_text = ""
        if symbolic_guidance:
            guidance_statements = [g[self.guidance_statement_key] for g in symbolic_guidance if g.get(self.guidance_confidence_key, 0) > 0.5 and self.guidance_statement_key in g]
            if guidance_statements:
                guidance_text = "\nRelevant information:\n- " + "\n- ".join(guidance_statements)
        prompt = f"Context:{guidance_text}\n{context}\n\nQuestion: {question}\nAnswer:"
        return prompt