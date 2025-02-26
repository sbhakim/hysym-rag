# src/reasoners/neural_retriever.py

import re
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
        self.logger = logger  # Added self.logger = logger

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

            # Standardize symbolic guidance format
            formatted_guidance = []
            if symbolic_guidance:
                self.logger.info(f"Processing {len(symbolic_guidance)} symbolic guidance items")

                # Define domains specific to your application
                domain_keywords = {
                    "deforestation": ["forest", "tree", "biodiversity", "erosion", "carbon", "soil"],
                    "climate": ["temperature", "global warming", "carbon dioxide", "climate change", "greenhouse"],
                    "water": ["water", "rain", "precipitation", "cycle", "drought", "flood"]
                }

                for rule in symbolic_guidance:
                    # Handle string rules
                    if isinstance(rule, str):
                        rule_text = rule.strip()
                        if not rule_text:
                            continue

                        # Heuristic to classify string type
                        domain_confidence = 0.7  # Default
                        for domain, keywords in domain_keywords.items():
                            if any(keyword in rule_text.lower() for keyword in keywords):
                                domain_confidence = 0.85  # Higher confidence for domain-related rules
                                break

                        formatted_guidance.append({
                            "response": rule_text,
                            "confidence": domain_confidence
                        })

                    # Handle dictionary rules
                    elif isinstance(rule, dict):
                        # Case 1: Already has response key
                        if "response" in rule and rule["response"]:
                            formatted_guidance.append({
                                "response": rule["response"],
                                "confidence": rule.get("confidence", 0.8)
                            })
                        else:
                            # Case 2: Extract response from other fields
                            response_text = None
                            for key in ["statement", "text", "source_text", "content"]:
                                if key in rule and rule[key]:
                                    response_text = rule[key]
                                    break

                            if response_text:
                                formatted_guidance.append({
                                    "response": response_text,
                                    "confidence": rule.get("confidence", 0.8)
                                })
                            else:
                                self.logger.warning(f"Could not extract response from rule: {rule}")

            # Log guidance statistics
            if formatted_guidance:
                self.logger.info(f"Successfully formatted {len(formatted_guidance)} guidance rules for retrieval")
                avg_confidence = sum(rule.get("confidence", 0) for rule in formatted_guidance) / len(formatted_guidance)
                self.logger.info(f"Average guidance confidence: {avg_confidence:.2f}")
            else:
                self.logger.info("No valid guidance rules found for retrieval")

            # Validate context before processing
            if isinstance(context, str) and context.strip():
                # Minimum length check (e.g., 10 characters)
                if len(context.strip()) < 10:
                    logger.warning("Context too short for meaningful processing")
                    return "No relevant context found."
                try:
                    context_chunks = self._chunk_context(context)
                    if not context_chunks:
                        return "No relevant context found."
                except Exception as e:
                    logger.error(f"Error chunking context: {str(e)}")
                    return "Error processing context."
            else:
                return "Invalid context provided."

            if supporting_facts:
                context_chunks = self._prioritize_supporting_facts(
                    context_chunks,
                    supporting_facts
                )

            question_embedding = self._encode_safely(question)

            relevant_chunks = self._get_relevant_chunks(
                question_embedding,
                context_chunks,
                formatted_guidance  # Use formatted_guidance for consistency
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

            # Use the enhanced prompt creation method
            prompt = self._create_enhanced_prompt(
                question,
                relevant_chunks,
                formatted_guidance  # Use formatted_guidance for consistency
            )

            inputs_pt = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)

            with logging.getLogger('transformers').handlers[0].lock:
                try:
                    outputs = self.model.generate(
                        **inputs_pt,
                        max_new_tokens=250,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except Exception as model_e:
                    logger.error(f"Error during model generation: {model_e}")
                    self.stats['errors'].append(str(model_e))
                    return "Error generating answer."

            # Check if outputs is empty
            if outputs is None or outputs.numel() == 0:
                logger.error("Model generated empty output")
                return "Error: Model failed to generate response."

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            result = self._ensure_complete_sentences(response)
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

    def _encode_safely(self, text: str) -> Optional[torch.Tensor]:
        """
        Enhanced safe encoding with robust fallback mechanism.
        Attempts standard encoding with retries; on final failure, falls back to token-based encoding.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Standard encoding attempt
                encoded = self.encoder.encode(text, convert_to_tensor=True)
                return encoded.to(self.device)
            except Exception as e:
                logger.warning(f"Encoding attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Final fallback: try simpler token-based encoding
                    try:
                        tokens = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
                        return tokens['input_ids'].to(self.device)
                    except Exception as e2:
                        logger.error(f"Final encoding attempt failed: {str(e2)}")
                        return None
            time.sleep(0.1 * (attempt + 1))
        return None

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
                    pad_token_id=self.tokenizer.eos_token_id
                )
            if outputs is None or outputs.numel() == 0:
                logger.error("Model generated empty output in _generate_response")
                return "Error: Model failed to generate response."
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
        """
        Enhanced context chunking with robust error handling and validation.
        """
        try:
            if not isinstance(context, str):
                logger.error("Invalid context type provided")
                return []
            if not context.strip():
                logger.warning("Empty context provided")
                return []
            if len(context.strip()) < 10:
                logger.warning("Context too short for meaningful processing")
                return []

            sentences = []
            for sent in re.split(r'[.!?]', context):
                cleaned = sent.strip()
                if cleaned and len(cleaned) > 3:
                    sentences.append(cleaned)
            if not sentences:
                logger.warning("No valid sentences extracted from context")
                return []

            chunks = []
            current_chunk = []
            current_length = 0
            overlap_buffer = []

            for idx, sentence in enumerate(self._process_batch(sentences)):
                try:
                    try:
                        sentence_tokens = len(self.tokenizer.encode(sentence))
                    except Exception as e:
                        logger.warning(f"Tokenization failed, using fallback: {e}")
                        sentence_tokens = len(sentence.split())

                    if current_length + sentence_tokens > self.chunk_size and current_chunk:
                        overlap_text = ' '.join(overlap_buffer[-self.overlap:]) if overlap_buffer else ''
                        chunk_text = ' '.join(current_chunk)
                        try:
                            chunk_embedding = self._encode_safely(chunk_text)
                        except Exception as e:
                            logger.error(f"Chunk embedding failed: {e}")
                            chunk_embedding = None

                        chunks.append({
                            'text': chunk_text,
                            'embedding': chunk_embedding,
                            'sentences': current_chunk.copy(),
                            'start_idx': max(0, idx - len(current_chunk)),
                            'end_idx': idx - 1,
                            'overlap_text': overlap_text
                        })
                        current_chunk = overlap_buffer[-self.overlap:] if overlap_buffer else []
                        current_length = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
                    current_chunk.append(sentence)
                    overlap_buffer.append(sentence)
                    current_length += sentence_tokens
                except Exception as e:
                    logger.error(f"Error processing sentence {idx}: {e}")
                    continue

            if current_chunk:
                try:
                    chunk_text = ' '.join(current_chunk)
                    chunk_embedding = self._encode_safely(chunk_text)
                    chunks.append({
                        'text': chunk_text,
                        'embedding': chunk_embedding,
                        'sentences': current_chunk,
                        'start_idx': len(sentences) - len(current_chunk),
                        'end_idx': len(sentences) - 1,
                        'overlap_text': ''
                    })
                except Exception as e:
                    logger.error(f"Error processing final chunk: {e}")
            return chunks

        except Exception as e:
            logger.error(f"Critical error in _chunk_context: {str(e)}")
            return []

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

            # Process each chunk with proper scoring
            for chunk in chunks:
                if 'embedding' not in chunk or chunk['embedding'] is None:
                    continue

                chunk_emb = chunk['embedding'].view(-1)
                sim = util.cos_sim(chunk_emb.unsqueeze(0), question_emb.unsqueeze(0)).item()

                # Add support score boost if available
                if 'support_score' in chunk:
                    sim += chunk['support_score'] * self.support_boost

                # Apply guidance boost with enhanced calculation
                if symbolic_guidance:
                    guidance_boost = self._calculate_guidance_boost(chunk, symbolic_guidance)
                    sim += guidance_boost

                # Store the chunk with its similarity score
                scored_chunks.append((sim, chunk))

            # Sort chunks by similarity score
            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            # Always include at least top 3 chunks to ensure sufficient context
            top_chunks = [chunk for _, chunk in scored_chunks[:min(3, len(scored_chunks))]]

            # Add additional chunks that meet the threshold
            threshold_chunks = [chunk for sim, chunk in scored_chunks[min(3, len(scored_chunks)):] if sim > 0.3]

            relevant_chunks = top_chunks + threshold_chunks

            # Fallback if no relevant chunks found
            if not relevant_chunks and scored_chunks:
                logger.warning("No relevant chunks found, using top chunk as fallback.")
                relevant_chunks = [scored_chunks[0][1]]

            return relevant_chunks[:max(3, self.top_k)]  # Ensure at least 3 chunks
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            return [chunks[0]] if chunks else []

    def _calculate_guidance_boost(self,
                                  chunk: Dict,
                                  guidance: List[Dict]) -> float:
        boost = 0.0
        chunk_text = chunk['text'].lower()

        if not guidance:
            return 0.0

        # Calculate matches based on different rule formats
        for guide in guidance:
            # Handle different rule formats
            if isinstance(guide, dict):
                # First try the specified statement key
                statement = guide.get(self.guidance_statement_key, '')

                # If not found, try alternative keys
                if not statement and 'response' in guide:
                    statement = guide.get('response', '')

                # If still not found, try more alternatives
                if not statement:
                    for key in ['text', 'content', 'source_text']:
                        if key in guide and guide[key]:
                            statement = guide[key]
                            break

                # Get confidence with fallback
                confidence = guide.get(self.guidance_confidence_key, 0.7)

                # Check for statement in chunk
                if statement and statement.lower() in chunk_text:
                    boost += confidence * self.guidance_boost_multiplier

                # Also check for partial keyword matches
                elif statement:
                    words = [w for w in statement.lower().split() if len(w) > 4]
                    matches = sum(1 for word in words if word in chunk_text)
                    if matches >= 2:  # At least 2 significant words match
                        boost += (confidence * 0.7) * self.guidance_boost_multiplier

            # Handle string guidance
            elif isinstance(guide, str) and guide.lower() in chunk_text:
                boost += 0.7 * self.guidance_boost_multiplier

        return min(boost, self.guidance_boost_limit)

    def _create_enhanced_prompt(self, question: str, chunks: List[Dict],
                                symbolic_guidance: Optional[List[Dict]] = None) -> str:
        """
        Create an enhanced prompt with better structure and guidance integration.
        """
        try:
            # Filter chunks and guidance for relevance to the question
            question_lower = question.lower()
            question_keywords = set(re.findall(r'\b\w{4,}\b', question_lower))

            # Only use chunks with sufficient relevance to query
            relevant_chunks = []
            for chunk in chunks:
                chunk_text = chunk.get('text', '').lower()
                chunk_keywords = set(re.findall(r'\b\w{4,}\b', chunk_text))
                common_keywords = question_keywords.intersection(chunk_keywords)

                # Only include chunk if it shares keywords with the question
                if len(common_keywords) > 0 or len(question_keywords) < 3:
                    relevant_chunks.append(chunk)

            # Use filtered chunks, or all chunks if filtering removed too much
            valid_chunks = relevant_chunks if len(relevant_chunks) >= 2 else chunks
            context_parts = []
            for chunk in valid_chunks:
                chunk_text = chunk['text'].strip()
                if chunk_text:
                    context_parts.append(chunk_text)

            # Deduplicate context parts
            unique_context = []
            seen_segments = set()
            for part in context_parts:
                if part not in seen_segments:
                    seen_segments.add(part)
                    unique_context.append(part)

            # Build context from unique parts
            context = "\n\n".join(unique_context[:3])  # Limit to 3 parts for conciseness

            # Filter symbolic guidance for relevance to query
            guidance_text = ""
            if symbolic_guidance:
                valid_statements = []
                for guide in symbolic_guidance:
                    if isinstance(guide, dict):
                        statement = guide.get('response', '')
                        if not statement and guide.get('text'):
                            statement = guide.get('text')

                        # Check if statement is relevant to the question
                        if statement:
                            statement_lower = statement.lower()
                            statement_keywords = set(re.findall(r'\b\w{4,}\b', statement_lower))
                            common_keywords = question_keywords.intersection(statement_keywords)

                            # Only include if it shares keywords with the question
                            if len(common_keywords) > 0:
                                valid_statements.append(statement)

                if valid_statements:
                    guidance_text = "\nRelevant background:\n- " + "\n- ".join(valid_statements[:3])

            # Create the final prompt
            prompt = (
                f"Context:{guidance_text}\n\n"
                f"{context}\n\n"
                f"Question: {question}\n"
                f"Using the provided context, answer the question accurately and concisely.\n"
                f"Answer: "
            )
            return prompt
        except Exception as e:
            logger.error(f"Error creating enhanced prompt: {e}")
            return f"Question: {question}\nAnswer: "

    def _create_fallback_prompt(self, question: str) -> str:
        """
        Create a minimal prompt when normal prompt creation fails.
        """
        return f"Question: {question}\nAnswer: "

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text ends with complete sentences."""
        if not text:
            return text

        # If text doesn't end with sentence-ending punctuation, try to find the last complete sentence
        if not text.rstrip().endswith(('.', '!', '?')):
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > len(text) * 0.7:  # Only truncate if we've got most of the content
                return text[:last_period + 1]

        return text