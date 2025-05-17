# src/reasoners/neural_retriever.py

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
from sentence_transformers import SentenceTransformer, util
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from src.utils.progress import tqdm, ProgressManager
from src.utils.device_manager import DeviceManager
import time
import hashlib
import spacy


try:
    from dateutil import parser as date_parser # Import the parser module
except ImportError:
    # Log a warning or error if dateutil is not installed, as it's used for date parsing
    logging.getLogger(__name__).warning(
        "The 'python-dateutil' library is not installed. "
        "Date parsing capabilities in NeuralRetriever will be limited or fail. "
        "Please install it by running: pip install python-dateutil"
    )
    date_parser = None # Set to None so attempts to use it can be handled gracefully


logger = logging.getLogger(__name__)

class NeuralRetriever:
    """Enhanced Neural Retriever with supporting facts awareness for HotpotQA and guidance caching."""

    def __init__(self,
                 model_name: str,
                 use_quantization: bool = True,
                 max_context_length: int = 2048,
                 chunk_size: int = 512,
                 overlap: int = 128,
                 top_k: int = 3,
                 support_boost: float = 0.3,
                 guidance_boost_limit: float = 0.3,
                 guidance_boost_multiplier: float = 0.2,
                 guidance_statement_key: str = 'response',
                 guidance_confidence_key: str = 'confidence',
                 device: Optional[torch.device] = None):
        """
        Initialize the Neural Retriever with model and configuration.
        [Updated May 16, 2025]: Enabled spaCy dependency parser to support noun_chunks and improved error handling.

        Args:
            model_name: Name of the transformer model to use.
            use_quantization: Whether to use 8-bit quantization (default: True).
            max_context_length: Maximum context length for the model.
            chunk_size: Size of context chunks.
            overlap: Overlap between chunks.
            top_k: Number of top chunks to retrieve.
            support_boost: Boost for chunks containing supporting facts.
            guidance_boost_limit: Maximum boost from symbolic guidance.
            guidance_boost_multiplier: Multiplier for guidance confidence.
            guidance_statement_key: Key for guidance statements.
            guidance_confidence_key: Key for guidance confidence.
            device: Device to run the model on.
        """
        print(f"Initializing Neural Retriever with model: {model_name}...")
        if device is None:
            device = DeviceManager.get_device()
        self.device = device
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)  # Use DEBUG for development

        ProgressManager.disable_progress()  # Ensure tqdm is off by default if needed
        transformers_logging.set_verbosity_error()  # Reduce transformer logging noise

        # Set logging level for libraries known to be verbose
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)  # Keep warnings for ST

        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set pad token if it's missing (common for some models like Llama)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("Tokenizer pad_token set to eos_token.")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise  # Critical error, re-raise

        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # Automatically distribute across devices if possible
                torch_dtype="auto",  # Use appropriate dtype
                load_in_8bit=use_quantization,
                # trust_remote_code=True  # May be needed for some models
            )
            # Ensure model uses the same pad token ID as tokenizer
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Model {model_name} loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise  # Critical error, re-raise

        try:
            # Load sentence encoder
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.logger.info("SentenceTransformer encoder loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer encoder: {e}")
            self.encoder = None  # Allow operation without encoder, but log error

        try:
            # [Updated May 16, 2025]: Enabled dependency parser for noun_chunks
            self.nlp = spacy.load('en_core_web_sm', disable=['lemmatizer'])
            self.logger.info("spaCy model 'en_core_web_sm' loaded successfully for NER and parsing.")
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {e}. Span parsing will be limited.")
            self.nlp = None

        # Store configuration parameters
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.support_boost = support_boost
        self.guidance_boost_limit = guidance_boost_limit
        self.guidance_boost_multiplier = guidance_boost_multiplier
        self.guidance_statement_key = guidance_statement_key
        self.guidance_confidence_key = guidance_confidence_key

        # Initialize caches and stats
        self.stats = defaultdict(list)
        self.guidance_cache = {}
        self.context_chunk_cache = {}
        self._last_scored_chunks = []  # Initialize attribute to store scored chunks for fallback

    def _process_batch(self, inputs, disable_progress=True):
        """Wrapper for tqdm progress bar, respecting global settings."""
        # This method seems unused in the current flow, but kept for potential future use
        return tqdm(inputs, disable=disable_progress or not ProgressManager.enabled)

    def retrieve_answer(self,
                        context: str,
                        question: str,
                        symbolic_guidance: Optional[List[Dict]] = None,
                        supporting_facts: Optional[List[Tuple[str, int]]] = None,
                        query_complexity: Optional[float] = None,  # Keep for potential future use
                        dataset_type: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Retrieves an answer from the context given a question, using symbolic guidance.

        Args:
            context: Context text.
            question: Query text.
            symbolic_guidance: List of guidance dictionaries.
            supporting_facts: List of supporting facts (title, sentence index) for HotpotQA.
            query_complexity: Complexity score of the query (currently unused here).
            dataset_type: Dataset type ('hotpotqa' or 'drop').

        Returns:
            For DROP dataset: A dictionary representing a DROP answer object.
            For other datasets: Answer string, or an error message string starting with "Error:".
        """
        start_time = time.time()
        # Use hash of question for consistent logging ID if query_id not passed from above
        query_id_for_log = hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]
        self.logger.debug(f"[NR Start QID:{query_id_for_log}] Context len: {len(context)}, Question: '{question[:50]}...', Dataset: {dataset_type}")

        # --- Input Validation ---
        try:
            if not isinstance(context, str) or not isinstance(question, str):
                # Log the types for easier debugging
                self.logger.error(f"[NR Error QID:{query_id_for_log}] Invalid input types: Context={type(context).__name__}, Question={type(question).__name__}")
                # Consistent error format
                if dataset_type and dataset_type.lower() == 'drop':
                    return self._empty_drop()
                return "Error: Context and question must be strings."
            if not context.strip() or not question.strip():
                self.logger.error(f"[NR Error QID:{query_id_for_log}] Empty context or question.")
                if dataset_type and dataset_type.lower() == 'drop':
                    return self._empty_drop()
                return "Error: Empty context or question provided."
            if self.encoder is None:
                # Critical component missing
                self.logger.error(f"[NR Error QID:{query_id_for_log}] Sentence encoder not available. Cannot proceed.")
                if dataset_type and dataset_type.lower() == 'drop':
                    return self._empty_drop()
                return "Error: Neural Retriever embedding component failed to initialize."
        except Exception as val_err:
            # Catch potential errors during validation itself
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Input validation failed: {val_err}")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return f"Error: Input validation failed: {val_err}"

        # --- Guidance Processing ---
        formatted_guidance = []
        if symbolic_guidance:
            try:
                # Use content hash for caching
                guidance_content_str = json.dumps(symbolic_guidance, sort_keys=True)
                guidance_key = hashlib.sha256(guidance_content_str.encode('utf-8')).hexdigest()
                if guidance_key in self.guidance_cache:
                    formatted_guidance = self.guidance_cache[guidance_key]
                    self.logger.debug(f"[NR Cache Hit QID:{query_id_for_log}] Using cached guidance (Key: {guidance_key[:8]}...).")
                else:
                    self.logger.info(f"[NR QID:{query_id_for_log}] Processing {len(symbolic_guidance)} symbolic guidance items.")
                    formatted_guidance = self._format_guidance(symbolic_guidance)
                    self.guidance_cache[guidance_key] = formatted_guidance
                    self.logger.debug(f"[NR Cache Miss QID:{query_id_for_log}] Cached new guidance (Key: {guidance_key[:8]}...). Count: {len(formatted_guidance)}")
            except Exception as guid_err:
                self.logger.error(f"[NR Warning QID:{query_id_for_log}] Failed to process symbolic guidance: {guid_err}. Proceeding without guidance.")
                formatted_guidance = []

        # --- Context Chunking ---
        context_chunks = []
        try:
            context_key = hashlib.sha256(context.encode('utf-8')).hexdigest()
            if context_key in self.context_chunk_cache:
                context_chunks = self.context_chunk_cache[context_key]
                self.logger.debug(f"[NR Cache Hit QID:{query_id_for_log}] Using cached context chunks (Key: {context_key[:8]}...). Count: {len(context_chunks)}")
            else:
                self.logger.info(f"[NR QID:{query_id_for_log}] Chunking context (Length: {len(context)}).")
                context_chunks = self._chunk_context(context)
                # Handle case where chunking might fail or return empty
                if not context_chunks:
                    self.logger.warning(f"[NR QID:{query_id_for_log}] Context chunking yielded no chunks. Using full context as fallback chunk.")
                    fallback_embedding = self._encode_safely(context)
                    if fallback_embedding is not None:
                        context_chunks = [{
                            'text': context, 'embedding': fallback_embedding,
                            'sentences': [context], 'start_sentence_idx': 0, 'end_sentence_idx': 0
                        }]
                    else:
                        # If even fallback encoding fails, cannot proceed
                        self.logger.error(f"[NR Error QID:{query_id_for_log}] Failed to encode fallback context.")
                        if dataset_type and dataset_type.lower() == 'drop':
                            return self._empty_drop()
                        return "Error: Failed to process context."

                self.context_chunk_cache[context_key] = context_chunks
                self.logger.debug(f"[NR Cache Miss QID:{query_id_for_log}] Cached new context chunks (Key: {context_key[:8]}...). Count: {len(context_chunks)}")
        except Exception as chunk_err:
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Context chunking failed: {chunk_err}")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return f"Error: Failed during context processing: {chunk_err}"

        # --- Chunk Prioritization (HotpotQA) ---
        context_chunks_for_retrieval = context_chunks
        if supporting_facts and dataset_type != 'drop':
            try:
                self.logger.debug(f"[NR QID:{query_id_for_log}] Prioritizing chunks based on {len(supporting_facts)} supporting facts.")
                context_chunks_for_retrieval = self._prioritize_supporting_facts(context_chunks, supporting_facts)
            except Exception as sf_err:
                self.logger.warning(f"[NR Warning QID:{query_id_for_log}] Failed to prioritize supporting facts: {sf_err}. Using original chunk order.")

        # --- Question Encoding ---
        question_embedding = self._encode_safely(question)
        if question_embedding is None:
            self.logger.error(f"[NR Error QID:{query_id_for_log}] Failed to encode question.")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return "Error: Failed to process question embedding."

        # --- Relevant Chunk Retrieval ---
        relevant_chunks = []
        try:
            self.logger.debug(f"[NR QID:{query_id_for_log}] Getting relevant chunks from {len(context_chunks_for_retrieval)} candidates.")
            relevant_chunks = self._get_relevant_chunks(question_embedding, context_chunks_for_retrieval, formatted_guidance)
            if not relevant_chunks:
                # Fallback if no relevant chunks found (e.g., low similarity)
                self.logger.warning(f"[NR QID:{query_id_for_log}] No relevant chunks found after scoring; using top chunk if available.")
                if context_chunks_for_retrieval:
                    # Find the chunk with highest score even if below threshold
                    if hasattr(self, '_last_scored_chunks') and self._last_scored_chunks:
                        relevant_chunks = [self._last_scored_chunks[0]['chunk']]
                    else:  # Absolute fallback
                        relevant_chunks = [context_chunks_for_retrieval[0]]
                else:
                    self.logger.error(f"[NR Error QID:{query_id_for_log}] No context chunks available at all.")
                    if dataset_type and dataset_type.lower() == 'drop':
                        return self._empty_drop()
                    return "Error: No context available for retrieval."
        except Exception as ret_err:
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Chunk retrieval failed: {ret_err}")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return f"Error: Failed during chunk retrieval: {ret_err}"

        # --- Prompt Creation ---
        try:
            self.logger.debug(f"[NR QID:{query_id_for_log}] Creating prompt with {len(relevant_chunks)} relevant chunks.")
            prompt = self._create_enhanced_prompt(question, relevant_chunks, formatted_guidance, dataset_type)
            self.logger.debug(f"[NR Prompt QID:{query_id_for_log}] Prompt length: {len(prompt)}")
        except Exception as prmpt_err:
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Prompt creation failed: {prmpt_err}")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return f"Error: Failed during prompt creation: {prmpt_err}"

        # --- Tokenization ---
        try:
            # Ensure prompt is not excessively long before tokenization
            # Heuristic: limit chars based on avg token length
            # Using tokenizer max_length is the primary control, but this adds a safety layer
            max_chars = self.max_context_length * 5
            if len(prompt) > max_chars:
                self.logger.warning(f"[NR QID:{query_id_for_log}] Prompt length {len(prompt)} chars exceeds safety limit {max_chars}. Truncating input string.")
                prompt = prompt[:max_chars]

            inputs_pt = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,  # Ensure truncation happens if still too long for model
                max_length=self.max_context_length  # Use model's max length
            ).to(self.device)
            # Log actual token length after tokenization
            token_len = inputs_pt['input_ids'].shape[1]
            self.logger.debug(f"[NR QID:{query_id_for_log}] Tokenized prompt length: {token_len} tokens.")
            if token_len >= self.max_context_length:
                self.logger.warning(f"[NR QID:{query_id_for_log}] Input prompt was truncated to max model length ({self.max_context_length}).")

        except Exception as e:
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Tokenization failed: {e}")  # Log stack trace
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return "Error: Failed during input tokenization."

        # --- LLM Generation ---
        self.logger.info(f"[NR Generate QID:{query_id_for_log}] Generating response (Input tokens: {token_len})...")
        response_text = "Error: Generation failed"  # Default
        try:
            with torch.no_grad():  # Ensure no gradients calculated during inference
                outputs = self.model.generate(
                    **inputs_pt,
                    # Generation parameters - consider making configurable
                    max_new_tokens=150,  # Reduced max tokens for more concise answers, esp. for DROP
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,  # Prevent repetitive phrasing
                    pad_token_id=self.tokenizer.eos_token_id,  # Use EOS for padding
                    do_sample=True,  # Use sampling for potentially more natural answers
                    temperature=0.6,  # Slightly lower temp for more focused answers
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id  # Stop generation at EOS
                )
            # Decode response, skipping prompt tokens and special tokens
            # Ensure correct slicing based on actual input length
            input_token_length = inputs_pt['input_ids'].shape[1]
            response_text = self.tokenizer.decode(
                outputs[0][input_token_length:],  # Slice correctly to get only generated tokens
                skip_special_tokens=True
            )
            self.logger.info(f"[NR Generated QID:{query_id_for_log}] Raw Response: '{response_text[:100]}...'")
        except torch.cuda.OutOfMemoryError as oom_err:
            # Specific OOM handling
            self.logger.error(f"[NR Error QID:{query_id_for_log}] GPU OOM during generation: {oom_err}")
            # Try to clean cache before returning error
            torch.cuda.empty_cache()
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return "Error: GPU out of memory during answer generation."
        except Exception as e:
            self.logger.exception(f"[NR Error QID:{query_id_for_log}] Error during model generation: {e}")
            if dataset_type and dataset_type.lower() == 'drop':
                return self._empty_drop()
            return f"Error: Model generation failed: {e}"

        # --- Post-processing ---
        # [Updated May 16, 2025]: For DROP, parse response and return a DROP answer object
        if dataset_type and dataset_type.lower() == 'drop':
            try:
                self.logger.debug(f"[NR DROP PostProc QID:{query_id_for_log}] Original: '{response_text[:100]}...'")
                # Parse response into structured DROP answer
                parsed_result = self._parse_neural_for_drop(response_text.strip(), question)
                if parsed_result and parsed_result.get('type') in ['number', 'spans', 'date'] and parsed_result.get('value') is not None:
                    answer_obj = self._create_drop_answer_obj(parsed_result['type'], parsed_result['value'])
                    answer_obj.update({
                        'status': 'success',
                        'confidence': parsed_result['confidence'],
                        'rationale': 'Neural parsed result'
                    })
                    self.logger.info(f"[NR Success QID:{query_id_for_log}] Processing time: {time.time() - start_time:.2f}s. Final Result: {answer_obj}")
                    return answer_obj
                else:
                    self.logger.warning(f"[NR QID:{query_id_for_log}] Failed to parse neural output: {response_text[:100]}...")
                    answer_obj = self._empty_drop()
                    answer_obj.update({
                        'status': 'error',
                        'confidence': 0.1,
                        'rationale': f'Failed to parse neural output: {response_text[:50]}...',
                        'type': 'error'
                    })
                    return answer_obj
            except Exception as post_err:
                self.logger.error(f"[NR Error QID:{query_id_for_log}] DROP post-processing failed: {post_err}")
                answer_obj = self._empty_drop()
                answer_obj.update({
                    'status': 'error',
                    'confidence': 0.1,
                    'rationale': f'Post-processing failed: {post_err}',
                    'type': 'error'
                })
                return answer_obj
        else:
            # Non-DROP datasets (e.g., HotpotQA)
            try:
                final_result = self._post_process_response(response_text.strip(), dataset_type, query_id_for_log, question)
            except Exception as post_err:
                self.logger.error(f"[NR Error QID:{query_id_for_log}] Post-processing failed: {post_err}. Returning raw response.")
                final_result = response_text.strip()  # Return raw if post-processing fails

            # --- Final Logging and Return ---
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.logger.info(f"[NR Success QID:{query_id_for_log}] Processing time: {processing_time:.2f}s. Final Result: '{str(final_result)[:100]}...'")  # Use str() for safety
            return final_result

    def _format_guidance(self, symbolic_guidance: List[Union[Dict, str]]) -> List[Dict]:
        """Formats and validates symbolic guidance, extracting confidence."""
        formatted_guidance = []
        # Example domain keywords (could be loaded from config)
        domain_keywords = {
            "sports": ["touchdown", "field goal", "score", "game", "team", "player", "quarterback"],
            "finance": ["stock", "market", "investment", "price", "company", "revenue"],
            # Add more domains as needed
        }

        for item_idx, rule in enumerate(symbolic_guidance):
            rule_text: Optional[str] = None
            confidence: float = 0.7  # Default confidence

            try:
                if isinstance(rule, str):
                    rule_text = rule.strip()
                    if not rule_text:
                        continue  # Skip empty strings
                    # Simple heuristic: Boost confidence slightly if domain keywords are present
                    for domain, keywords in domain_keywords.items():
                        if any(keyword in rule_text.lower() for keyword in keywords):
                            confidence = 0.8
                            break

                elif isinstance(rule, dict):
                    # More robustly check for various possible keys containing the statement
                    for key in [self.guidance_statement_key, 'response', 'statement', 'text', 'source_text', 'content', 'rationale']:
                        potential_text = rule.get(key)
                        if isinstance(potential_text, str) and potential_text.strip():
                            rule_text = potential_text.strip()
                            break  # Use the first valid text found
                    if not rule_text:
                        self.logger.warning(f"Could not extract usable text from guidance dict at index {item_idx}: {rule}")
                        continue

                    # Extract confidence, ensuring it's a float between 0 and 1
                    extracted_conf = rule.get(self.guidance_confidence_key, rule.get('confidence'))  # Check both keys
                    if isinstance(extracted_conf, (float, int)):
                        confidence = min(1.0, max(0.0, float(extracted_conf)))
                    # else: use default confidence

                else:
                    self.logger.warning(f"Unsupported guidance format at index {item_idx}: {type(rule)}. Skipping.")
                    continue

                # Add if valid text was found
                if rule_text:
                    formatted_guidance.append({
                        "response": rule_text,  # Standardized key
                        "confidence": confidence
                    })
            except Exception as format_err:
                self.logger.error(f"Error formatting guidance item at index {item_idx}: {format_err}. Item: {rule}")
                continue  # Skip this item on error

        if formatted_guidance:
            try:
                # Calculate average only if list is not empty
                avg_confidence = sum(g['confidence'] for g in formatted_guidance) / len(formatted_guidance)
                self.logger.info(f"Formatted {len(formatted_guidance)} guidance items. Average confidence: {avg_confidence:.2f}")
            except ZeroDivisionError:
                self.logger.info("Formatted 0 guidance items.")

        return formatted_guidance

    def _encode_safely(self, text: str) -> Optional[torch.Tensor]:
        """Encodes text using self.encoder, with error handling and length check."""
        if not text or not isinstance(text, str) or not text.strip():
            self.logger.warning(f"Attempted to encode invalid/empty text: '{str(text)[:50]}...'")
            return None
        if self.encoder is None:
            self.logger.error("Cannot encode text: Sentence encoder is not available.")
            return None

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Simple character length check as a proxy for token length
                # Adjust multiplier as needed based on typical tokenization ratio
                max_input_chars = getattr(self.encoder, 'max_seq_length', 512) * 5
                if len(text) > max_input_chars:
                    self.logger.warning(f"Encoding long text ({len(text)} chars). Truncating to {max_input_chars} chars.")
                    text = text[:max_input_chars]

                # Encode with no_grad
                with torch.no_grad():
                    encoded = self.encoder.encode(text, convert_to_tensor=True)
                return encoded.to(self.device)  # Move to device *after* encoding

            except Exception as e:
                self.logger.warning(f"Encoding attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Final encoding attempt failed for text: '{text[:100]}...'. Error: {str(e)}")
                    return None
                time.sleep(0.1 * (attempt + 1))  # Basic backoff
        return None

    def _normalize_drop_number_for_comparison(self,
                                              value_str: Optional[Any]
                                              ) -> Optional[float]:
        """Normalizes numbers (string, int, float) to float for comparison, handles None and errors.
        [Added May 16, 2025]: Copied from hybrid_integrator.py to ensure consistent number normalization.
        """
        if value_str is None:
            return None
        try:
            if isinstance(value_str, (int, float)):
                result = float(value_str)
            else:
                s = str(value_str).replace(",", "").strip().lower()
                if not s:
                    return None

                words = {
                    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
                    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
                    "ten": 10.0
                }
                if s in words:
                    result = words[s]
                elif re.fullmatch(r'-?\d+(\.\d+)?', s):
                    result = float(s)
                else:
                    self.logger.debug(f"Could not normalize '{value_str}' to a number.")
                    return None

            # Convert to int if the number is a whole number
            if result.is_integer():
                result = float(int(result))
            return result

        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error normalizing number '{value_str}': {e}")
            return None

    def _parse_neural_for_drop(self, neural_raw_output: Optional[str], query: str,
                               expected_type_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parses the neural raw output for DROP dataset, extracting structured answers.
        Enhanced cleaning and prioritized line parsing.
        """
        if not neural_raw_output or not isinstance(neural_raw_output,
                                                   str) or not neural_raw_output.strip():  # Added isinstance check
            self.logger.debug(
                f"Cannot parse empty or invalid type neural output for query: {query[:50]}... Type: {type(neural_raw_output)}")
            return None

        query_lower = query.lower()

        # --- Determine Expected Answer Type (same as your existing logic source [937]-[939]) ---
        answer_type = expected_type_hint
        if not answer_type:
            if 'how many' in query_lower or 'difference' in query_lower or 'how much' in query_lower:
                answer_type = 'number'
            elif 'who' in query_lower or 'which team' in query_lower or 'which player' in query_lower or 'what team' in query_lower:
                answer_type = 'spans'
            elif 'when' in query_lower or 'what date' in query_lower or 'which year' in query_lower:
                answer_type = 'date'
            elif 'longest' in query_lower or 'shortest' in query_lower or 'most' in query_lower:
                if 'who' in query_lower or 'which' in query_lower:
                    answer_type = 'spans'
                else:
                    answer_type = 'number'
            else:
                answer_type = 'spans'  # Default if unsure

        self.logger.debug(
            f"Parsing neural output. Query: '{query[:50]}...'. Determined/Hinted Answer Type: {answer_type}.")

        # --- Enhanced Cleaning & Prioritized Line Processing ---
        cleaned_full_output = neural_raw_output.strip()
        # Remove common LLM prefixes that might precede the actual answer.
        cleaned_full_output = re.sub(r"^(here's the answer:|the answer is:|answer:)\s*", "", cleaned_full_output,
                                     flags=re.IGNORECASE)

        lines = cleaned_full_output.splitlines()
        text_to_parse_primary = lines[0].strip() if lines else cleaned_full_output  # Focus on the first non-empty line

        # If the primary line is very short (e.g. just a number/short span) and there are more lines,
        # it's likely the answer. If it's longer, it might contain explanations.
        self.logger.debug(
            f"Primary text for parsing: '{text_to_parse_primary[:100]}...' (from full output: '{cleaned_full_output[:100]}...')")

        parsed_value: Any = None
        confidence = 0.0  # Default confidence, to be updated based on parsing success

        try:
            # --- 1. Try Parsing based on determined answer_type from text_to_parse_primary ---
            if answer_type == 'number':
                # Look for a number at the beginning of the primary text
                num_match = re.match(r'^\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\b', text_to_parse_primary)
                if num_match:
                    number_str = num_match.group(1).replace(',', '')
                    try:
                        # Use the retriever's own normalization for consistency
                        parsed_value = self._normalize_drop_number_for_comparison(number_str)
                        if parsed_value is not None:
                            confidence = 0.85  # High confidence if clearly parsed number
                            self.logger.debug(f"Parsed number via regex from primary line: {parsed_value}")
                    except ValueError:
                        self.logger.debug(f"Failed to convert '{number_str}' to float from primary line.")
                if parsed_value is None:  # Fallback: check for written numbers
                    first_word_primary = text_to_parse_primary.split()[0].lower().strip(
                        '.,!?') if text_to_parse_primary.split() else ''
                    num_from_word = self._normalize_drop_number_for_comparison(first_word_primary)
                    if num_from_word is not None:
                        parsed_value = num_from_word
                        confidence = 0.7
                        self.logger.debug(f"Parsed number via word mapping from primary line: {parsed_value}")

            elif answer_type == 'spans':
                if self.nlp:
                    # Try NER on the primary line first
                    doc_primary = self.nlp(text_to_parse_primary)
                    expected_labels = {'PERSON', 'ORG', 'GPE', 'NORP', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                                       'LOC'}  # Broader set for spans
                    if 'who' in query_lower:
                        expected_labels = {'PERSON'}
                    elif 'team' in query_lower:
                        expected_labels = {'ORG'}
                    elif 'where' in query_lower:
                        expected_labels = {'GPE', 'LOC'}

                    extracted_spans_primary = [ent.text.strip() for ent in doc_primary.ents if
                                               ent.label_ in expected_labels]

                    # Filter spans using semantic similarity to query if encoder is available
                    if extracted_spans_primary and self.encoder and util:
                        query_embedding = self._encode_safely(query_lower)
                        if query_embedding is not None:
                            filtered_spans_sem = []
                            query_embedding = query_embedding.view(1, -1)
                            for span in extracted_spans_primary:
                                span_embedding = self._encode_safely(span)
                                if span_embedding is not None:
                                    similarity = util.cos_sim(query_embedding, span_embedding.view(1, -1)).item()
                                    if similarity >= 0.45:  # Similarity threshold
                                        filtered_spans_sem.append(span)
                            extracted_spans_primary = filtered_spans_sem

                    if extracted_spans_primary:
                        seen = set()  # Deduplicate
                        parsed_value = [s for s in extracted_spans_primary if
                                        not (s.lower() in seen or seen.add(s.lower()))]
                        confidence = 0.7
                        self.logger.debug(f"Parsed spans from primary line via spaCy NER: {parsed_value}")

                    # Fallback 1: Noun chunks on primary line
                    if not parsed_value:
                        non_stop_chunks_primary = [chunk.text.strip() for chunk in doc_primary.noun_chunks if
                                                   not all(tok.is_stop for tok in chunk) and len(
                                                       chunk.text.strip()) > 1 and len(chunk.text.split()) < 7]
                        if non_stop_chunks_primary:  # If any noun chunks, take the first one as a candidate
                            parsed_value = [non_stop_chunks_primary[0]]  # Could also do semantic scoring here
                            confidence = 0.55
                            self.logger.debug(f"Parsed span from primary line via Noun Chunk: {parsed_value}")

                    # Fallback 2: If primary line failed and there are multiple lines, try NER on more context
                    if not parsed_value and len(lines) > 1 and text_to_parse_primary != cleaned_full_output:
                        self.logger.debug(
                            f"Primary line span parsing failed. Trying NER on broader context (up to 3 lines).")
                        broader_text = "\n".join(lines[:3]).strip()  # Use first few lines
                        doc_broader = self.nlp(broader_text)
                        extracted_spans_broader = [ent.text.strip() for ent in doc_broader.ents if
                                                   ent.label_ in expected_labels]
                        if extracted_spans_broader and self.encoder and util:  # Semantic filter
                            # (Repeat semantic filtering logic as above for extracted_spans_broader)
                            pass  # Placeholder for brevity
                        if extracted_spans_broader:
                            seen = set()
                            parsed_value = [s for s in extracted_spans_broader if
                                            not (s.lower() in seen or seen.add(s.lower()))]
                            confidence = 0.6
                            self.logger.debug(f"Parsed spans from broader context via spaCy NER: {parsed_value}")

                # Fallback 3: If no NLP or still no spans, use primary line if it looks like a short answer
                if not parsed_value:
                    if len(text_to_parse_primary) > 1 and len(text_to_parse_primary.split()) < 10 and any(
                            c.isalpha() for c in text_to_parse_primary):
                        parsed_value = [text_to_parse_primary]
                        confidence = 0.4
                        self.logger.debug(f"Parsed span via primary line heuristic: {parsed_value}")

            elif answer_type == 'date':
                # Try parsing date from the primary line first
                date_patterns = [  # Simplified, assuming output is clean
                    (r'^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b',
                     lambda m: {'day': m.group(1).split('/')[1], 'month': m.group(1).split('/')[0],
                                'year': m.group(1).split('/')[2]}),
                    (r'^\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
                     lambda m: {'day': re.search(r'\d{1,2}', m.group(1)).group(),
                                'month': re.match(r'[A-Za-z]+', m.group(1)).group(),
                                'year': re.search(r'\d{4}', m.group(1)).group()}),
                    (r'^\s*(1[89]\d{2}|20\d{2})\b', lambda m: {'day': '', 'month': '', 'year': m.group(1)})  # Year only
                ]
                for pattern, parser_func in date_patterns:
                    match = re.match(pattern, text_to_parse_primary, re.IGNORECASE)
                    if match:
                        try:
                            parsed_value = parser_func(match)
                            # Basic validation of parsed date components
                            yr = parsed_value.get('year')
                            if yr and (1000 <= int(yr) <= 2100):  # Basic year check
                                confidence = 0.8
                                self.logger.debug(f"Parsed date via regex from primary line: {parsed_value}")
                                break  # Stop on first successful date pattern match
                            else:
                                parsed_value = None  # Invalid year
                        except Exception:
                            parsed_value = None  # Parsing error

                if not parsed_value:  # Fallback to dateutil on the primary line
                    try:
                        parsed_dt_obj = date_parser.parse(text_to_parse_primary, fuzzy=False)  # Less fuzzy
                        parsed_value = {'day': str(parsed_dt_obj.day), 'month': str(parsed_dt_obj.month),
                                        'year': str(parsed_dt_obj.year)}
                        confidence = 0.7
                        self.logger.debug(f"Parsed date via dateutil from primary line: {parsed_value}")
                    except (ValueError, OverflowError, TypeError) as date_err:
                        self.logger.debug(
                            f"Dateutil parsing failed for primary line '{text_to_parse_primary}': {date_err}")

            # --- Finalize Result ---
            if parsed_value is not None:  # Ensure something was actually parsed
                # Validate against empty strings if spans or numbers resulted in empty after normalization
                if answer_type == 'spans' and not any(s for s in parsed_value if s):  # All spans are empty
                    self.logger.debug(f"Parsed spans resulted in only empty strings. Discarding.")
                    return None
                if answer_type == 'number' and parsed_value == '':  # Number parsed to empty string
                    self.logger.debug(f"Parsed number resulted in empty string. Discarding.")
                    return None

                return {
                    'type': answer_type,
                    'value': parsed_value,
                    'confidence': confidence
                }

            self.logger.warning(
                f"Failed to parse neural output for expected type '{answer_type}'. Primary text: '{text_to_parse_primary[:100]}...' Full cleaned: '{cleaned_full_output[:100]}...'")
            return None

        except Exception as e:
            self.logger.error(
                f"Critical error parsing neural output for DROP: {str(e)}. Query: '{query[:50]}...'. Raw output: '{neural_raw_output[:100]}...'",
                exc_info=True)
            return None

    def _create_drop_answer_obj(self, answer_type: Optional[str], value: Any) -> Dict[str, Any]:
        """
        Create a DROP answer object with validation.
        Handles 'extreme_value', 'count', and 'difference' types appropriately.
        Ensures all required fields are included for evaluation.
        """
        obj = {
            'number': '',
            'spans': [],
            'date': {'day': '', 'month': '', 'year': ''},
            'status': 'success',
            'confidence': 0.5,
            'rationale': 'Created DROP answer object',
            'type': answer_type or 'unknown'
        }
        try:
            if answer_type in ["number", "count", "difference"]:
                num_val = self._normalize_drop_number_for_comparison(value)
                if num_val is not None:
                    # Convert to int if the value is a whole number
                    if num_val.is_integer():
                        num_val = int(num_val)
                    obj["number"] = str(num_val)
                    obj["rationale"] = f"Parsed number value: {obj['number']}"
                    self.logger.debug(f"Created DROP answer: number={obj['number']}")
                else:
                    obj["status"] = "error"
                    obj["rationale"] = f"Invalid number value: {value}"
                    self.logger.warning(f"Invalid number value for DROP obj: {value}")

            elif answer_type == "extreme_value":
                if isinstance(value, (int, float, str)) and str(value).replace('.', '').isdigit():
                    num_val = self._normalize_drop_number_for_comparison(value)
                    if num_val is not None:
                        if num_val.is_integer():
                            num_val = int(num_val)
                        obj["number"] = str(num_val)
                        obj["rationale"] = f"Parsed extreme_value as number: {num_val}"
                        self.logger.debug(f"Created DROP answer (extreme_value as number): number={obj['number']}")
                    else:
                        obj["status"] = "error"
                        obj["rationale"] = f"Invalid extreme_value number: {value}"
                elif isinstance(value, list):
                    spans_in = value if isinstance(value, list) else ([value] if value is not None else [])
                    # Deduplicate spans while (case-insensitive)
                    seen = set()
                    obj["spans"] = [str(v).strip() for v in spans_in if str(v).strip() and str(v).strip().lower() not in seen and not seen.add(str(v).strip().lower())]
                    obj["rationale"] = f"Parsed extreme_value as spans: {obj['spans']}"
                    self.logger.debug(f"Created DROP answer (extreme_value as spans): spans={obj['spans']}")
                else:
                    obj["status"] = "error"
                    obj["rationale"] = f"Invalid extreme_value value: {value}"
                    self.logger.warning(f"Invalid extreme_value value: {value}")

            elif answer_type in ["spans", "entity_span"]:
                spans_in = value if isinstance(value, list) else ([value] if value is not None else [])
                # Deduplicate spans (case-insensitive)
                seen = set()
                obj["spans"] = [str(v).strip() for v in spans_in if str(v).strip() and str(v).strip().lower() not in seen and not seen.add(str(v).strip().lower())]
                obj["rationale"] = f"Parsed spans: {obj['spans']}"
                self.logger.debug(f"Created DROP answer: type={answer_type}, spans={obj['spans']}")

            elif answer_type == "date":
                if isinstance(value, dict) and all(k in value for k in ['day', 'month', 'year']):
                    try:
                        d = int(value.get('day', '')) if value.get('day', '') else 0
                        m = int(value.get('month', '')) if value.get('month', '') else 0
                        y = int(value.get('year', '')) if value.get('year', '') else 0
                        if (1 <= d <= 31 or d == 0) and (1 <= m <= 12 or m == 0) and (1000 <= y <= 3000 or y == 0):
                            obj["date"] = {k: str(v).strip() for k, v in value.items() if k in ['day', 'month', 'year']}
                            obj["rationale"] = f"Parsed date: {obj['date']}"
                            self.logger.debug(f"Created DROP answer: date={obj['date']}")
                        else:
                            raise ValueError("Invalid date components")
                    except (ValueError, TypeError):
                        obj["status"] = "error"
                        obj["rationale"] = f"Invalid date component values: {value}"
                        self.logger.warning(f"Invalid date component values: {value}")
                else:
                    obj["status"] = "error"
                    obj["rationale"] = f"Invalid date dictionary value: {value}"
                    self.logger.warning(f"Invalid date dictionary value: {value}")

            elif answer_type == "error":
                obj["status"] = "error"
                obj["rationale"] = str(value).strip() if value else "Unknown error"
                self.logger.debug(f"Created DROP error answer: {obj['rationale']}")

            else:
                obj["status"] = "error"
                obj["rationale"] = f"Unsupported or invalid answer type: {answer_type}"
                self.logger.warning(obj["rationale"])

            return obj

        except Exception as e:
            self.logger.error(f"Error creating DROP answer object (Type: {answer_type}, Value: {value}): {str(e)}")
            error_obj = {
                'number': '',
                'spans': [],
                'date': {'day': '', 'month': '', 'year': ''},
                'status': 'error',
                'confidence': 0.1,
                'rationale': f"Internal error creating answer object: {str(e)}",
                'type': answer_type or 'error'
            }
            return error_obj

    def _empty_drop(self) -> Dict[str, Any]:
        """Return an empty DROP answer object dictionary."""
        return {
            'number': '',
            'spans': [],
            'date': {'day': '', 'month': '', 'year': ''},
            'status': 'error',
            'confidence': 0.1,
            'rationale': 'Empty DROP answer object',
            'type': 'error'
        }

    def _post_process_response(self, response: str, dataset_type: Optional[str], query_id: str, query: str) -> str:
        """
        Post-processes the raw LLM response for cleaner output.
        Improved logic for DROP to extract core answer with span filtering.
        """
        if not response or not isinstance(response, str):
            self.logger.warning(f"[NR QID:{query_id}] Empty or invalid response type for post-processing: {type(response)}")
            return ""

        # General cleaning: remove leading/trailing whitespace
        response = response.strip()
        if not response:
            return ""

        # --- DROP Specific Post-processing ---
        if dataset_type == 'drop':
            self.logger.debug(f"[NR DROP PostProc QID:{query_id}] Original: '{response[:100]}...'")
            # Remove common explanation prefixes/suffixes more aggressively
            response = re.sub(r"^\s*(Answer:|Explanation:|Rationale:|The answer is)[:\s]*", "", response, flags=re.IGNORECASE).strip()
            response = re.sub(r"\s*(Explanation:|Rationale:).*", "", response, flags=re.IGNORECASE).strip()
            # Remove units
            response = re.sub(r'\s+\b(yards?|points?|feet|meters?)\b', '', response, flags=re.IGNORECASE).strip()

            # Parse response into structured DROP answer
            parsed = self._parse_neural_for_drop(response, query)
            if parsed:
                answer_type = parsed.get('type')
                value = parsed.get('value')
                if answer_type in ['number', 'count', 'difference']:
                    response = str(value)
                    self.logger.debug(f"[NR DROP PostProc QID:{query_id}] Extracted number: {response}")
                elif answer_type in ['spans', 'entity_span']:
                    spans = value if isinstance(value, list) else [value]
                    response = ' '.join(spans)
                    self.logger.debug(f"[NR DROP PostProc QID:{query_id}] Extracted spans: {response}")
                elif answer_type == 'date':
                    date = value
                    response = f"{date.get('month', '')}/{date.get('day', '')}/{date.get('year', '')}".strip('/')
                    self.logger.debug(f"[NR DROP PostProc QID:{query_id}] Extracted date: {response}")
            else:
                # Fallback: use cleaned response
                response = re.sub(r'[.,!?;:]$', '', response).strip()
                self.logger.warning(f"[NR DROP PostProc QID:{query_id}] Could not parse structured answer. Using cleaned response: '{response[:50]}...'")

        # --- General Text QA Post-processing ---
        else:  # HotpotQA or other text types
            self.logger.debug(f"[NR Text PostProc QID:{query_id}] Original: '{response[:100]}...'")
            # Remove common instruction/prompt remnants
            response = re.sub(r"^(Answer:|Response:|Based on the context.*?:)\s*", "", response, flags=re.IGNORECASE).strip()
            # Remove text after a potential follow-up question or explanation marker if model adds them
            stop_phrases = ["\nQuestion:", "\nExplanation:", "\nRationale:", "\nContext:", "\n---"]
            for phrase in stop_phrases:
                if phrase in response:
                    response = response.split(phrase)[0].strip()

        self.logger.debug(f"[NR PostProc QID:{query_id}] Final processed response: '{response[:100]}...'")
        # Final check for empty string after processing
        return response if response else " "  # Return single space if empty, to avoid downstream errors

    def _chunk_context(self, context: str) -> List[Dict]:
        """Chunks context into smaller pieces with overlap, adding embeddings."""
        if not isinstance(context, str) or not context.strip():
            self.logger.warning("Attempted to chunk empty or invalid context.")
            return []
        if self.encoder is None:
            self.logger.error("Cannot chunk context: Sentence encoder not available.")
            return []

        # Use regex for sentence splitting (more robust than just '.')
        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.!?])\s+', context) if s.strip()]
        if not sentences:
            self.logger.warning("No sentences found after splitting context. Using full context as one sentence.")
            sentences = [context.strip()]

        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        sentence_embeddings = []  # Store embeddings for chunks

        # Pre-tokenize all sentences (more efficient for length checks)
        # Note: This uses the LLM tokenizer for length check, not the sentence encoder's tokenizer
        sentence_tokens = self.tokenizer(sentences, add_special_tokens=False).input_ids

        # Pre-encode sentences if possible (if encoder supports batching well)
        try:
            with torch.no_grad():
                # Use a reasonable batch size for sentence transformer
                sentence_embeddings_all = self.encoder.encode(sentences, convert_to_tensor=True, batch_size=64, show_progress_bar=False).to(self.device)
        except Exception as enc_err:
            self.logger.error(f"Batch sentence encoding failed: {enc_err}. Will encode individually.")
            sentence_embeddings_all = None  # Flag to encode individually

        start_sentence_idx = 0
        for i, sentence in enumerate(sentences):
            # Use pre-tokenized length
            num_tokens_in_sentence = len(sentence_tokens[i]) if i < len(sentence_tokens) else len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # If adding this sentence exceeds chunk size, finalize the previous chunk
            if current_chunk_sentences and current_chunk_length + num_tokens_in_sentence > self.chunk_size:
                chunk_text = " ".join(current_chunk_sentences)
                # Use pre-encoded embeddings or encode chunk text
                chunk_embedding = None
                if sentence_embeddings:  # Check if list is not empty
                    try:
                        # Simple averaging of sentence embeddings for chunk embedding
                        if len(sentence_embeddings) > 0:
                            chunk_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
                        else:  # Should not happen if current_chunk_sentences was non-empty, but safeguard
                            chunk_embedding = self._encode_safely(chunk_text)
                    except Exception as avg_err:
                        self.logger.warning(f"Failed to average sentence embeddings: {avg_err}. Encoding full chunk text.")
                        chunk_embedding = self._encode_safely(chunk_text)
                else:  # Fallback if sentence embeddings failed or list was empty
                    chunk_embedding = self._encode_safely(chunk_text)

                if chunk_embedding is not None:
                    chunks.append({
                        'text': chunk_text,
                        'embedding': chunk_embedding,
                        'sentences': current_chunk_sentences.copy(),  # Use copy
                        'start_sentence_idx': start_sentence_idx,
                        'end_sentence_idx': start_sentence_idx + len(current_chunk_sentences) - 1  # Correct end index
                    })
                else:
                    self.logger.warning(f"Failed to embed chunk ending at sentence {start_sentence_idx + len(current_chunk_sentences) - 1}: {chunk_text[:100]}...")

                # Start new chunk with overlap
                # Find overlap point (approx. self.overlap tokens back)
                overlap_len = 0
                overlap_start_rel_idx = len(current_chunk_sentences) - 1  # Start from last added sentence index relative to current chunk
                while overlap_start_rel_idx >= 0:
                    # Get token length for the sentence at the corresponding global index
                    global_idx = start_sentence_idx + overlap_start_rel_idx
                    len_prev_sent = len(sentence_tokens[global_idx]) if global_idx < len(sentence_tokens) else 0
                    if overlap_len + len_prev_sent > self.overlap and overlap_start_rel_idx < len(current_chunk_sentences) - 1:  # Ensure overlap doesn't exceed limit AND we keep at least one sentence if possible
                        break  # Stop before adding this sentence if it exceeds overlap limit
                    overlap_len += len_prev_sent
                    overlap_start_rel_idx -= 1  # Move to previous sentence

                # The actual starting relative index for the new chunk's overlap
                overlap_start_rel_idx = max(0, overlap_start_rel_idx + 1)

                # Reset for new chunk starting from overlap
                new_start_sentence_idx = start_sentence_idx + overlap_start_rel_idx
                current_chunk_sentences = current_chunk_sentences[overlap_start_rel_idx:]  # Slice sentences
                # Recalculate length based on actual sentences kept
                current_chunk_length = sum(len(sentence_tokens[new_start_sentence_idx + k]) for k in range(len(current_chunk_sentences)))
                # Update sentence embeddings for the new chunk
                sentence_embeddings = sentence_embeddings[overlap_start_rel_idx:]  # Slice embeddings
                start_sentence_idx = new_start_sentence_idx  # Update the global start index for the next chunk

            # Add current sentence to the chunk
            current_chunk_sentences.append(sentence)
            current_chunk_length += num_tokens_in_sentence
            # Add sentence embedding if available
            if sentence_embeddings_all is not None and i < len(sentence_embeddings_all):
                sentence_embeddings.append(sentence_embeddings_all[i])
            # else: need to handle case where batch encoding failed - maybe encode individually here? (slower)

        # Add the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_embedding = None
            if sentence_embeddings:  # Check if list is not empty
                try:
                    if len(sentence_embeddings) > 0:
                        chunk_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
                    else:
                        chunk_embedding = self._encode_safely(chunk_text)
                except Exception:
                    chunk_embedding = self._encode_safely(chunk_text)
            else:
                chunk_embedding = self._encode_safely(chunk_text)

            if chunk_embedding is not None:
                chunks.append({
                    'text': chunk_text,
                    'embedding': chunk_embedding,
                    'sentences': current_chunk_sentences,
                    'start_sentence_idx': start_sentence_idx,
                    'end_sentence_idx': len(sentences) - 1  # Ends with last sentence
                })
            else:
                self.logger.warning(f"Failed to embed final chunk: {chunk_text[:100]}...")

        if not chunks:
            self.logger.error("No valid chunks created after processing. This should not happen.")
            # Create a minimal fallback if absolutely necessary
            fb_emb = self._encode_safely(context)
            if fb_emb is not None:
                return [{'text': context, 'embedding': fb_emb, 'sentences': [context], 'start_sentence_idx': 0, 'end_sentence_idx': 0}]
            else:
                return []  # Return empty if even fallback fails

        self.logger.info(f"Context chunking resulted in {len(chunks)} chunks.")
        return chunks

    def _prioritize_supporting_facts(self, chunks: List[Dict], supporting_facts: List[Tuple[str, int]]) -> List[Dict]:
        """Re-scores chunks based on presence of supporting fact sentences for HotpotQA."""
        if not supporting_facts:
            return chunks  # No facts to prioritize by

        # Create a set of global sentence indices that are supporting facts
        supporting_indices_global = {sent_idx for _, sent_idx in supporting_facts}
        if not supporting_indices_global:
            return chunks

        for chunk in chunks:
            chunk_start = chunk.get('start_sentence_idx', -1)
            chunk_end = chunk.get('end_sentence_idx', -1)

            # Skip if chunk indices are invalid
            if chunk_start == -1 or chunk_end == -1 or chunk_start > chunk_end:
                chunk['support_score'] = 0.0
                continue

            num_sentences_in_chunk = (chunk_end - chunk_start + 1)
            if num_sentences_in_chunk <= 0:  # Should not happen with valid indices
                chunk['support_score'] = 0.0
                continue

            # Count how many sentences within this chunk's range are supporting facts
            support_count = 0
            for global_sent_idx in range(chunk_start, chunk_end + 1):
                if global_sent_idx in supporting_indices_global:
                    support_count += 1

            # Calculate score as fraction of supporting sentences in the chunk
            chunk['support_score'] = support_count / num_sentences_in_chunk

        # Return chunks with scores added; sorting happens in _get_relevant_chunks
        self.logger.debug(f"Added support scores to {len(chunks)} chunks.")
        return chunks

    def _get_relevant_chunks(self, question_embedding: Optional[torch.Tensor], chunks: List[Dict],
                             symbolic_guidance: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Finds chunks most relevant to the question using similarity and guidance boosts.
        [Updated May 16, 2025]: Added numerical boost for 'difference' or 'more' queries to prioritize chunks with numbers,
        increased top_k for numerical queries, and enhanced logging for debugging.
        """
        if not chunks:
            self.logger.warning("No chunks provided to _get_relevant_chunks.")
            return []
        if question_embedding is None or question_embedding.nelement() == 0:
            self.logger.error("Question embedding is None or empty. Cannot score chunks.")
            # Fallback: return top_k chunks based on support score if available
            return sorted(chunks, key=lambda x: x.get('support_score', 0), reverse=True)[:self.top_k]

        scored_chunks = []
        question_emb = question_embedding.view(1, -1)  # Ensure 2D for similarity calculation

        # [Added May 16, 2025]: Determine if query is numerical (e.g., 'difference', 'more')
        is_numerical_query = any(keyword in chunks[0].get('text', '').lower() for keyword in ['difference', 'more'])
        effective_top_k = self.top_k * 2 if is_numerical_query and len(chunks) > self.top_k else self.top_k
        self.logger.debug(
            f"Effective top_k for chunk selection: {effective_top_k}{' (increased for numerical query)' if is_numerical_query else ''}")

        # --- Calculate Scores ---
        try:
            for i, chunk in enumerate(chunks):
                chunk_emb = chunk.get('embedding')
                # Validate chunk embedding
                if chunk_emb is None or not isinstance(chunk_emb, torch.Tensor) or chunk_emb.nelement() == 0:
                    self.logger.warning(f"Chunk {i} missing valid embedding. Assigning score 0.")
                    score = 0.0
                    debug_scores = {'base_sim': 0.0, 'support_boost': 0.0, 'guidance_boost': 0.0,
                                    'numerical_boost': 0.0}
                else:
                    chunk_emb = chunk_emb.to(self.device).view(1, -1)  # Ensure 2D and on correct device
                    # Dimension check
                    if chunk_emb.shape[1] != question_emb.shape[1]:
                        self.logger.error(
                            f"Dimension mismatch: question ({question_emb.shape}), chunk {i} ({chunk_emb.shape}). Skipping chunk.")
                        continue

                    # Calculate base similarity
                    base_sim = util.cos_sim(chunk_emb, question_emb).item()

                    # Calculate boosts
                    support_boost = chunk.get('support_score', 0.0) * self.support_boost
                    guidance_boost = self._calculate_guidance_boost(chunk,
                                                                    symbolic_guidance) if symbolic_guidance else 0.0
                    # [Added May 16, 2025]: Boost chunks with numerical content for numerical queries
                    numerical_boost = 0.0
                    if is_numerical_query:
                        chunk_text = chunk.get('text', '').lower()
                        if re.search(r'\b\d+\b', chunk_text):  # Presence of numbers
                            numerical_boost = 0.2  # Boost numerical chunks
                            self.logger.debug(
                                f"Applied numerical boost of {numerical_boost} to chunk {i} containing numbers.")

                    # Final score with boosts
                    score = base_sim + support_boost + guidance_boost + numerical_boost
                    debug_scores = {
                        'base_sim': base_sim,
                        'support_boost': support_boost,
                        'guidance_boost': guidance_boost,
                        'numerical_boost': numerical_boost
                    }

                scored_chunks.append({
                    'score': score,
                    'chunk': chunk,  # Keep reference to original chunk dict
                    'debug_scores': debug_scores
                })

        except Exception as e:
            self.logger.exception(f"Error scoring chunks: {e}")
            # Fallback: return top_k based on support score or original order
            return sorted(chunks, key=lambda x: x.get('support_score', 0), reverse=True)[:self.top_k]

        # Store scores for potential fallback use later
        self._last_scored_chunks = scored_chunks  # Store before sorting might change order if scores are same

        # --- Select Top Chunks ---
        # Sort by final score
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)

        # [Updated May 16, 2025]: Enhanced logging to include numerical boost
        debug_strings = [
            "{:.3f} (S:{:.2f} P:{:.2f} G:{:.2f} N:{:.2f})".format(
                sc['score'],
                sc['debug_scores']['base_sim'],
                sc['debug_scores']['support_boost'],
                sc['debug_scores']['guidance_boost'],
                sc['debug_scores']['numerical_boost']
            )
            for sc in scored_chunks[:5]
        ]
        self.logger.debug(f"Top 5 chunk scores (Base+Boosts): {debug_strings}")

        # Select effective_top_k, ensuring unique chunks
        relevant_chunks = []
        ids_added = set()
        for sc in scored_chunks:
            chunk_to_add = sc['chunk']
            chunk_id = id(chunk_to_add)  # Use object id for uniqueness check
            if chunk_id not in ids_added:
                relevant_chunks.append(chunk_to_add)
                ids_added.add(chunk_id)
                if len(relevant_chunks) >= effective_top_k:
                    break  # Stop once we have effective_top_k unique chunks

        if not relevant_chunks and chunks:  # Ensure we always return at least one chunk if possible
            self.logger.warning(
                "No chunks met selection criteria (e.g., all had errors or low scores), returning highest scored chunk as fallback.")
            relevant_chunks = [scored_chunks[0]['chunk']] if scored_chunks else [chunks[0]]  # Safest fallback

        self.logger.info(f"Selected {len(relevant_chunks)} relevant chunks for prompt.")
        return relevant_chunks

    def _calculate_guidance_boost(self, chunk: Dict, guidance: List[Dict]) -> float:
        """Calculates boost based on semantic similarity of guidance statements to chunk."""
        total_boost = 0.0
        # Basic validation checks
        if not guidance or not isinstance(chunk.get('text'), str) or not chunk['text'].strip():
            return 0.0
        if self.encoder is None:
            self.logger.warning("Encoder unavailable, cannot calculate guidance boost.")
            return 0.0

        chunk_emb = chunk.get('embedding')  # Use pre-calculated chunk embedding
        if chunk_emb is None or chunk_emb.nelement() == 0:
            self.logger.warning("Chunk embedding unavailable, cannot calculate guidance boost.")
            return 0.0  # Cannot calculate boost without chunk embedding

        chunk_emb = chunk_emb.to(self.device).view(1, -1)  # Ensure 2D and on correct device

        try:
            # Pre-encode guidance statements for efficiency if many chunks are checked against the same guidance
            guide_embeddings = []
            valid_guides = []
            for guide in guidance:
                if isinstance(guide, dict):
                    statement = guide.get('response', '')
                    confidence = guide.get('confidence', 0.0)
                    if isinstance(statement, str) and statement.strip() and isinstance(confidence, (float, int)) and confidence > 0.4:  # Threshold for considering guidance
                        guide_emb = self._encode_safely(statement)
                        if guide_emb is not None:
                            guide_embeddings.append(guide_emb.view(1, -1))  # Ensure 2D
                            valid_guides.append({'confidence': confidence})  # Store confidence

            if not guide_embeddings:
                return 0.0  # No valid guidance embeddings

            # Calculate similarities in batch if possible
            guide_embeddings_tensor = torch.cat(guide_embeddings, dim=0)  # Stack to (N, Dim)

            # Dimension check before similarity
            if guide_embeddings_tensor.shape[1] != chunk_emb.shape[1]:
                self.logger.error(f"Dimension mismatch between chunk ({chunk_emb.shape}) and guidance ({guide_embeddings_tensor.shape}) embeddings.")
                return 0.0

            similarities = util.cos_sim(chunk_emb, guide_embeddings_tensor).squeeze(0)  # Result shape (N,)

            # Apply boost based on similarity and confidence
            for i, similarity in enumerate(similarities):
                sim_item = similarity.item()
                if sim_item > 0.5:  # Only apply boost if reasonably similar
                    boost_amount = (sim_item * valid_guides[i]['confidence']) * self.guidance_boost_multiplier
                    total_boost += boost_amount

            # Clamp boost to limit
            final_boost = min(total_boost, self.guidance_boost_limit)
            # if final_boost > 0: self.logger.debug(f"Calculated guidance boost: {final_boost:.3f}")
            return final_boost

        except Exception as e:
            self.logger.error(f"Error calculating guidance boost: {e}")
            return 0.0  # Return 0 boost on error

    def _create_enhanced_prompt(self, question: str, chunks: List[Dict], symbolic_guidance: Optional[List[Dict]] = None,
                                dataset_type: Optional[str] = None) -> str:
        if not question or not isinstance(question, str) or not question.strip():
            self.logger.error("Cannot create prompt: Invalid question provided.")
            return self._create_fallback_prompt(question)

        prompt_parts = []

        # --- Guidance Section (Keep existing logic from source [1075]-[1079]) ---
        try:
            relevant_guidance_texts = []
            if symbolic_guidance:
                sorted_guidance = sorted([g for g in symbolic_guidance if isinstance(g, dict)],
                                         key=lambda x: x.get('confidence', 0.0), reverse=True)
                for guide in sorted_guidance[:3]:
                    statement = guide.get('response', '')  # 'response' is the standardized key from _format_guidance
                    confidence = guide.get('confidence', 0.0)
                    if statement and isinstance(statement,
                                                str) and statement.strip() and confidence > 0.5:  # Check type
                        safe_statement = statement.replace("{", "{{").replace("}", "}}")
                        relevant_guidance_texts.append(f"- {safe_statement} (Confidence: {confidence:.2f})")
            if relevant_guidance_texts:
                prompt_parts.append("You MAY use the following background information if relevant:")  # Softer phrasing
                prompt_parts.extend(relevant_guidance_texts)
                prompt_parts.append("---")  # Clearer separator
        except Exception as e:
            self.logger.warning(f"Error processing guidance for prompt: {e}")

        # --- Context Section (Keep existing logic from source [1080]-[1084]) ---
        context_parts = []
        added_content = set()
        chars_added = 0
        context_char_limit = int(self.max_context_length * 2.5)

        for chunk_idx, chunk in enumerate(chunks):  # Iterate relevant chunks
            chunk_text = chunk.get('text', '').strip()
            if chunk_text and chunk_text not in added_content:
                if chars_added + len(chunk_text) < context_char_limit:
                    # Add chunk index for clarity in prompt if many chunks
                    context_parts.append(f"Passage {chunk_idx + 1}:\n{chunk_text}")
                    added_content.add(chunk_text)
                    chars_added += len(chunk_text)
                else:
                    self.logger.debug("Context character limit reached for prompt. Stopping context addition.")
                    break
        if not context_parts:
            prompt_parts.append("Context:\n[No relevant context passages could be retrieved for this query.]")
        else:
            prompt_parts.append("Use ONLY the following context passages to answer the question:")
            prompt_parts.append("\n\n---\n\n".join(context_parts))  # Better separation for multiple passages
        prompt_parts.append("---")

        # --- Instruction and Question Section ---
        instruction = ""
        if dataset_type == 'drop':
            # MODIFICATION: More direct and specific instructions for DROP
            question_lower = question.lower()
            if 'how many' in question_lower or 'difference' in question_lower or re.search(
                    r'\b(what|which) (is|was) the (longest|shortest|highest|lowest|first|last)\b.*\b(number|value|score|yards|points|count)\b',
                    question_lower):
                instruction = "Answer with ONLY the single, precise numerical value. Do NOT include units (e.g., 'yards', 'points'). Do NOT provide any explanation or reasoning. Just the number."
                if 'difference' in question_lower:
                    instruction += " For difference queries, state only the final calculated difference."
            elif 'who' in question_lower or 'which team' in question_lower or 'which player' in question_lower or 'what team' in question_lower or \
                    re.search(
                        r'\b(what|which) (is|was) the (longest|shortest|highest|lowest|first|last)\b.*\b(player|team|name)\b',
                        question_lower):
                instruction = "Answer with ONLY the name(s) of the player(s), team(s), or entity(ies). If multiple, separate with a standard delimiter if natural. Do NOT provide any explanation or reasoning."
            elif 'when' in question_lower or 'what date' in question_lower or 'which year' in question_lower:
                instruction = "Answer with ONLY the date. If a full date (month, day, year), provide it. If only year, provide just the year. Format as MM/DD/YYYY or YYYY. Do NOT provide any explanation or reasoning."
            else:  # Default DROP instruction
                instruction = "Based ONLY on the context passages, provide the single, most precise answer. This might be a number, a short text span (like a name), or a date. Do NOT include explanations. Do NOT include units unless explicitly part of the answer span."

            self.logger.debug(f"DROP Instruction: {instruction}")

        else:  # Default / HotpotQA
            instruction = "Based ONLY on the context passages and any relevant background information provided, answer the following question accurately and concisely. Provide only the answer, no explanations."

        prompt_parts.append(f"Question: {question.strip()}")
        prompt_parts.append(f"\n{instruction}")
        # MODIFICATION: Change "Answer:" to a more explicit instruction for the LLM's output start.
        prompt_parts.append("\n\nPrecise Answer:")  # More direct

        final_prompt = "\n\n".join(prompt_parts)
        final_chars = len(final_prompt)
        if final_chars > self.max_context_length * 5:
            self.logger.warning(
                f"Final prompt length ({final_chars} chars) may exceed model limits for LLM {self.model.config.model_type if hasattr(self.model, 'config') else 'N/A'}.")
        elif final_chars > self.max_context_length:  # Max context length of the tokenizer
            self.logger.warning(
                f"Final prompt length ({final_chars} chars) exceeds tokenizer max_length ({self.max_context_length}). Will be truncated by tokenizer.")

        return final_prompt

    def _create_fallback_prompt(self, question: str) -> str:
        """Minimal prompt for use when other methods fail."""
        self.logger.warning("Using fallback prompt generation (simple question/answer format).")
        # Basic instruction
        instruction = "Answer the following question:"
        return f"{instruction}\nQuestion: {question.strip()}\nAnswer:"

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensures the output text ends with a complete sentence (heuristic)."""
        # This can be problematic and might add incorrect punctuation. Use with caution.
        # Often better handled by LLM prompting or simpler post-processing.
        # Keeping commented out as it might be too aggressive.
        # --- (Previous logic commented) ---
        return text.strip()  # Just strip whitespace for safety now