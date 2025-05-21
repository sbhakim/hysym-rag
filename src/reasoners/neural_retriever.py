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
import os # Added for path operations

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
                 device: Optional[torch.device] = None,
                 few_shot_examples_path: Optional[str] = None):  # This parameter is key
        """
        Initialize the Neural Retriever with model and configuration.
        [Updated May 21, 2025]: Logic to conditionally load few-shot examples based on few_shot_examples_path.
        """
        # Ensure self.logger is initialized early if used by other initializers
        self.logger = logging.getLogger(__name__)  # Use the module's logger
        # self.logger.setLevel(logging.DEBUG) # Or set level based on global config/args

        print(f"Initializing Neural Retriever with model: {model_name}...")
        if device is None:
            device = DeviceManager.get_device()
        self.device = device

        ProgressManager.disable_progress()
        transformers_logging.set_verbosity_error()
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("Tokenizer pad_token set to eos_token.")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
                load_in_8bit=use_quantization,
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Model {model_name} loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise

        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.logger.info("SentenceTransformer encoder loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer encoder: {e}")
            self.encoder = None

        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['lemmatizer', 'tagger'])
            self.logger.info("spaCy model 'en_core_web_sm' loaded successfully for NER and parsing.")
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {e}. Span parsing will be limited.")
            self.nlp = None

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
        self.guidance_cache = {}
        self.context_chunk_cache = {}
        self._last_scored_chunks = []

        # Load few-shot examples conditionally based on the provided path
        self.few_shot_examples = []
        if few_shot_examples_path:  # If a path is provided
            if os.path.exists(few_shot_examples_path):
                try:
                    with open(few_shot_examples_path, 'r', encoding='utf-8') as f:
                        self.few_shot_examples = json.load(f)
                    self.logger.info(
                        f"Successfully loaded {len(self.few_shot_examples)} few-shot examples from {few_shot_examples_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load or parse few-shot examples from {few_shot_examples_path}: {e}")
                    self.few_shot_examples = []  # Ensure it's empty on error
            else:
                self.logger.warning(
                    f"Few-shot examples path ('{few_shot_examples_path}') provided but file not found. No few-shot examples will be used.")
                self.few_shot_examples = []  # Ensure it's empty if file not found
        else:
            self.logger.info(
                "No few-shot examples path provided. Neural Retriever will operate without few-shot examples.")

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
                    # Add more common number words if needed, e.g., up to twenty, decades, hundreds
                }
                if s in words:
                    result = words[s]
                elif re.fullmatch(r'-?\d+(\.\d+)?', s): # Regex to match float/int strings
                    result = float(s)
                else:
                    # Try to remove trailing non-numeric if it's like "14..."
                    match_clean_num = re.match(r'(-?\d+(?:\.\d+)?)', s)
                    if match_clean_num:
                        result = float(match_clean_num.group(1))
                        self.logger.debug(f"Normalized '{value_str}' to {result} by stripping trailing non-numeric content.")
                    else:
                        self.logger.debug(f"Could not normalize '{value_str}' (cleaned: '{s}') to a recognized number format or word.")
                        return None
            return result # Return as float; conversion to int if whole happens later if needed

        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error normalizing number '{value_str}': {e}")
            return None

    def _parse_neural_for_drop(self, neural_raw_output: Optional[str], query: str,
                               expected_type_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parses the neural raw output for DROP dataset, extracting structured answers.
        [UPDATED]: Enhanced cleaning, prioritized line parsing, more robust type-specific extraction,
                   and confidence heuristics.
        """
        if not neural_raw_output or not isinstance(neural_raw_output, str) or not neural_raw_output.strip():
            self.logger.debug(
                f"Cannot parse empty or invalid neural output. Query: {query[:50]}... Type: {type(neural_raw_output)}")
            return None

        query_id_for_log = hashlib.sha1(query.encode('utf-8')).hexdigest()[:8]
        query_lower = query.lower()

        # --- Determine Expected Answer Type (copied from your _create_enhanced_prompt) ---
        answer_type = expected_type_hint
        if not answer_type:
            if 'how many' in query_lower or 'difference' in query_lower or 'how much' in query_lower or 'count the number of' in query_lower or \
                    re.search(
                        r'\b(what|which) (is|was) the (longest|shortest|highest|lowest|first|last)\b.*\b(number|value|score|yards|points|count|total)\b',
                        query_lower):
                answer_type = 'number'
            elif 'who' in query_lower or \
                    any(ph in query_lower for ph in ['which team', 'which player', 'what team', 'what player']) or \
                    re.search(
                        r'\b(what|which) (is|was) the (longest|shortest|first|last)\b.*\b(player|team|name|entity)\b',
                        query_lower):
                answer_type = 'spans'
            elif 'when' in query_lower or 'what date' in query_lower or 'which year' in query_lower:
                answer_type = 'date'
            else:
                answer_type = 'spans'

        self.logger.debug(
            f"[QID:{query_id_for_log}] Parsing neural output for DROP. Query: '{query[:50]}...'. Determined/Hinted Answer Type: {answer_type}.")

        # --- Enhanced Cleaning ---
        cleaned_output = neural_raw_output.strip()
        # More comprehensive list of prefixes/suffixes to strip, including common LLM conversational fillers
        patterns_to_strip = [
            r"^\s*(based on the context, the answer is:|the final answer is:|the answer is clearly:|the answer is:|answer:|explanation:|rationale:|according to the passage, the answer is:)\s*",
            r"\s*(Therefore, the final answer is|This means the answer is|So, the answer is|Hence, the answer is).*",
            r"\s*\n\n.*",  # Remove anything after a double newline, often explanation
            r"\s*Is there anything else I can help with\?.*$",
            r"\s*I hope this helps!.*$",
            r"^\s*Okay\s*[,.]?\s*",
            r"^\s*Sure\s*[,.]?\s*",
            r"^\s*Got it\s*[,.]?\s*",
            r"^\s*Alright\s*[,.]?\s*",
            r"\s*\(Note:.*?\)",  # Remove (Note: ...)
            r"\s*\(Confidence:.*?\)",
            r"\s*\[Source:.*?\]",
            r"\s*The question asks.*The answer is\s*",
            r"^\s*Output:\s*"
        ]
        for pattern in patterns_to_strip:
            cleaned_output = re.sub(pattern, "", cleaned_output, flags=re.IGNORECASE | re.DOTALL).strip()

        if not cleaned_output.strip() or re.fullmatch(r"[.,!?;:]*", cleaned_output):
            self.logger.debug(
                f"[QID:{query_id_for_log}] Cleaned output is trivial or empty: '{cleaned_output}'. Treating as unparsable.")
            return None

        lines = [line.strip() for line in cleaned_output.split('\n') if line.strip()]
        text_to_parse_primary = lines[0] if lines else cleaned_output  # Focus on the first non-empty line

        self.logger.debug(
            f"[QID:{query_id_for_log}] Primary text for parsing: '{text_to_parse_primary[:150]}...' (from cleaned output: '{cleaned_output[:200]}...')")

        parsed_value: Any = None
        confidence = 0.0  # Default low confidence

        try:
            if answer_type == 'number':
                # Attempt 1: Strict match at the beginning
                num_match_strict = re.match(r'^\s*(-?\s*\d[\d,.]*)\b', text_to_parse_primary)
                if num_match_strict:
                    parsed_value = self._normalize_drop_number_for_comparison(num_match_strict.group(1))
                    if parsed_value is not None: confidence = 0.85

                # Attempt 2: Find any number in the string, prioritize if only one or if context matches
                if parsed_value is None:
                    all_numbers_found = re.findall(r'-?\d[\d,.]*', text_to_parse_primary)
                    normalized_numbers = [self._normalize_drop_number_for_comparison(n) for n in all_numbers_found]
                    valid_numbers = [n for n in normalized_numbers if n is not None]
                    if valid_numbers:
                        if len(valid_numbers) == 1:
                            parsed_value = valid_numbers[0]
                            confidence = 0.75
                        else:  # Multiple numbers, be more cautious
                            # Heuristic: if a keyword like "total", "difference", "result" is near a number, prefer it.
                            # For now, just take the last one as a simple heuristic if multiple.
                            parsed_value = valid_numbers[-1]  # Could be improved
                            confidence = 0.60
                            self.logger.debug(
                                f"Multiple numbers {valid_numbers} for type 'number', chose last: {parsed_value}")

                # Attempt 3: Word numbers
                if parsed_value is None:
                    first_few_words = " ".join(text_to_parse_primary.split()[:3]).lower()
                    # Check for simple number words (can be expanded)
                    word_num = self._normalize_drop_number_for_comparison(first_few_words.split()[0].rstrip('.,'))
                    if word_num is not None:
                        parsed_value = word_num
                        confidence = 0.65


            elif answer_type == 'spans':
                candidate_spans = []
                # Prefer NER if available and produces results
                if self.nlp:
                    doc_primary = self.nlp(text_to_parse_primary)
                    expected_labels = {'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'NORP', 'FAC'}
                    if 'who' in query_lower or 'player' in query_lower:
                        expected_labels = {'PERSON'}
                    elif 'team' in query_lower or 'organization' in query_lower:
                        expected_labels = {'ORG', 'NORP'}
                    elif 'where' in query_lower or 'city' in query_lower or 'country' in query_lower:
                        expected_labels = {'GPE', 'LOC', 'FAC'}

                    ner_spans = [ent.text.strip(" .,'\"") for ent in doc_primary.ents if
                                 ent.label_ in expected_labels and len(ent.text.split()) < 7]  # Max 6 words for a span
                    if ner_spans: candidate_spans.extend(ner_spans)

                # If NER didn't yield, or for general spans, try noun chunks or quoted material.
                if not candidate_spans and self.nlp:
                    doc_primary = self.nlp(text_to_parse_primary)  # Re-parse if needed
                    # Prioritize longer noun chunks that aren't just determiners/pronouns
                    noun_chunks = [nc.text.strip(" .,'\"") for nc in doc_primary.noun_chunks
                                   if
                                   len(nc.text.split()) > 1 or (len(nc.text.split()) == 1 and nc.root.pos_ == 'PROPN')]
                    noun_chunks = [nc for nc in noun_chunks if nc and len(nc.split()) < 7]
                    if noun_chunks: candidate_spans.extend(noun_chunks)

                # Fallback: if still no spans, and the text is short, consider the whole text_to_parse_primary.
                # Also consider quoted text.
                if not candidate_spans:
                    quoted_spans = re.findall(r'["\']([^"\']+)["\']', text_to_parse_primary)
                    if quoted_spans:
                        candidate_spans.extend([qs.strip(" .,'") for qs in quoted_spans if qs.strip()])
                    elif len(text_to_parse_primary.split()) <= 7 and text_to_parse_primary:
                        candidate_spans.append(text_to_parse_primary.strip(" .,'\""))

                if candidate_spans:
                    seen_spans = set()  # Deduplicate
                    parsed_value = [s for s in candidate_spans if
                                    s and s.lower() not in seen_spans and not seen_spans.add(s.lower())]
                    if parsed_value:
                        confidence = 0.70 if self.nlp and ner_spans else (0.60 if self.nlp and noun_chunks else 0.5)
                    else:
                        parsed_value = None

            elif answer_type == 'date':
                date_found = None
                if date_parser:
                    try:
                        # Prioritize parsing the beginning of the string, as LLMs might append rationale
                        first_potential_date_segment = " ".join(
                            text_to_parse_primary.split()[:5])  # Look in first 5 words
                        parsed_dt_obj = date_parser.parse(first_potential_date_segment, fuzzy=False)  # Stricter parsing
                        if 1000 <= parsed_dt_obj.year <= 2100:  # Plausibility check
                            date_found = {'day': str(parsed_dt_obj.day), 'month': str(parsed_dt_obj.month),
                                          'year': str(parsed_dt_obj.year)}
                            confidence = 0.85
                    except (ValueError, OverflowError, TypeError):  # If strict fails, try fuzzy on whole string
                        try:
                            parsed_dt_obj = date_parser.parse(text_to_parse_primary, fuzzy=True)
                            if 1000 <= parsed_dt_obj.year <= 2100:
                                date_found = {'day': str(parsed_dt_obj.day), 'month': str(parsed_dt_obj.month),
                                              'year': str(parsed_dt_obj.year)}
                                confidence = 0.70
                        except (ValueError, OverflowError, TypeError):
                            self.logger.debug(
                                f"[QID:{query_id_for_log}] Dateutil fuzzy parsing failed for: '{text_to_parse_primary}'")

                # Regex fallback if dateutil fails or isn't available
                if not date_found:
                    # More specific year regex first
                    year_match_specific = re.search(r'\b(1[7-9]\d{2}|20[0-2]\d)\b',
                                                    text_to_parse_primary)  # Years 1700-2029
                    if year_match_specific:
                        date_found = {'day': '', 'month': '', 'year': year_match_specific.group(1)}
                        confidence = 0.65
                    else:  # Broader date patterns
                        # Consider patterns like MM/DD/YYYY, Month D, YYYY, YYYY
                        date_patterns = [
                            r"(\d{1,2}/\d{1,2}/\d{2,4})",  # MM/DD/YYYY or M/D/YY etc.
                            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
                            # Month D, YYYY
                            r"(\d{4})"  # Just year
                        ]
                        for dp in date_patterns:
                            match = re.search(dp, text_to_parse_primary, re.IGNORECASE)
                            if match and date_parser:  # Try to parse with dateutil for structure
                                try:
                                    parsed_dt_obj = date_parser.parse(match.group(1), fuzzy=True)
                                    if 1000 <= parsed_dt_obj.year <= 2100:
                                        date_found = {'day': str(parsed_dt_obj.day), 'month': str(parsed_dt_obj.month),
                                                      'year': str(parsed_dt_obj.year)}
                                        confidence = 0.60
                                        break
                                except:  # If parsing the regex match fails, it might just be a year
                                    if dp == r"(\d{4})":  # If the pattern was for a year
                                        date_found = {'day': '', 'month': '', 'year': match.group(1)}
                                        confidence = 0.55
                                        break
                            elif match and dp == r"(\d{4})":  # If no date_parser, but it's a year regex
                                date_found = {'day': '', 'month': '', 'year': match.group(1)}
                                confidence = 0.50
                                break
                parsed_value = date_found

            # Final validation of parsed_value content
            if parsed_value is not None:
                if answer_type == 'spans' and (
                        not parsed_value or not any(s.strip() for s in parsed_value if isinstance(s, str))):
                    parsed_value = None;
                    confidence = 0.0
                elif answer_type == 'number' and (str(parsed_value).strip() == '' or parsed_value is None):
                    parsed_value = None;
                    confidence = 0.0
                elif answer_type == 'date' and (
                        not parsed_value or not any(str(v).strip() for v in parsed_value.values())):
                    parsed_value = None;
                    confidence = 0.0

            if parsed_value is not None:
                self.logger.debug(
                    f"[QID:{query_id_for_log}] Successfully parsed: Type={answer_type}, Value='{str(parsed_value)[:100]}', Conf={confidence:.2f}")
                return {'type': answer_type, 'value': parsed_value, 'confidence': confidence}
            else:
                self.logger.warning(
                    f"[QID:{query_id_for_log}] Failed to parse neural output for expected type '{answer_type}'. Output: '{cleaned_output[:100]}...'")
                return None

        except Exception as e:
            self.logger.error(
                f"[QID:{query_id_for_log}] Critical error parsing neural output for DROP: {e}. Raw: '{neural_raw_output[:100]}'",
                exc_info=True)
            return None

    def _create_drop_answer_obj(self, answer_type: Optional[str], value: Any) -> Dict[str, Any]:
        """
        Create a DROP answer object with validation and correct native types for numbers.
        """
        # Initialize with native types or appropriate empty structures
        obj = {
            'number': "",  # Standard: empty string for no number
            'spans': [],
            'date': {'day': '', 'month': '', 'year': ''},
            'status': 'success',
            'confidence': 0.5,  # Default, will be updated by caller if parser provided one
            'rationale': 'DROP answer object created',
            'type': answer_type or 'unknown'
        }

        try:
            final_native_number_value: Optional[Union[int, float]] = None

            if answer_type in ["number", "count", "difference", "extreme_value_numeric", "temporal_difference"] or \
                    (answer_type == "extreme_value" and isinstance(value, (int, float, str)) and str(value).strip() and \
                     self._normalize_drop_number_for_comparison(str(value)) is not None):

                normalized_num = self._normalize_drop_number_for_comparison(value)
                if normalized_num is not None:
                    final_native_number_value = int(normalized_num) if normalized_num.is_integer() else float(
                        normalized_num)
                    obj["number"] = str(final_native_number_value)
                    obj["rationale"] = f"Parsed number value: {obj['number']}"
                else:
                    obj["status"] = "error"
                    obj["rationale"] = f"Invalid or unnormalizable number value: {value}"
                    obj["number"] = ""
                    self.logger.warning(f"Invalid number value for DROP obj: '{value}' for type '{answer_type}'")

            elif answer_type == "extreme_value" and isinstance(value, list):
                spans_in = value
                seen = set()
                obj["spans"] = [str(s).strip() for s in spans_in if
                                str(s).strip() and str(s).strip().lower() not in seen and not seen.add(
                                    str(s).strip().lower())]
                obj["rationale"] = f"Parsed extreme_value as spans: {obj['spans']}"
                if not obj["spans"] and spans_in:
                    self.logger.warning(
                        f"Value '{value}' for extreme_value spans resulted in empty list after cleaning.")


            elif answer_type in ["spans", "entity_span"]:
                spans_in = value if isinstance(value, list) else (
                    [str(value)] if value is not None and str(value).strip() else [])
                seen = set()
                obj["spans"] = [str(s).strip() for s in spans_in if
                                str(s).strip() and str(s).strip().lower() not in seen and not seen.add(
                                    str(s).strip().lower())]
                obj["rationale"] = f"Parsed spans: {obj['spans']}"
                if not obj["spans"] and spans_in:
                    self.logger.warning(
                        f"Value '{value}' for spans resulted in empty list after cleaning for type '{answer_type}'")


            elif answer_type == "date":
                if isinstance(value, dict) and all(k in value for k in ['day', 'month', 'year']):
                    try:
                        d_str = str(value.get('day', '')).strip()
                        m_str = str(value.get('month', '')).strip()
                        y_str = str(value.get('year', '')).strip()

                        # Validate only if a component is non-empty
                        d = int(d_str) if d_str and d_str.isdigit() else 0
                        m = int(m_str) if m_str and m_str.isdigit() else 0
                        y = int(y_str) if y_str and y_str.isdigit() else 0

                        is_valid_construction = False
                        if y_str:  # Year is mandatory if any date part is specified
                            if not (1000 <= y <= 2100): raise ValueError("Year out of plausible range.")
                            if d_str and m_str:  # Full date
                                if not (1 <= d <= 31 and 1 <= m <= 12): raise ValueError("Invalid day or month.")
                                is_valid_construction = True
                            elif m_str and not d_str:  # Month and year
                                if not (1 <= m <= 12): raise ValueError("Invalid month.")
                                is_valid_construction = True
                            elif not m_str and not d_str:  # Year only
                                is_valid_construction = True
                            # else: Day with Year but no Month is usually invalid.
                        elif not d_str and not m_str and not y_str:  # All empty is a valid "no date"
                            is_valid_construction = True

                        if is_valid_construction:
                            obj["date"] = {'day': d_str, 'month': m_str, 'year': y_str}
                            obj["rationale"] = f"Parsed date: {obj['date']}"
                        else:  # Some components present but not forming a valid date structure
                            raise ValueError(f"Inconsistent date components: D:'{d_str}', M:'{m_str}', Y:'{y_str}'")
                    except (ValueError, TypeError) as e:
                        obj["status"] = "error"
                        obj["rationale"] = f"Invalid date component values: {value}. Error: {e}"
                        self.logger.warning(f"Invalid date components: {value} for type {answer_type}. Error: {e}")
                elif value is None or (isinstance(value, dict) and not any(
                        str(v).strip() for v in value.values())):  # Explicitly empty date
                    obj["rationale"] = "Empty date value provided."
                else:
                    obj["status"] = "error"
                    obj["rationale"] = f"Invalid date dictionary value format: {value}"
                    self.logger.warning(f"Invalid date dictionary value: {value} for type {answer_type}")

            elif answer_type == "error":
                obj["status"] = "error"
                obj["rationale"] = str(value).strip() if value else "Unknown error type passed to creator"

            else:  # Unknown or unhandled answer type for creator
                obj["status"] = "error"
                obj[
                    "rationale"] = f"Internal Error: Unsupported answer type ('{answer_type}') or value ('{str(value)[:50]}...') for DROP object creation."
                self.logger.error(obj["rationale"])

            # If after all processing, status is success but all answer fields are effectively empty,
            # it might indicate a problem.
            if obj['status'] == 'success' and \
                    (obj['number'] == "" or obj['number'] is None) and \
                    not obj['spans'] and \
                    not any(str(v).strip() for v in obj['date'].values()):
                obj['status'] = 'error'  # Change status
                obj[
                    'rationale'] = f"Content successfully processed as type '{answer_type}' but resulted in an empty answer."
                obj['type'] = 'error_empty_result' # More specific error type
                self.logger.warning(
                    f"DROP object created successfully but is effectively empty for type '{answer_type}'. Input value: '{str(value)[:100]}'")

            return obj

        except Exception as e:
            self.logger.error(
                f"Critical error creating DROP answer object (Type: {answer_type}, Value: {value}): {str(e)}",
                exc_info=True)
            return {
                'number': "", 'spans': [], 'date': {'day': '', 'month': '', 'year': ''},
                'status': 'error', 'confidence': 0.05,
                'rationale': f"System error creating answer object: {str(e)}",
                'type': 'error_obj_creation'
            }

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
        For DROP, primarily relies on the structured object already created by _parse_neural_for_drop.
        This function aims to provide a clean *string* representation if needed elsewhere,
        but the primary result for DROP in retrieve_answer is the structured dictionary.
        """
        if not response or not isinstance(response, str):
            self.logger.warning(
                f"[NR QID:{query_id}] Empty or invalid response type for post-processing: {type(response)}")
            return ""  # Return empty string for invalid input

        response = response.strip()
        if not response:
            return ""  # Return empty string if stripping leads to empty

        self.logger.debug(f"[NR PostProc QID:{query_id}] Original response to post-process: '{response[:150]}...'")

        if dataset_type == 'drop':
            # For DROP, the structured parsing is done in _parse_neural_for_drop.
            # This function should aim to return a concise string representation if needed,
            # but the main output of retrieve_answer for DROP is the dict.
            # Let's try parsing it again to get a clean representation if it wasn't already a structured output.
            # However, retrieve_answer() should already return the dict from _create_drop_answer_obj.
            # This path in _post_process_response for DROP might be less critical if retrieve_answer passes dicts.

            # Aggressive cleaning of common LLM conversational fluff if not already handled
            cleaned_response = re.sub(r"^\s*(here's the answer:|the answer is:|answer:|explanation:|rationale:)\s*", "",
                                      response, flags=re.IGNORECASE).strip()
            cleaned_response = re.sub(
                r"\s*(explanation:|rationale:|\(conf:.*?\)|\|---.*|\s*note:.*|\s*important:.*|\s*therefore,.*$).*", "",
                cleaned_response, flags=re.IGNORECASE | re.DOTALL).strip()

            # If the cleaning results in an empty string, use the original (stripped) response
            if not cleaned_response and response:
                cleaned_response = response

            # Heuristic: if the cleaned response is very short, it might be the direct answer.
            # Otherwise, it might still contain explanations.
            # Focus on the first meaningful line.
            first_meaningful_line = cleaned_response.split('\n')[0].strip()

            # Further remove common units if they are trailing the number for cleaner numerical strings.
            # This specific cleaning is more about generating a "clean string" than parsing the structure.
            if re.match(r"^-?\d+(\.\d+)?\s+(yards?|points?|feet|meters?|degrees|percent|usd|\$|eur|€)$",
                        first_meaningful_line, flags=re.IGNORECASE):
                first_meaningful_line = re.sub(r'\s+\b(yards?|points?|feet|meters?|degrees|percent|usd|\$|eur|€)\b', '',
                                               first_meaningful_line, flags=re.IGNORECASE).strip()

            final_string_representation = re.sub(r'[.,!?;:]$', '',
                                                 first_meaningful_line).strip()  # Remove trailing punctuation for simple strings

            self.logger.debug(
                f"[NR PostProc QID:{query_id}] DROP - Processed string representation: '{final_string_representation[:100]}...' (Original input to func: '{response[:50]}...')")
            return final_string_representation if final_string_representation else " "  # Ensure non-empty return

        else:  # HotpotQA or other text types
            # Remove common instruction/prompt remnants
            processed_text = re.sub(r"^(Answer:|Response:|Based on the context.*?:)\s*", "", response,
                                    flags=re.IGNORECASE).strip()
            # Remove text after a potential follow-up question or explanation marker
            stop_phrases = ["\nQuestion:", "\nExplanation:", "\nRationale:", "\nContext:", "\n---",
                            "\nI hope this helps!"]
            for phrase in stop_phrases:
                if phrase in processed_text:
                    processed_text = processed_text.split(phrase)[0].strip()

            # Attempt to isolate the most direct answer sentence if possible (simple heuristic)
            sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.!?])\s+', processed_text) if
                         s.strip()]
            if sentences:
                # If first sentence is short and seems like a direct answer, use it.
                # Otherwise, if multiple sentences, check if one is a clear summary or answer.
                # This is complex; for now, let's take the first sentence if it's substantial,
                # or the whole processed_text if it's already concise.
                if len(sentences[0].split()) > 2 and len(sentences[0]) > 10:  # Heuristic for a "substantial" sentence
                    final_string_representation = sentences[0]
                else:
                    final_string_representation = processed_text  # Use the cleaned full text if first sentence is too short
            else:
                final_string_representation = processed_text

            self.logger.debug(
                f"[NR PostProc QID:{query_id}] TextQA - Processed string representation: '{final_string_representation[:100]}...'")
            return final_string_representation if final_string_representation else " "

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
                             symbolic_guidance: Optional[List[Dict]] = None,
                             query_complexity: Optional[float] = 0.5,  # Added query_complexity
                             question_text: Optional[str] = None  # Added question_text for numerical check
                             ) -> List[Dict]:
        """
        Finds chunks most relevant to the question using similarity and guidance boosts.
        [UPDATED]: Made top_k adaptive, refined numerical query detection, and scoring.
        """
        if not chunks:
            self.logger.warning("No chunks provided to _get_relevant_chunks.")
            return []
        if question_embedding is None or question_embedding.nelement() == 0:
            self.logger.error("Question embedding is None or empty. Cannot score chunks.")
            # Fallback: if supporting facts are available, use those, otherwise top_k random/first chunks
            return sorted(chunks, key=lambda x: x.get('support_score', 0), reverse=True)[:self.top_k]

        scored_chunks = []
        question_emb = question_embedding.view(1, -1)

        # Numerical query detection using actual question text
        is_numerical_query = False
        if question_text:
            qt_lower = question_text.lower()
            is_numerical_query = any(keyword in qt_lower for keyword in
                                     ['how many', 'difference', 'sum', 'total', 'average', 'count', 'number of',
                                      'what is the value']) or \
                                 any(keyword in qt_lower for keyword in
                                     ['more', 'less', 'fewer', 'greater', 'smaller', 'longest', 'shortest', 'highest',
                                      'lowest'])

        # Adaptive top_k based on query complexity
        if query_complexity is not None:
            if query_complexity < 0.3:  # Very simple query
                effective_top_k = max(1, self.top_k - 1)
            elif query_complexity > 0.7:  # Complex query
                effective_top_k = self.top_k + 1
            else:
                effective_top_k = self.top_k
        else:
            effective_top_k = self.top_k

        # Further increase top_k for numerical queries if many chunks, as numbers can be sparse
        if is_numerical_query and len(chunks) > effective_top_k * 1.5:
            effective_top_k = min(len(chunks), int(effective_top_k * 1.5))

        numerical_boost_value = 0.25 if is_numerical_query else 0.0  # Slightly reduced default numerical boost
        self.logger.debug(
            f"Effective top_k: {effective_top_k}. Numerical query: {is_numerical_query} (Boost: {numerical_boost_value}). Query Complexity: {query_complexity}")

        try:
            for i, chunk in enumerate(chunks):
                chunk_emb = chunk.get('embedding')
                if chunk_emb is None or not isinstance(chunk_emb, torch.Tensor) or chunk_emb.nelement() == 0:
                    self.logger.warning(f"Chunk {i} missing valid embedding. Assigning score 0.")
                    score = 0.0
                    debug_scores = {'base_sim': 0.0, 'support_boost': 0.0, 'guidance_boost': 0.0,
                                    'numerical_boost': 0.0, 'keyword_boost': 0.0}
                else:
                    chunk_emb = chunk_emb.to(self.device).view(1, -1)
                    if chunk_emb.shape[1] != question_emb.shape[1]:
                        self.logger.error(
                            f"Dimension mismatch: question ({question_emb.shape}), chunk {i} ({chunk_emb.shape}). Skipping chunk.")
                        continue

                    base_sim = util.cos_sim(chunk_emb, question_emb).item()
                    support_boost = chunk.get('support_score', 0.0) * self.support_boost
                    guidance_boost = self._calculate_guidance_boost(chunk,
                                                                    symbolic_guidance) if symbolic_guidance else 0.0

                    current_numerical_boost = 0.0
                    if is_numerical_query:
                        chunk_text_lower = chunk.get('text', '').lower()
                        if re.search(r'\b\d+\b', chunk_text_lower) or \
                                any(num_word in chunk_text_lower for num_word in
                                    ["one", "two", "three", "four", "five", "percent", "average", "total", "sum",
                                     "difference"]):
                            current_numerical_boost = numerical_boost_value
                            # More boost if specific number words from query are present
                            if question_text:
                                query_num_keywords = [w for w in question_text.lower().split() if
                                                      w.isdigit() or w in ["one", "two", "three"]]  # Add more if needed
                                if any(qkw in chunk_text_lower for qkw in query_num_keywords):
                                    current_numerical_boost += 0.1  # Additional small boost

                    # Keyword overlap boost (simple version)
                    keyword_boost = 0.0
                    if question_text and self.nlp:
                        query_doc = self.nlp(question_text.lower())
                        query_keywords = {token.lemma_ for token in query_doc if
                                          not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "PROPN",
                                                                                                      "VERB", "ADJ"}}
                        chunk_doc = self.nlp(chunk.get('text', '').lower())
                        chunk_keywords = {token.lemma_ for token in chunk_doc if
                                          not token.is_stop and not token.is_punct}
                        overlap = len(query_keywords.intersection(chunk_keywords))
                        if query_keywords:
                            keyword_boost = (overlap / len(query_keywords)) * 0.1  # Max 0.1 boost

                    score = base_sim + support_boost + guidance_boost + current_numerical_boost + keyword_boost
                    debug_scores = {
                        'base_sim': base_sim,
                        'support_boost': support_boost,
                        'guidance_boost': guidance_boost,
                        'numerical_boost': current_numerical_boost,
                        'keyword_boost': keyword_boost
                    }
                scored_chunks.append({'score': score, 'chunk': chunk, 'debug_scores': debug_scores})
        except Exception as e:
            self.logger.exception(f"Error scoring chunks: {e}")
            return sorted(chunks, key=lambda x: x.get('support_score', 0), reverse=True)[:self.top_k]

        self._last_scored_chunks = scored_chunks  # Store for potential fallback
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)

        debug_strings = [
            "{:.3f} (S:{:.2f} P:{:.2f} G:{:.2f} N:{:.2f} K:{:.2f})".format(
                sc['score'], sc['debug_scores']['base_sim'], sc['debug_scores']['support_boost'],
                sc['debug_scores']['guidance_boost'], sc['debug_scores']['numerical_boost'],
                sc['debug_scores']['keyword_boost']
            ) for sc in scored_chunks[:5]
        ]
        self.logger.debug(f"Top 5 chunk scores (Sim+Support+Guidance+Num+Keyword Boosts): {debug_strings}")

        relevant_chunks = []
        ids_added = set()
        for sc in scored_chunks:
            chunk_to_add = sc['chunk']
            # Ensure chunk has some text before adding
            if not chunk_to_add.get('text', '').strip():
                continue
            chunk_id = id(
                chunk_to_add)  # Using id might be tricky if chunks are regenerated; consider text hash if issues arise
            if chunk_id not in ids_added:
                relevant_chunks.append(chunk_to_add)
                ids_added.add(chunk_id)
                if len(relevant_chunks) >= effective_top_k:
                    break

        # Fallback if no chunks are selected (e.g., all scores are too low or chunks are empty)
        if not relevant_chunks and chunks:
            self.logger.warning("No chunks met selection criteria after scoring, attempting fallback.")
            # Fallback to highest scored, even if score is low, provided it has text
            if scored_chunks and scored_chunks[0]['chunk'].get('text', '').strip():
                relevant_chunks = [scored_chunks[0]['chunk']]
            else:  # If even that fails, take the first non-empty original chunk
                for original_chunk in chunks:
                    if original_chunk.get('text', '').strip():
                        relevant_chunks = [original_chunk]
                        break
                if not relevant_chunks:
                    self.logger.error("All provided chunks are empty or invalid. Cannot select relevant chunks.")

        self.logger.info(
            f"Selected {len(relevant_chunks)} relevant chunks for prompt (effective_top_k: {effective_top_k}).")
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

    def _create_enhanced_prompt(self, question: str, chunks: List[Dict],
                                symbolic_guidance: Optional[List[Dict]] = None,
                                dataset_type: Optional[str] = None) -> str:
        if not question or not isinstance(question, str) or not question.strip():
            self.logger.error("Cannot create prompt: Invalid question provided.")
            return self._create_fallback_prompt(question)

        prompt_parts = []
        query_id_for_log = hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]

        # --- Few-Shot Examples Section ---
        # This check ensures few-shots are only added if loaded AND for the correct dataset type
        if dataset_type and dataset_type.lower() == 'drop' and self.few_shot_examples:
            prompt_parts.append("Here are some examples of how to answer questions based on context:")
            # Select a limited number of few-shot examples to avoid overly long prompts
            # You can make num_few_shot_to_include configurable if needed
            num_few_shot_to_include = min(len(self.few_shot_examples), 2)

            for i, example in enumerate(self.few_shot_examples[:num_few_shot_to_include]):
                example_query = example.get("query", "")
                example_context_snippet = example.get("context_snippet", "")  # Optional: include if useful
                example_answer_obj = example.get("answer", {})

                answer_str_parts = []
                if example_answer_obj.get("number") is not None and str(example_answer_obj.get("number")).strip():
                    answer_str_parts.append(f"Number: {example_answer_obj['number']}")

                # Ensure spans is a list before joining
                spans_data = example_answer_obj.get("spans", [])
                if isinstance(spans_data, list) and spans_data:
                    answer_str_parts.append(f"Spans: {', '.join(str(s) for s in spans_data)}")

                date_obj = example_answer_obj.get("date", {})
                if isinstance(date_obj, dict) and any(str(v).strip() for v in date_obj.values()):
                    date_parts = []
                    if str(date_obj.get('month', '')).strip(): date_parts.append(str(date_obj['month']))
                    if str(date_obj.get('day', '')).strip(): date_parts.append(str(date_obj['day']))
                    if str(date_obj.get('year', '')).strip(): date_parts.append(str(date_obj['year']))
                    if date_parts:
                        answer_str_parts.append(f"Date: {'/'.join(filter(None, date_parts))}")

                formatted_answer = " | ".join(answer_str_parts) if answer_str_parts else "Not Applicable"

                prompt_parts.append(f"\nExample {i + 1}:")
                if example_context_snippet and "REPLACE THIS" not in example_context_snippet:  # Only include valid snippets
                    prompt_parts.append(f"Context Snippet: {example_context_snippet}")
                prompt_parts.append(f"Question: {example_query}")
                prompt_parts.append(f"Answer: {formatted_answer}")
            prompt_parts.append("---")
        elif dataset_type and dataset_type.lower() == 'drop' and not self.few_shot_examples:
            self.logger.debug(
                f"[QID:{query_id_for_log}] Few-shot examples for DROP are configured to be used, but none were loaded (e.g., file not found or empty).")

        # --- Guidance Section ---
        try:
            relevant_guidance_texts = []
            if symbolic_guidance:
                # Sort guidance by confidence, take top N (e.g., 2)
                sorted_guidance = sorted(
                    [g for g in symbolic_guidance if isinstance(g, dict) and g.get(self.guidance_statement_key)],
                    key=lambda x: x.get(self.guidance_confidence_key, 0.0),
                    reverse=True
                )
                for guide_idx, guide in enumerate(sorted_guidance[:2]):  # Take top 2
                    statement = guide.get(self.guidance_statement_key, '')
                    confidence = guide.get(self.guidance_confidence_key, 0.0)
                    # Ensure statement is a string and not empty after stripping
                    if isinstance(statement, str) and statement.strip() and confidence > 0.45:  # Confidence threshold
                        safe_statement = statement.replace("{", "{{").replace("}", "}}")  # Escape for f-string
                        relevant_guidance_texts.append(
                            f"- Clue {guide_idx + 1}: {safe_statement} (Symbolic Confidence: {confidence:.2f})"
                        )
            if relevant_guidance_texts:
                prompt_parts.append(
                    "Hint: Consider the following potentially relevant facts derived from symbolic reasoning (their accuracy is not guaranteed):"
                )
                prompt_parts.extend(relevant_guidance_texts)
                prompt_parts.append("---")
        except Exception as e:
            self.logger.warning(f"[QID:{query_id_for_log}] Error processing guidance for prompt: {e}")

        # --- Context Section ---
        context_parts = []
        added_content_hashes = set()  # To avoid adding duplicate chunk text
        chars_added = 0

        # Prioritize shorter, more distinct chunks if many are available, or use as is if few
        # This sorting can be computationally minor for small N, but consider its impact if chunks list is huge.
        sorted_chunks_for_prompt = sorted(chunks, key=lambda c: len(c.get('text', ''))) if len(
            chunks) > self.top_k * 2 else chunks

        for chunk_idx, chunk in enumerate(sorted_chunks_for_prompt):
            chunk_text = chunk.get('text', '').strip()
            # Use a hash of the chunk text to avoid adding effectively identical chunks if they appear multiple times
            chunk_hash = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()

            if chunk_text and chunk_hash not in added_content_hashes:
                # Estimate token length for context budget (simple character-based heuristic)
                # A more accurate way would be to tokenize, but that's slower here.
                # Assuming an average of ~4 chars/token for English.
                # Max context length for LLM is self.max_context_length (tokens).
                # Leave room for question, instructions, guidance, few-shot examples.
                # Let's say context should take up to 70-80% of the remaining budget.
                # This is a rough estimate and might need tuning.
                # A simpler approach: fixed character limit for context part of the prompt.
                context_char_limit = int(
                    self.max_context_length * 2.5)  # Heuristic: allow more chars than strict token count

                if chars_added + len(chunk_text) < context_char_limit:
                    context_parts.append(f"Passage {chunk_idx + 1}:\n{chunk_text}")
                    added_content_hashes.add(chunk_hash)
                    chars_added += len(chunk_text)
                else:
                    self.logger.debug(
                        f"[QID:{query_id_for_log}] Context character limit ({context_char_limit} chars) reached for prompt. Added {len(context_parts)} passages.")
                    break  # Stop adding chunks if limit is reached

        if not context_parts:
            # It's crucial to provide *some* context, even if it's a warning.
            # Otherwise, the LLM has no basis for answering.
            prompt_parts.append("Context:\n[No relevant context passages were retrieved or all were empty.]")
        else:
            prompt_parts.append("Based ONLY on the following context passages, answer the question.")
            prompt_parts.append("\n\n---\n\n".join(context_parts))  # Join passages with clear separators
        prompt_parts.append("---")

        # --- Instruction and Question Section ---
        instruction = ""
        query_lower = question.lower()

        if dataset_type == 'drop':
            self.logger.debug(
                f"[QID:{query_id_for_log}] Determining DROP-specific instruction for query: '{query_lower[:50]}...'")
            # Keywords for different DROP answer types
            is_count_query = 'how many' in query_lower or 'count the number of' in query_lower
            is_difference_query = 'difference' in query_lower or 'how many more' in query_lower or 'how many less' in query_lower
            is_extreme_value_numeric_query = re.search(
                r'\b(what|which) (is|was) the (longest|shortest|highest|lowest|first|last|greatest|smallest|maximum|minimum)\b.*\b(number|value|score|yards|points|count|total|amount|length|duration|age)\b',
                query_lower) or \
                                             ('how much' in query_lower and any(
                                                 ext in query_lower for ext in
                                                 ['longest', 'shortest', 'highest', 'lowest']))

            is_who_which_what_span_query = 'who' in query_lower or \
                                           any(ph in query_lower for ph in
                                               ['which team', 'which player', 'what team', 'what player',
                                                'what is the name of', 'name the']) or \
                                           re.search(
                                               r'\b(what|which) (is|was) the (longest|shortest|first|last|main|primary)\b.*\b(player|team|name|entity|person|organization|group|city|state|country|location|event|title)\b',
                                               query_lower)
            is_date_query = 'when' in query_lower or 'what date' in query_lower or 'which year' in query_lower or 'what month' in query_lower or 'what day' in query_lower

            if is_count_query or is_difference_query or is_extreme_value_numeric_query:
                instruction = "Your task is to provide ONLY the single, final numerical answer. Do NOT include units (like 'yards' or 'points') unless the question explicitly asks for the unit as part of the answer. Do NOT include any reasoning, explanation, or introductory phrases. For example, if the answer is 7, respond with '7'."
                if is_difference_query:
                    instruction += " If the question asks for a difference, calculate it and provide only the resulting number."
                elif is_extreme_value_numeric_query:
                    instruction += " If the question asks for an extreme value (e.g., longest), provide that specific number."
            elif is_who_which_what_span_query:
                instruction = "Your task is to provide ONLY the name(s) of the player(s), team(s), or specific entity(ies) requested. If multiple distinct entities are requested by the question, separate them with a comma. Do NOT provide explanations or introductory phrases. The answer should be a direct span from the text if possible."
            elif is_date_query:
                instruction = "Your task is to provide ONLY the date. If the answer is a full date, format it as MM/DD/YYYY. If only a year, provide just the YYYY. If only month and year, format as MM/YYYY. Do NOT provide explanations or introductory phrases."
            else:  # Default DROP instruction
                instruction = "Carefully analyze the question and context. Provide the single, most precise answer. This could be a number (output only the number), a short text span (like a name or specific phrase from the text), or a date (MM/DD/YYYY or YYYY). Do NOT include explanations or units unless they are explicitly part of the answer span itself."
            self.logger.debug(f"[QID:{query_id_for_log}] DROP Instruction: {instruction}")
        else:  # Default / HotpotQA
            instruction = "Based ONLY on the context passages and any relevant background information provided, answer the following question accurately and concisely. Provide only the answer, no explanations."

        prompt_parts.append(f"Question: {question.strip()}")
        prompt_parts.append(f"\n{instruction}")
        prompt_parts.append("\n\nPrecise Answer:")  # Clear call to action for the LLM

        final_prompt = "\n\n".join(prompt_parts)

        # --- Prompt Truncation Logic ---
        # This is a simplified version; more sophisticated truncation might be needed
        # depending on tokenizer and model specifics.
        final_chars = len(final_prompt)
        # Estimate max tokens based on average characters per token (e.g., 4).
        # This is a heuristic; actual token count depends on the tokenizer.
        estimated_tokens = final_chars / 4

        # Use tokenizer.model_max_length if available, otherwise self.max_context_length
        # The tokenizer's max length is usually the most reliable for pre-truncation.
        model_max_len = getattr(self.tokenizer, 'model_max_length', self.max_context_length)

        if estimated_tokens > model_max_len * 0.95:  # If estimated tokens are close to or exceed 95% of model max length
            self.logger.warning(
                f"[QID:{query_id_for_log}] Estimated prompt tokens ({estimated_tokens:.0f}) close to model_max_length ({model_max_len}). Attempting to truncate context part if possible.")

            # Try to truncate the context part more aggressively
            # This assumes 'context_parts' holds the main text blocks.
            # A more robust solution would re-calculate non_context_len precisely.

            # Simplified: if context_parts is large, reduce it.
            if len(context_parts) > 1:  # If there's more than one passage
                # Calculate length of non-context parts (few-shot, guidance, question, instructions)
                non_context_prompt_parts = []
                context_block_found = False
                for part in prompt_parts:
                    if "Passage 1:" in part or "[No relevant context passages were retrieved.]" in part:
                        context_block_found = True  # Mark that we've hit the context block
                        continue  # Skip the actual context block for length calculation of other parts
                    if context_block_found and "---" == part.strip() and context_parts:  # Separator after context
                        context_block_found = False  # Reset after context block
                        continue
                    non_context_prompt_parts.append(part)

                non_context_len_chars = len("\n\n".join(non_context_prompt_parts))
                # Target character length for context, aiming for (model_max_len * safety_factor_chars_per_token)
                # Safety factor (e.g., 3.5-4 chars/token)
                target_total_chars = model_max_len * 3.5
                available_for_context_chars = target_total_chars - non_context_len_chars - 100  # -100 for safety margin

                if available_for_context_chars > 0:
                    current_context_str = "\n\n---\n\n".join(context_parts)
                    if len(current_context_str) > available_for_context_chars:
                        truncated_context_str = current_context_str[
                                                :int(available_for_context_chars)] + "... [CONTEXT TRUNCATED]"

                        # Rebuild prompt_parts with truncated context
                        new_prompt_parts_rebuild = []
                        context_header_found_rebuild = False
                        original_context_block_skipped = False

                        for part_str in prompt_parts:
                            if "Based ONLY on the following context passages" in part_str:
                                new_prompt_parts_rebuild.append(part_str)  # Add context header
                                new_prompt_parts_rebuild.append(truncated_context_str)  # Add truncated context
                                context_header_found_rebuild = True
                            elif context_header_found_rebuild and (
                                    "Passage 1:" in part_str or "[No relevant context passages were retrieved.]" in part_str) and not original_context_block_skipped:
                                # This is the original context block, skip it as we've added the truncated one
                                original_context_block_skipped = True
                                continue
                            elif not (
                                    "Passage 1:" in part_str or "[No relevant context passages were retrieved.]" in part_str and context_header_found_rebuild and not original_context_block_skipped):
                                # Add other parts if they are not the original context block that we are replacing
                                new_prompt_parts_rebuild.append(part_str)

                        if context_header_found_rebuild:  # Only if we successfully rebuilt
                            final_prompt = "\n\n".join(new_prompt_parts_rebuild)
                            self.logger.info(
                                f"[QID:{query_id_for_log}] Context aggressively truncated. New prompt length: {len(final_prompt)} chars.")
                        else:
                            self.logger.warning(
                                f"[QID:{query_id_for_log}] Could not find context header for aggressive truncation. Using original long prompt.")
                else:
                    self.logger.warning(
                        f"[QID:{query_id_for_log}] Not enough available space for context after calculating non-context parts for truncation.")
            else:
                self.logger.info(
                    f"[QID:{query_id_for_log}] Prompt length seems acceptable or only one context passage, less aggressive truncation applied by tokenizer if needed.")

        # Final check against tokenizer's max length (the tokenizer will truncate if this is still too long)
        if len(self.tokenizer.encode(final_prompt)) > model_max_len:
            self.logger.warning(
                f"Final prompt (est. tokens: {len(self.tokenizer.encode(final_prompt))}) still exceeds model_max_length ({model_max_len}). Will be truncated by tokenizer.")

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
