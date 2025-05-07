# src/integrators/hybrid_integrator.py

import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import defaultdict
import re
from dateutil import parser as date_parser

# Conditional imports for essential components
try:
    from src.knowledge_integrator import AlignmentLayer
    from src.reasoners.rg_retriever import RuleGuidedRetriever
    from src.utils.device_manager import DeviceManager
    from src.utils.dimension_manager import DimensionalityManager
except ImportError as e:
    # logger instantiation needs to happen before logging error
    _logger_init = logging.getLogger(__name__)
    _logger_init.error(f"ImportError in hybrid_integrator.py: {e}. Check paths.")
    AlignmentLayer = None
    RuleGuidedRetriever = None
    DeviceManager = None
    DimensionalityManager = None

# Initialize spaCy for parsing neural outputs
try:
    import spacy
    # Check if model is already loaded to avoid redundant loading/warnings
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    _logger_init = logging.getLogger(__name__)
    _logger_init.warning("spaCy not installed or model not downloaded. Neural output parsing will be limited.")
    nlp = None
except Exception as spacy_load_err:
    _logger_init = logging.getLogger(__name__)
    _logger_init.error(f"Error loading spaCy model: {spacy_load_err}")
    nlp = None

logger = logging.getLogger(__name__)  # Define logger for the rest of the class

class HybridIntegrator:
    """
    HybridIntegrator orchestrates symbolic and neural components,
    supporting both text-based QA (e.g., HotpotQA) and discrete
    reasoning QA (e.g., DROP).
    """

    def __init__(
        self,
        symbolic_reasoner,
        neural_retriever,
        query_expander: Optional[Any] = None,
        cache_ttl: int = 3600,
        fusion_threshold: float = 0.6,  # Used for text fusion, DROP has its own logic
        dim_manager: Optional[DimensionalityManager] = None,
        dataset_type: Optional[str] = None
    ):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.query_expander = query_expander
        self.fusion_threshold = fusion_threshold
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)  # Use DEBUG for more verbose logging during development

        # Ensure utilities are available
        if DeviceManager is None or DimensionalityManager is None:
            raise RuntimeError("DeviceManager or DimensionalityManager failed to import.")

        self.device = DeviceManager.get_device()
        self.dim_manager = dim_manager or DimensionalityManager(target_dim=768, device=self.device)

        self.dataset_type = dataset_type.lower().strip() if isinstance(dataset_type, str) else None
        self.logger.info(f"HybridIntegrator initialized for dataset type: {self.dataset_type}")

        # AlignmentLayer for text fusion
        self.alignment_layer = None
        if self.dataset_type != 'drop' and AlignmentLayer is not None:
            try:
                # Use actual dimensions from dim_manager if available, else defaults
                sym_dim = getattr(self.dim_manager, 'sym_dim', 384)  # Assuming default symbolic dim
                neural_dim = getattr(self.dim_manager, 'neural_dim', 768)  # Assuming default neural dim
                target_dim = self.dim_manager.target_dim
                self.alignment_layer = AlignmentLayer(
                    sym_dim=sym_dim,
                    neural_dim=neural_dim,
                    target_dim=target_dim,
                    dim_manager=self.dim_manager
                ).to(self.device)
                self.logger.info("AlignmentLayer initialized for text-based fusion.")
            except Exception as e:
                self.logger.error(f"Failed to initialize AlignmentLayer: {e}")
                self.alignment_layer = None
        elif self.dataset_type == 'drop':
            self.logger.info("AlignmentLayer not initialized (DROP path).")
        else:
            self.logger.warning("AlignmentLayer class not available.")

        # RuleGuidedRetriever (context filtering)
        self.rg_retriever = None
        if RuleGuidedRetriever is not None:
            try:
                # Pass the embedder from symbolic or neural if available
                encoder = getattr(self.symbolic_reasoner, 'embedder', None) or \
                          getattr(self.neural_retriever, 'encoder', None)
                if encoder:
                    self.rg_retriever = RuleGuidedRetriever(encoder=encoder)
                    self.logger.info("RuleGuidedRetriever initialized.")
                else:
                    self.logger.warning("No embedder found in reasoners for RuleGuidedRetriever.")
            except Exception as e:
                self.logger.error(f"Failed to initialize RuleGuidedRetriever: {e}")
        else:
            self.logger.warning("RuleGuidedRetriever class not available.")

        # Metrics and caches
        self.fusion_metrics = {
            'confidence': [],
            'query_complexity_at_fusion': [],
            'attention_patterns': [],
            'text_fusion_strategies_counts': defaultdict(int),
            'drop_fusion_strategies_counts': defaultdict(int)
        }
        self.integration_metrics = {
            'alignment_scores': [], 'fusion_quality': [],
            'reasoning_steps': [], 'path_lengths': [],
            'resource_usage': defaultdict(list)
        }
        self.reasoning_chains: Dict[str, Any] = defaultdict(dict)
        self.cache: Dict[str, Tuple[Any, str, float]] = {}  # Store (result, source, timestamp)
        self.cache_ttl = cache_ttl
        self.component_times = {"symbolic": 0.0, "neural": 0.0, "fusion": 0.0}
        # Store spaCy instance
        self.nlp = nlp  # Use the globally loaded nlp instance

    def process_query(
        self,
        query: str,
        context: str,
        query_complexity: float = 0.5,
        supporting_facts: Optional[List[Tuple[str,int]]] = None,
        query_id: Optional[str] = None
    ) -> Tuple[Any, str]:
        """
        Main entry: dispatches to DROP or text QA path.
        Returns (answer_obj, processing_source).
        """
        qid = query_id or "unknown"
        self.logger.debug(f"[Start QID:{qid}] '{query[:50]}...' (type={self.dataset_type}) Complexity: {query_complexity:.3f}")

        # Validate inputs
        if not query or not isinstance(query, str) or not query.strip():
            return self._error("Invalid query (must be non-empty string).", qid)
        if not context or not isinstance(context, str) or not context.strip():
            return self._error("Invalid context (must be non-empty string).", qid)
        if not self.symbolic_reasoner or not self.neural_retriever:
            return self._error("Core reasoner components not initialized.", qid)

        # Cache Check First
        key = self._generate_cache_key(query, context, qid)
        cached = self._get_cache(key)
        if cached:
            self.logger.info(f"Cache hit for QID {qid}: {key}")
            return cached  # Return cached (result, source) tuple

        # Reset component times for this query
        self.component_times = {"symbolic": 0.0, "neural": 0.0, "fusion": 0.0}

        # Dispatch based on dataset type
        if self.dataset_type == 'drop':
            try:
                result, source = self._handle_drop_path(query, context, query_complexity, qid)
            except Exception as e:
                self.logger.exception(f"[DROP QID:{qid}] Critical error in _handle_drop_path: {e}")
                result, source = self._error(f"DROP Exception: {e}", qid)
        else:  # Default to text path
            try:
                result, source = self._handle_text_path(query, context, query_complexity, supporting_facts, qid)
            except Exception as e:
                self.logger.exception(f"[Text QID:{qid}] Critical error in _handle_text_path: {e}")
                result, source = self._error(f"Text QA Exception: {e}", qid)

        # Cache the result if successful
        if source != 'error' and result is not None:
            self._set_cache(key, (result, source))

        # Log component times for analysis
        self.logger.debug(f"[Times QID:{qid}] Sym: {self.component_times['symbolic']:.3f}s, Neu: {self.component_times['neural']:.3f}s, Fus: {self.component_times['fusion']:.3f}s")
        return result, source

    def _handle_text_path(
         self, query: str, context: str, query_complexity: float,
         supporting_facts: Optional[List[Tuple[str,int]]], qid: str
     ) -> Tuple[Any, str]:
        """Handles the text QA path (e.g., HotpotQA)."""
        # --- Symbolic Processing ---
        start_sym = time.time()
        sym_list = []
        try:
            sym_list = self._process_symbolic_for_text(query, context, qid)
        except Exception as e:
            self.logger.error(f"[Text QID:{qid}] Symbolic processing error: {e}")
            # Allow continuation, neural path might still work
        self.component_times['symbolic'] = time.time() - start_sym

        # --- Neural Processing (+ Context Filtering) ---
        start_neu = time.time()
        neu = "Error: Neural processing failed"  # Default neural result
        try:
            guid = self._get_textual_symbolic_guidance(sym_list)
            ctx = context
            # Optional context filtering
            if self.rg_retriever:
                try:
                    ctx = self.rg_retriever.filter_context_by_rules(
                        ctx, guid, query_complexity=query_complexity, supporting_facts=supporting_facts
                    )
                except Exception as rg_err:
                    self.logger.warning(f"[Text QID:{qid}] RuleGuidedRetriever failed: {rg_err}. Using original context.")

            # Retrieve answer using neural model
            neu = self.neural_retriever.retrieve_answer(
                ctx, query, symbolic_guidance=guid,
                query_complexity=query_complexity, dataset_type=self.dataset_type  # Pass dataset type
            )
        except Exception as e:
            self.logger.error(f"[Text QID:{qid}] Neural retrieval error: {e}")
            # If neural fails, rely on symbolic if available
            if sym_list:
                self.component_times['neural'] = time.time() - start_neu
                return (" ".join(sym_list), 'symbolic_fallback')
            else:
                return self._error(f"Neural processing failed: {e}", qid)

        self.component_times['neural'] = time.time() - start_neu  # Initial neural time before fusion

        # --- Fusion ---
        start_fus = time.time()
        final_result: Any
        source: str
        if self.alignment_layer:
            try:
                fused_result, conf, dbg = self._fuse_symbolic_neural_for_text(query, sym_list, neu, query_complexity, qid)
                source = 'hybrid_text'
                final_result = fused_result  # Use the fused result
                self._update_fusion_metrics(conf, query_complexity, dbg)
            except Exception as e:
                self.logger.error(f"[Text QID:{qid}] Text fusion failed: {e}. Using fallback.")
                final_result = self._create_fallback_text_fusion(query, sym_list, neu)
                source = 'fusion_fallback'  # Indicate fusion failed
        else:
            # No alignment layer, decide based on content
            final_result = self._create_fallback_text_fusion(query, sym_list, neu)
            source = 'neural_dominant' if neu else ('symbolic_dominant' if sym_list else 'no_result')

        self.component_times['fusion'] = time.time() - start_fus
        # Adjust neural time to exclude fusion time
        self.component_times['neural'] -= self.component_times['fusion']

        return final_result, source

    def _handle_drop_path(
        self, query: str, context: str,
        query_complexity: float, query_id: Optional[str]
    ) -> Tuple[Dict[str,Any], str]:
        """Handles the DROP dataset path."""
        qid = query_id or "unknown"
        self.logger.debug(f"[DROP Start QID:{qid}] Complexity: {query_complexity:.3f}")

        # --- Symbolic Processing ---
        start_sym = time.time()
        sym_obj = {**self._empty_drop(), 'status':'error','rationale':'Symbolic processing not run'}
        try:
            sym_obj = self._process_symbolic_for_drop(query, context, query_id)
            if not isinstance(sym_obj, dict) or 'status' not in sym_obj:
                raise ValueError("Symbolic reasoner returned invalid format.")
        except Exception as e:
            self.logger.error(f"[DROP QID:{qid}] Symbolic processing error: {e}")
            sym_obj = {**self._empty_drop(), 'status':'error','rationale':f"Symbolic Exception: {e}"}
        self.component_times['symbolic'] = time.time() - start_sym
        self.logger.debug(f"[DROP QID:{qid}] Symbolic Result: {sym_obj}")

        # --- Direct High-Confidence Symbolic Return ---
        sym_conf = sym_obj.get('confidence', 0.0)
        if sym_obj.get('status') == 'success' and sym_conf >= 0.85:
            self.logger.info(f"[DROP QID:{qid}] Returning high-confidence symbolic result (Conf: {sym_conf:.2f})")
            answer_obj = self._create_drop_answer_obj(sym_obj.get('type'), sym_obj.get('value'))
            answer_obj.update({
                'status': 'success',
                'confidence': sym_conf,
                'rationale': sym_obj.get('rationale', 'High-confidence symbolic result'),
                'type': sym_obj.get('type')
            })
            return answer_obj, 'symbolic_high_conf'

        # --- Neural Processing ---
        start_neu = time.time()
        neu_raw_output = "Error: Neural processing failed"
        try:
            guid = self._get_drop_symbolic_guidance(sym_obj, query)
            neu_raw_output = self.neural_retriever.retrieve_answer(
                context, query, symbolic_guidance=guid,
                query_complexity=query_complexity, dataset_type=self.dataset_type
            )
        except Exception as e:
            self.logger.error(f"[DROP QID:{qid}] Neural retrieval failed: {str(e)}")
            if sym_obj.get('status') == 'success':
                self.logger.info(f"[DROP QID:{qid}] Neural failed, falling back to successful symbolic result (Conf: {sym_conf:.2f})")
                self.component_times['neural'] = time.time() - start_neu
                answer_obj = self._create_drop_answer_obj(sym_obj.get('type'), sym_obj.get('value'))
                answer_obj.update({
                    'status': 'success',
                    'confidence': sym_conf,
                    'rationale': sym_obj.get('rationale', 'Symbolic fallback due to neural error'),
                    'type': sym_obj.get('type')
                })
                return answer_obj, 'symbolic_fallback_on_neural_error'
            else:
                return self._error(f"Neural processing failed, no symbolic success: {e}", qid)
        self.component_times['neural'] = time.time() - start_neu
        self.logger.debug(f"[DROP QID:{qid}] Neural Raw Output: {neu_raw_output[:100]}...")

        # --- Fusion ---
        start_fus = time.time()
        try:
            answer_obj, final_confidence, fusion_debug = self._fuse_symbolic_neural_for_drop(
                query, sym_obj, neu_raw_output, query_complexity, qid
            )
            fusion_strategy = fusion_debug.get('fusion_strategy_drop', 'unknown')
            source = f'hybrid_drop_{fusion_strategy}'
            self.logger.info(f"[DROP QID:{qid}] Fusion successful: Strategy={fusion_strategy}, Final Confidence={final_confidence:.2f}")

        except Exception as e:
            self.logger.exception(f"[DROP QID:{qid}] Fusion critical error: {e}")
            answer_obj, source = self._error(f"DROP Fusion Exception: {e}", qid)

        self.component_times['fusion'] = time.time() - start_fus
        self.component_times['neural'] -= self.component_times['fusion']

        return answer_obj, source

    def _error(self, msg: str, qid: str) -> Tuple[Any,str]:
        """Logs error and returns structured error object."""
        self.logger.error(f"[Error QID:{qid}] {msg}")
        if self.dataset_type=='drop':
            error_obj = self._empty_drop()
            error_obj['error'] = msg
            return error_obj, 'error'
        return f"Error: {msg}", 'error'

    def _process_symbolic_for_text(self, query, context, qid) -> List[str]:
        """Calls symbolic reasoner for text, returns list of response strings."""
        out = self.symbolic_reasoner.process_query(
            query, context=context, dataset_type='text', query_id=qid
        )
        if isinstance(out, list) and all(isinstance(s, str) for s in out):
            return [s.strip() for s in out if s.strip()]
        self.logger.warning(f"[Text QID:{qid}] Symbolic reasoner returned unexpected format: {type(out)}. Expected List[str].")
        return []

    def _process_symbolic_for_drop(self, query, context, qid) -> Dict[str,Any]:
        """Calls symbolic reasoner for DROP, returns structured dict."""
        out = self.symbolic_reasoner.process_query(
            query=query, context=context, dataset_type='drop', query_id=qid
        )
        if isinstance(out, dict) and 'status' in out:
            out.setdefault('type', None)
            out.setdefault('value', None)
            out.setdefault('confidence', 0.0)
            out.setdefault('rationale', 'No rationale provided.')
            return out
        self.logger.error(f"[DROP QID:{qid}] Symbolic reasoner returned invalid format: {out}. Expected Dict.")
        return {**self._empty_drop(), 'status':'error','rationale':'Invalid format from symbolic reasoner'}

    def _is_multi_hop_query(self, query: str) -> bool:
        """Determines if a text query likely requires multi-hop reasoning."""
        if self.query_expander and hasattr(self.query_expander,'determine_query_type'):
            try:
                qt = self.query_expander.determine_query_type(query)
                return qt in {'bridge','comparison','temporal_multi','causal_multi','composite', 'multi-hop'}
            except Exception as e:
                self.logger.warning(f"Error determining query type via expander: {e}")
        indicators = [" and "," then "," after "," before ", " result in ", " because ", " compare ", " difference "]
        cnt = sum(1 for i in indicators if i in query.lower())
        wh_words = sum(1 for wh in ['who','what','where','when','which','how'] if wh in query.lower().split())
        return cnt >= 1 or wh_words > 1

    def _handle_single_hop_text_query(
        self, query, context, sym_list, supporting_facts, qid
    ) -> Tuple[str,str]:
        """Handles single-hop text queries."""
        return self._handle_text_path(query, context, 0.5, supporting_facts, qid)

    def _handle_multi_hop_text_query(
        self, query, context, sym_list, supporting_facts, qid
    ) -> Tuple[str,str]:
        """Handles multi-hop text queries (simple decomposition fallback)."""
        self.logger.info(f"[Text MultiHop QID:{qid}] Attempting decomposition.")
        hops = self._extract_textual_reasoning_chain(query)
        if len(hops) <= 1:
            self.logger.warning(f"[Text MultiHop QID:{qid}] Decomposition failed, treating as single hop.")
            return self._handle_single_hop_text_query(query, context, sym_list, supporting_facts, qid)

        results = []
        intermediate_context = ""
        orig_cq = self.query_expander.get_query_complexity(query) if self.query_expander else 0.7

        for i, step in enumerate(hops):
            step_query = step.get('question', '')
            if not step_query:
                continue

            current_context = context + ("\nIntermediate Results:\n" + intermediate_context if intermediate_context else '')
            step_guidance = self._get_textual_symbolic_guidance(sym_list + results)

            filtered_step_context = current_context
            if self.rg_retriever:
                try:
                    filtered_step_context = self.rg_retriever.filter_context_by_rules(
                        current_context, step_guidance, query_complexity=orig_cq
                    )
                except Exception as rg_err:
                    self.logger.warning(f"[Text MultiHop QID:{qid} Step {i}] RG Retriever failed: {rg_err}")

            step_result = "Error: Step processing failed"
            try:
                step_result = self.neural_retriever.retrieve_answer(
                    filtered_step_context, step_query, symbolic_guidance=step_guidance,
                    query_complexity=orig_cq, dataset_type=self.dataset_type
                )
                if step_result and not step_result.startswith("Error"):
                    results.append(step_result)
                    intermediate_context += f"\n- Step {i+1} Answer: {step_result}"
                else:
                    self.logger.warning(f"[Text MultiHop QID:{qid} Step {i}] Failed: Query='{step_query[:50]}...' Result='{step_result}'")
            except Exception as e:
                self.logger.error(f"[Text MultiHop QID:{qid} Step {i}] Error: {e}")

        if not results:
            self.logger.warning(f"[Text MultiHop QID:{qid}] No successful steps, falling back to single hop on original query.")
            return self._handle_single_hop_text_query(query,context,sym_list,supporting_facts,qid)

        final_combined_text = self._combine_text_hop_results(results, hops)

        start_fus = time.time()
        final_fused_result: Any
        source: str
        if self.alignment_layer:
            try:
                final_fused_result, conf, dbg = self._fuse_symbolic_neural_for_text(query, sym_list, final_combined_text, orig_cq, qid)
                source = 'hybrid_multihop'
                self._update_fusion_metrics(conf, orig_cq, dbg)
            except Exception as e:
                self.logger.error(f"[Text MultiHop QID:{qid}] Final fusion failed: {e}")
                final_fused_result = final_combined_text
                source = 'multihop_no_fusion'
        else:
            final_fused_result = final_combined_text
            source = 'multihop_no_fusion'

        self.component_times['fusion'] = time.time() - start_fus
        return final_fused_result, source

    def _get_textual_symbolic_guidance(self, sym_list: List[str]) -> List[Dict[str,Any]]:
        """Generates guidance list for text QA from symbolic results."""
        guidance = []
        sorted_sym = sorted([s for s in sym_list if s and s.strip()], key=len)
        for i, t in enumerate(sorted_sym[:3]):
            guidance.append({'response': t, 'confidence': 0.75 - i*0.05})
        return guidance

    def _get_drop_symbolic_guidance(self, sym_obj: Optional[Dict[str, Any]], query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generate symbolic guidance for DROP queries from symbolic results.
        Focuses on rationale and operation type/value if successful.
        """
        guidance = []
        if not sym_obj:
            self.logger.debug(f"No symbolic object for guidance: {query[:50]}...")
            return None

        conf = sym_obj.get('confidence', 0.0)
        rationale = sym_obj.get('rationale')
        op_type = sym_obj.get('type')
        value = sym_obj.get('value')
        status = sym_obj.get('status', 'error')

        if rationale and conf > 0.3:
            guidance.append({'response': f"Symbolic rationale: {rationale}", 'confidence': conf})

        if status == 'success' and conf > 0.4:
            guidance_text = f"Symbolic operation result: Type={op_type}"
            if op_type == 'number' and value is not None:
                guidance_text += f", Value={value}"
            elif op_type == 'spans' and isinstance(value, list) and value:
                guidance_text += f", Spans=[{', '.join(value)}]"
            elif op_type == 'date' and isinstance(value, dict):
                guidance_text += f", Date={value.get('month','')}/{value.get('day','')}/{value.get('year','')}"
            if len(guidance_text) > len(f"Symbolic operation result: Type={op_type}"):
                guidance.append({'response': guidance_text, 'confidence': conf})

        self.logger.debug(f"[DROP QID:{sym_obj.get('query_id','unknown')}] Generated {len(guidance)} guidance items.")
        return guidance[:3] if guidance else None

    def _extract_textual_reasoning_chain(self, query: str) -> List[Dict[str,str]]:
        """Placeholder for extracting textual reasoning chain (e.g., from QueryExpander)."""
        parts = [p.strip() for p in query.split(' and ') if len(p.strip()) > 5]
        if len(parts) > 1:
            return [{'question': part, 'type': 'sub_and'} for part in parts]
        if self.nlp:
            sents = [s.text.strip() for s in self.nlp(query).sents if s.text.strip()]
            if len(sents) > 1:
                return [{'question': s, 'type': 'sub_sent'} for s in sents]
        return [{'question': query, 'type': 'single'}]

    def _combine_text_hop_results(self, results: List[str], steps: List[Dict[str,str]]) -> str:
        """Combines results from multi-hop text query processing."""
        combined = ""
        for i, res in enumerate(results):
            step_q = steps[i].get('question', f'Step {i+1}')
            combined += f"Result for '{step_q[:30]}...': {res}\n"
        return combined.strip()

    def _fuse_symbolic_neural_for_text(
        self, query: str, sym_list: List[str], neu: str,
        query_complexity: float, qid: Optional[str]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Fuses symbolic and neural results for text QA using AlignmentLayer if available."""
        debug_info = {'fusion_strategy_text': 'init', 'symbolic_used': bool(sym_list), 'neural_used': bool(neu)}
        symbolic_text = ' '.join(sym_list[:2]).strip()
        neural_text = neu.strip() if neu else ""

        if not symbolic_text and not neural_text:
            debug_info['fusion_strategy_text'] = 'no_input'
            return ('No answer available.', 0.1, debug_info)

        if not self.alignment_layer:
            final_response = self._create_fallback_text_fusion(query, symbolic_text, neural_text)
            conf = 0.7 if neural_text else (0.5 if symbolic_text else 0.1)
            debug_info['fusion_strategy_text'] = 'fallback_no_alignment_layer'
            self._update_fusion_metrics(conf, query_complexity, debug_info)
            return (final_response, conf, debug_info)

        try:
            sym_emb = self._get_symbolic_embedding(symbolic_text or "none")
            neu_emb = self._get_neural_embedding(neural_text or "none")

            if sym_emb is None or neu_emb is None:
                raise ValueError("Failed to get embeddings for fusion.")

            sym_emb = sym_emb.unsqueeze(0) if sym_emb.dim() == 1 else sym_emb
            neu_emb = neu_emb.unsqueeze(0) if neu_emb.dim() == 1 else neu_emb

            sym_emb_aligned = self.dim_manager.align_embeddings(sym_emb, f"sym_{qid}")
            neu_emb_aligned = self.dim_manager.align_embeddings(neu_emb, f"neu_{qid}")

            fused_embedding, fusion_confidence, alignment_debug = self.alignment_layer(
                sym_emb_aligned, neu_emb_aligned, rule_confidence=query_complexity
            )
            debug_info.update(alignment_debug)

            final_response = self._generate_reasoned_response_for_text(
                query, sym_list, neural_text, fusion_confidence
            )
            debug_info['fusion_strategy_text'] = f'alignment_layer_conf_{fusion_confidence:.2f}'
            debug_info['final_choice'] = 'neural_dominant' if fusion_confidence >= self.fusion_threshold and neural_text else 'symbolic_dominant'

            self._update_fusion_metrics(fusion_confidence, query_complexity, debug_info)
            return (final_response, fusion_confidence, debug_info)

        except Exception as e:
            self.logger.exception(f"[Text QID:{qid}] AlignmentLayer fusion failed: {e}")
            debug_info['error'] = str(e)
            debug_info['fusion_strategy_text'] = 'error_in_alignment'
            final_response = self._create_fallback_text_fusion(query, symbolic_text, neural_text)
            conf = 0.4
            self._update_fusion_metrics(conf, query_complexity, debug_info)
            return (final_response, conf, debug_info)

    def _fuse_symbolic_neural_for_drop(self,
                                       query: str,
                                       symbolic_result_obj: Dict[str, Any],
                                       neural_raw_output: str,
                                       query_complexity: float,
                                       query_id: Optional[str]
                                       ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """
        Fuse symbolic and neural outputs for DROP queries, returning a structured answer.
        Enhanced fusion logic with improved type handling and decision criteria.
        """
        qid = query_id or "unknown"
        debug = {'fusion_strategy_drop': 'init'}
        sym_conf = symbolic_result_obj.get('confidence', 0.0)
        sym_type = symbolic_result_obj.get('type')
        sym_value = symbolic_result_obj.get('value')
        sym_status = symbolic_result_obj.get('status', 'error')

        # Parse neural output with improved logic
        neu_parsed = self._parse_neural_for_drop(neural_raw_output, query, sym_type)
        neu_conf = neu_parsed.get('confidence', 0.0) if neu_parsed else 0.0
        neu_type = neu_parsed.get('type') if neu_parsed else None
        neu_value = neu_parsed.get('value') if neu_parsed else None

        self.logger.debug(f"[DROP QID:{qid}] Fusion Input - Symbolic: status={sym_status}, type={sym_type}, conf={sym_conf:.2f}, value={str(sym_value)[:50]}")
        self.logger.debug(f"[DROP QID:{qid}] Fusion Input - Neural Parsed: type={neu_type}, conf={neu_conf:.2f}, value={str(neu_value)[:50]}")

        final_answer_obj: Dict[str, Any]
        final_confidence: float
        final_source_type: Optional[str] = None

        # --- Decision Logic ---

        # 1. High-Confidence, Successful Symbolic Result
        if sym_status == 'success' and sym_conf >= 0.85 and sym_type in ['number', 'spans', 'date', 'count', 'difference', 'extreme_value']:
            debug['fusion_strategy_drop'] = 'prioritized_strong_symbolic'
            final_answer_obj = self._create_drop_answer_obj(sym_type, sym_value)
            final_confidence = sym_conf
            final_source_type = 'symbolic'

        # 2. Both Symbolic and Neural Succeeded
        elif sym_status == 'success' and neu_parsed and neu_conf > 0.6:
            sym_answer_obj = self._create_drop_answer_obj(sym_type, sym_value)
            neu_answer_obj = self._create_drop_answer_obj(neu_type, neu_value)

            # Check if types are compatible
            effective_sym_type = sym_type
            if sym_type in ['count', 'difference', 'extreme_value']:
                effective_sym_type = 'number' if isinstance(sym_value, (int, float, str)) and str(sym_value).replace('.', '').isdigit() else 'spans'
            effective_neu_type = neu_type

            if effective_sym_type == effective_neu_type:
                if self._are_drop_values_equivalent(sym_answer_obj, neu_answer_obj, effective_sym_type):
                    debug['fusion_strategy_drop'] = 'agreed_sym_neu'
                    final_answer_obj = neu_answer_obj
                    final_confidence = max(sym_conf, neu_conf)
                    final_source_type = 'agreed'
                else:
                    debug['fusion_strategy_drop'] = 'disagreed_sym_neu_types_match'
                    if neu_conf >= sym_conf:
                        final_answer_obj = neu_answer_obj
                        final_confidence = neu_conf
                        final_source_type = 'neural'
                    else:
                        final_answer_obj = sym_answer_obj
                        final_confidence = sym_conf
                        final_source_type = 'symbolic'
            else:
                # Types differ, prioritize higher confidence
                debug['fusion_strategy_drop'] = 'types_mismatch_select_higher_conf'
                if neu_conf >= sym_conf:
                    final_answer_obj = neu_answer_obj
                    final_confidence = neu_conf
                    final_source_type = 'neural'
                else:
                    final_answer_obj = sym_answer_obj
                    final_confidence = sym_conf
                    final_source_type = 'symbolic'

        # 3. Neural Result is Strong and Parsed Successfully
        elif neu_parsed and neu_conf > 0.6 and neu_type in ['number', 'spans', 'date']:
            debug['fusion_strategy_drop'] = 'parsed_neural_dominant'
            final_answer_obj = self._create_drop_answer_obj(neu_type, neu_value)
            final_confidence = neu_conf
            final_source_type = 'neural'

        # 4. Symbolic Succeeded (but lower conf), Neural Failed/Low Conf
        elif sym_status == 'success' and sym_conf > 0.35:
            debug['fusion_strategy_drop'] = 'weak_symbolic_fallback_no_good_neural'
            final_answer_obj = self._create_drop_answer_obj(sym_type, sym_value)
            final_confidence = sym_conf
            final_source_type = 'symbolic'

        # 5. Neural Parsed (but lower conf), Symbolic Failed
        elif neu_parsed and neu_conf > 0.35:
            debug['fusion_strategy_drop'] = 'weak_parsed_neural_fallback_no_good_symbolic'
            final_answer_obj = self._create_drop_answer_obj(neu_type, neu_value)
            final_confidence = neu_conf
            final_source_type = 'neural'

        # 6. Failure Case
        else:
            debug['fusion_strategy_drop'] = 'failure_no_confident_source'
            self.logger.warning(f"[DROP QID:{qid}] Fusion failed: No valid symbolic or neural result found.")
            final_answer_obj = self._create_drop_answer_obj('error', 'No confident answer found by either method.')
            final_confidence = 0.1
            final_source_type = 'none'

        self._update_fusion_metrics(final_confidence, query_complexity, debug)
        self.logger.info(f"[DROP QID:{qid}] Fusion Decision: Strategy='{debug['fusion_strategy_drop']}', Source='{final_source_type}', Confidence={final_confidence:.2f}")
        return final_answer_obj, final_confidence, debug

    def _create_drop_answer_obj(self, answer_type: Optional[str], value: Any) -> Dict[str, Any]:
        """
        Create a DROP answer object with validation.
        Handles 'extreme_value', 'count', and 'difference' types appropriately.
        """
        obj = self._empty_drop()
        try:
            if answer_type in ["number", "count", "difference"]:
                num_val = self._normalize_drop_number_for_comparison(value)
                obj["number"] = str(num_val) if num_val is not None else ""
                if obj["number"]:
                    self.logger.debug(f"Created DROP answer: number={obj['number']}")
                else:
                    self.logger.warning(f"Invalid number value for DROP obj: {value}")

            elif answer_type == "extreme_value":
                # Determine if the extreme_value result should be a number or spans based on value
                if isinstance(value, (int, float, str)) and str(value).replace('.', '').isdigit():
                    num_val = self._normalize_drop_number_for_comparison(value)
                    obj["number"] = str(num_val) if num_val is not None else ""
                    self.logger.debug(f"Created DROP answer (extreme_value as number): number={obj['number']}")
                elif isinstance(value, list):
                    obj["spans"] = [str(v).strip() for v in value if str(v).strip()]
                    self.logger.debug(f"Created DROP answer (extreme_value as spans): spans={obj['spans']}")
                else:
                    self.logger.warning(f"Invalid extreme_value value: {value}")
                    obj["error"] = f"Invalid extreme_value value: {value}"

            elif answer_type in ["spans", "entity_span"]:
                spans_in = value if isinstance(value, list) else ([value] if value is not None else [])
                obj["spans"] = [str(v).strip() for v in spans_in if str(v).strip()]
                self.logger.debug(f"Created DROP answer: type={answer_type}, spans={obj['spans']}")

            elif answer_type == "date":
                if isinstance(value, dict) and all(k in value for k in ['day', 'month', 'year']):
                    try:
                        d = int(value.get('day', '')) if value.get('day', '') else 0
                        m = int(value.get('month', '')) if value.get('month', '') else 0
                        y = int(value.get('year', '')) if value.get('year', '') else 0
                        if (1 <= d <= 31 or d == 0) and (1 <= m <= 12 or m == 0) and (1000 <= y <= 3000 or y == 0):
                            obj["date"] = {k: str(v).strip() for k, v in value.items() if k in ['day', 'month', 'year']}
                            self.logger.debug(f"Created DROP answer: date={obj['date']}")
                        else:
                            raise ValueError("Invalid date components")
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid date component values: {value}")
                else:
                    self.logger.warning(f"Invalid date dictionary value: {value}")

            elif answer_type == "error":
                obj["error"] = str(value).strip() if value else "Unknown error"
                self.logger.debug(f"Created DROP error answer: {obj['error']}")

            else:
                error_msg = f"Unsupported or invalid answer type: {answer_type}"
                self.logger.warning(error_msg)
                obj["error"] = error_msg

            return obj

        except Exception as e:
            self.logger.error(f"Error creating DROP answer object (Type: {answer_type}, Value: {value}): {str(e)}")
            error_obj = self._empty_drop()
            error_obj['error'] = f"Internal error creating answer object: {str(e)}"
            return error_obj

    def _are_drop_values_equivalent(self,
                                    obj1: Dict[str, Any],
                                    obj2: Dict[str, Any],
                                    value_type: str
                                    ) -> bool:
        """Compares DROP answer values, handling potential None values."""
        try:
            val1 = obj1.get(value_type) if value_type != 'date' else obj1.get('date')
            val2 = obj2.get(value_type) if value_type != 'date' else obj2.get('date')

            if value_type == "number":
                n1 = self._normalize_drop_number_for_comparison(val1)
                n2 = self._normalize_drop_number_for_comparison(val2)
                if n1 is None or n2 is None:
                    return False
                return abs(n1 - n2) < 1e-6

            elif value_type == "spans":
                spans1 = set(str(s).strip().lower() for s in val1 if str(s).strip()) if isinstance(val1, list) else set()
                spans2 = set(str(s).strip().lower() for s in val2 if str(s).strip()) if isinstance(val2, list) else set()
                return spans1 == spans2

            elif value_type == "date":
                if isinstance(val1, dict) and isinstance(val2, dict) and \
                   all(k in val1 for k in ['day','month','year']) and \
                   all(k in val2 for k in ['day','month','year']):
                    return all(str(val1.get(k,'')).strip() == str(val2.get(k,'')).strip() for k in ['day','month','year'])
                return False

        except Exception as e:
            self.logger.warning(f"Error during DROP value comparison for type '{value_type}': {e}")
            return False

        return False

    def _normalize_drop_number_for_comparison(self,
                                              value_str: Optional[Any]
                                              ) -> Optional[float]:
        """Normalizes numbers (string, int, float) to float for comparison, handles None and errors."""
        if value_str is None:
            return None
        try:
            if isinstance(value_str, (int, float)):
                return float(value_str)

            s = str(value_str).replace(",", "").strip().lower()
            if not s:
                return None

            words = {
                "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
                "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
                "ten": 10.0
            }
            if s in words:
                return words[s]

            if re.fullmatch(r'-?\d+(\.\d+)?', s):
                return float(s)
            else:
                self.logger.debug(f"Could not normalize '{value_str}' to a number.")
                return None

        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error normalizing number '{value_str}': {e}")
            return None

    def _parse_neural_for_drop(self, neural_raw_output: Optional[str], query: str,
                               expected_type_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parses raw neural output string to extract a structured DROP answer.
        Enhanced to handle numerical outputs for count, difference, and extreme_value.
        """
        if not neural_raw_output or not neural_raw_output.strip():
            self.logger.debug(f"Cannot parse empty neural output for query: {query[:50]}...")
            return None

        output = neural_raw_output.strip()
        query_lower = query.lower()
        # Determine Expected Answer Type
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
                answer_type = 'spans'

        self.logger.debug(f"Parsing neural output. Determined/Hinted Answer Type: {answer_type}. Raw Output: '{output[:100]}...'")

        parsed_value: Any = None
        confidence = 0.0

        try:
            # 1. Parse NUMBER (including count, difference, extreme_value as number)
            if answer_type in ['number', 'count', 'difference'] or (answer_type == 'extreme_value' and 'who' not in query_lower and 'which' not in query_lower):
                # Regex for leading number, handles commas and decimals
                match = re.match(r'^\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\b', output)
                if match:
                    num_str = match.group(1).replace(',', '')
                    parsed_value = self._normalize_drop_number_for_comparison(num_str)
                    if parsed_value is not None:
                        confidence = 0.75
                        self.logger.debug(f"Parsed number via regex: {parsed_value}")
                # Fallback: check word numbers
                if parsed_value is None:
                    first_word = output.split()[0].lower().strip('.,!?')
                    num_from_word = self._normalize_drop_number_for_comparison(first_word)
                    if num_from_word is not None:
                        parsed_value = num_from_word
                        confidence = 0.65
                        self.logger.debug(f"Parsed number via word: {parsed_value}")

            # 2. Parse SPANS (including extreme_value as spans)
            elif answer_type in ['spans', 'entity_span'] or (answer_type == 'extreme_value' and ('who' in query_lower or 'which' in query_lower)):
                if self.nlp:
                    doc = self.nlp(output)
                    priority_labels = {'PERSON', 'ORG', 'GPE', 'NORP'}
                    extracted_spans = [ent.text.strip() for ent in doc.ents if ent.label_ in priority_labels]
                    if extracted_spans:
                        parsed_value = extracted_spans
                        confidence = 0.7
                        self.logger.debug(f"Parsed spans via spaCy NER: {parsed_value}")
                    else:
                        non_stop_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if not all(tok.is_stop for tok in chunk)]
                        if non_stop_chunks:
                            parsed_value = [non_stop_chunks[0]]
                            confidence = 0.5
                            self.logger.debug(f"Parsed span via spaCy Noun Chunk: {parsed_value}")
                if parsed_value is None:
                    first_line = output.split('\n')[0].strip('.,!? ')
                    if len(first_line) > 1 and len(first_line.split()) < 10:
                        parsed_value = [first_line]
                        confidence = 0.4
                        self.logger.debug(f"Parsed span via first line heuristic: {parsed_value}")

            # 3. Parse DATE
            elif answer_type == 'date':
                try:
                    parsed = date_parser.parse(output, fuzzy=True)
                    parsed_value = {
                        'day': str(parsed.day),
                        'month': str(parsed.month),
                        'year': str(parsed.year)
                    }
                    confidence = 0.75
                    self.logger.debug(f"Parsed date via dateutil: {parsed_value}")
                except (ValueError, OverflowError) as date_err:
                    self.logger.debug(f"Dateutil parsing failed: {date_err}. Trying regex fallback.")
                    date_pattern = r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}|\d{4})\b'
                    match = re.search(date_pattern, output)
                    if match:
                        try:
                            parsed = date_parser.parse(match.group(0), fuzzy=True)
                            parsed_value = {'day': str(parsed.day), 'month': str(parsed.month), 'year': str(parsed.year)}
                            confidence = 0.6
                            self.logger.debug(f"Parsed date via regex fallback: {parsed_value}")
                        except (ValueError, OverflowError):
                            self.logger.debug("Regex date match could not be parsed by dateutil.")

            # Return parsed result if successful
            if parsed_value is not None:
                return {'type': answer_type, 'value': parsed_value, 'confidence': confidence}
            else:
                self.logger.debug(f"Failed to parse neural output for expected type '{answer_type}'. Output: '{output[:50]}...'")
                return None

        except Exception as e:
            self.logger.exception(f"Error parsing neural output: {str(e)}. Raw output: '{neural_raw_output[:100]}...'")
            return None

    def _create_fallback_text_fusion(self, query: str, sym_txt: str, neu_txt: str) -> str:
        """Creates fallback text response when alignment/fusion fails."""
        if neu_txt and len(neu_txt.split()) > 2:
            self.logger.debug("Using neural text as fallback.")
            return neu_txt
        if sym_txt:
            self.logger.debug("Using symbolic text as fallback.")
            return sym_txt
        self.logger.warning("No valid text from neural or symbolic for fallback.")
        return "Unable to provide a confident answer based on available information."

    def _generate_reasoned_response_for_text(self, query: str, sym_list: List[str], neu: str, conf: float) -> str:
        """Generates final text response based on fusion confidence."""
        if conf >= self.fusion_threshold and neu and len(neu.split()) > 1:
            return neu
        elif neu and len(neu.split()) > 1:
            return neu
        elif sym_list:
            return ' '.join(sym_list)
        return f"Unable to provide a confident answer (Confidence Score: {conf:.2f})"

    def _update_fusion_metrics(self, confidence: float, query_complexity: float, debug_info: Dict[str, Any]) -> None:
        """Updates internal metrics related to the fusion process."""
        try:
            conf_float = float(confidence) if confidence is not None else 0.0
            self.fusion_metrics['confidence'].append(conf_float)

            comp_float = float(query_complexity) if query_complexity is not None else 0.0
            self.fusion_metrics['query_complexity_at_fusion'].append(comp_float)

            strat_t = debug_info.get('fusion_strategy_text')
            strat_d = debug_info.get('fusion_strategy_drop')
            if strat_t:
                self.fusion_metrics['text_fusion_strategies_counts'][str(strat_t)] += 1
            if strat_d:
                self.fusion_metrics['drop_fusion_strategies_counts'][str(strat_d)] += 1

            if 'attention_weights' in debug_info:
                self.fusion_metrics['attention_patterns'].append(True)
            else:
                self.fusion_metrics['attention_patterns'].append(False)

        except Exception as e:
            self.logger.error(f"Error updating fusion metrics: {e}")

    def _get_embedding(self, text: str, source_type: str, qid: Optional[str]) -> Optional[torch.Tensor]:
        """Helper to get embedding from appropriate source with error handling."""
        embedder = None
        if source_type == 'symbolic':
            embedder = getattr(self.symbolic_reasoner, 'embedder', None)
        elif source_type == 'neural':
            embedder = getattr(self.neural_retriever, 'encoder', None)

        if not embedder:
            self.logger.warning(f"Embedder not found for source type '{source_type}' (QID: {qid})")
            return None
        try:
            if not text or not text.strip():
                self.logger.warning(f"Attempting to embed empty text for source '{source_type}' (QID: {qid})")
                return None
            return embedder.encode(text, convert_to_tensor=True).to(self.device)
        except Exception as e:
            self.logger.error(f"Error getting {source_type} embedding (QID: {qid}): {e}")
            return None

    def _get_symbolic_embedding(self, text: str, qid: Optional[str] = None) -> Optional[torch.Tensor]:
        """Gets embedding using the symbolic reasoner's embedder."""
        return self._get_embedding(text, 'symbolic', qid)

    def _get_neural_embedding(self, text: str, qid: Optional[str] = None) -> Optional[torch.Tensor]:
        """Gets embedding using the neural retriever's encoder."""
        return self._get_embedding(text, 'neural', qid)

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Returns aggregated metrics about the integration performance."""
        out: Dict[str, Any] = {}
        for k, v in self.integration_metrics.items():
            if isinstance(v, list) and v and all(isinstance(i, (int, float)) for i in v):
                try:
                    out[k] = {
                        'mean': float(np.mean(v)),
                        'std': float(np.std(v)),
                        'count': len(v)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not compute stats for metric '{k}': {e}")
        fc = self.fusion_metrics.get('confidence', [])
        out['fusion_confidence'] = {
            'mean': float(np.mean(fc)) if fc else 0.0,
            'std': float(np.std(fc)) if fc else 0.0,
            'count': len(fc)
        }
        out['text_fusion_strategy_counts'] = dict(self.fusion_metrics.get('text_fusion_strategies_counts', {}))
        out['drop_fusion_strategy_counts'] = dict(self.fusion_metrics.get('drop_fusion_strategies_counts', {}))
        return out

    def _generate_cache_key(self, query: str, context: str, qid: Optional[str] = None) -> str:
        """Generates a cache key incorporating query, context hash, dataset type, and optionally QID."""
        prefix = qid if qid else hashlib.sha1(query.encode('utf-8')).hexdigest()[:16]
        context_hash = hashlib.sha1(context.encode('utf-8')).hexdigest()[:16]
        return f"{self.dataset_type or 'unknown'}_{prefix}_{context_hash}"

    def _get_cache(self, key: str) -> Optional[Tuple[Any, str]]:
        """Retrieves from cache if entry exists and is not expired."""
        entry = self.cache.get(key)
        if entry:
            result_tuple, timestamp = entry
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Cache hit: Key={key}")
                if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                    return result_tuple
                else:
                    self.logger.warning(f"Invalid cached value format for key {key}. Removing.")
                    del self.cache[key]
            else:
                self.logger.debug(f"Cache expired: Key={key}")
                del self.cache[key]
        return None

    def _set_cache(self, key: str, val: Tuple[Any, str]) -> None:
        """Sets cache value with timestamp, managing cache size."""
        if not (isinstance(val, tuple) and len(val) == 2):
            self.logger.error(f"Attempted to cache invalid value format for key {key}. Value: {val}")
            return

        if len(self.cache) >= 1000:
            try:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.logger.debug(f"Cache limit reached. Evicted oldest entry: {oldest_key}")
            except Exception as e:
                self.logger.error(f"Error during cache eviction: {e}")

        self.cache[key] = (val, time.time())

    def _empty_drop(self) -> Dict[str, Any]:
        """Return an empty DROP answer object dictionary."""
        return {"number": "", "spans": [], "date": {"day": "", "month": "", "year": ""}}