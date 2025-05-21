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

# Import SentenceTransformer for semantic similarity in span filtering
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    _logger_init = logging.getLogger(__name__)
    _logger_init.warning("SentenceTransformer not installed. Span filtering will be limited.")
    util = None

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

        # Initialize SentenceTransformer for semantic similarity in span filtering
        try:
            # Reuse embedder from neural retriever if available
            self.encoder = getattr(self.neural_retriever, 'encoder', None)
            if self.encoder is None:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.logger.info("SentenceTransformer encoder initialized for span filtering.")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SentenceTransformer: {e}. Span filtering will be limited.")
            self.encoder = None
            util = None

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
    ) -> Tuple[Dict[str, Any], str]:
        """Handles the DROP dataset path.
        [Updated May 16, 2025]: Fixed TypeError by ensuring neu_raw_output is converted to string for logging.
        """
        qid = query_id or "unknown"
        self.logger.debug(f"[DROP Start QID:{qid}] Complexity: {query_complexity:.3f}")

        # --- Symbolic Processing ---
        start_sym = time.time()
        sym_obj = {**self._empty_drop(), 'status': 'error', 'rationale': 'Symbolic processing not run'}
        try:
            sym_obj = self._process_symbolic_for_drop(query, context, qid)
            if not isinstance(sym_obj, dict) or 'status' not in sym_obj:
                raise ValueError("Symbolic reasoner returned invalid format.")
        except Exception as e:
            self.logger.error(f"[DROP QID:{qid}] Symbolic processing error: {e}")
            sym_obj = {**self._empty_drop(), 'status': 'error', 'rationale': f"Symbolic Exception: {e}"}
        self.component_times['symbolic'] = time.time() - start_sym
        self.logger.debug(f"[DROP QID:{qid}] Symbolic Result: {sym_obj}")

        # --- Direct High-Confidence Symbolic Return ---
        sym_conf = sym_obj.get('confidence', 0.0)
        if sym_obj.get('status') == 'success' and sym_conf >= 0.85:
            self.logger.info(f"[DROP QID:{qid}] Returning high-confidence symbolic result (Conf: {sym_conf:.2f})")
            answer_obj = {
                'number': sym_obj.get('number', ''),
                'spans': sym_obj.get('spans', []),
                'date': sym_obj.get('date', {'day': '', 'month': '', 'year': ''}),
                'status': 'success',
                'confidence': sym_conf,
                'rationale': sym_obj.get('rationale', 'High-confidence symbolic result'),
                'type': sym_obj.get('type', 'unknown')
            }
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
            # [Added May 16, 2025]: Validate neu_raw_output type
            if not isinstance(neu_raw_output, (str, dict)):
                self.logger.error(f"[DROP QID:{qid}] Neural output invalid type: {type(neu_raw_output)}")
                raise ValueError(f"Neural retriever returned invalid type: {type(neu_raw_output)}")
        except Exception as e:
            self.logger.error(f"[DROP QID:{qid}] Neural retrieval failed: {str(e)}")
            if sym_obj.get('status') == 'success':
                self.logger.info(
                    f"[DROP QID:{qid}] Neural failed, falling back to successful symbolic result (Conf: {sym_conf:.2f})")
                self.component_times['neural'] = time.time() - start_neu
                answer_obj = {
                    'number': sym_obj.get('number', ''),
                    'spans': sym_obj.get('spans', []),
                    'date': sym_obj.get('date', {'day': '', 'month': '', 'year': ''}),
                    'status': 'success',
                    'confidence': sym_conf,
                    'rationale': sym_obj.get('rationale', 'Symbolic fallback due to neural error'),
                    'type': sym_obj.get('type', 'unknown')
                }
                return answer_obj, 'symbolic_fallback_on_neural_error'
            else:
                self.component_times['neural'] = time.time() - start_neu
                return self._error(f"Neural processing failed, no symbolic success: {e}", qid)
        self.component_times['neural'] = time.time() - start_neu
        # [Updated May 16, 2025]: Use str() to prevent TypeError with non-string outputs
        self.logger.debug(f"[DROP QID:{qid}] Neural Raw Output: {str(neu_raw_output)[:100]}...")

        # --- Fusion ---
        start_fus = time.time()
        try:
            answer_obj, final_confidence, fusion_debug = self._fuse_symbolic_neural_for_drop(
                query, sym_obj, neu_raw_output, query_complexity, qid
            )
            fusion_strategy = fusion_debug.get('fusion_strategy_drop', 'unknown')
            source = f'hybrid_drop_{fusion_strategy}'
            self.logger.info(
                f"[DROP QID:{qid}] Fusion successful: Strategy={fusion_strategy}, Final Confidence={final_confidence:.2f}")
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
            error_obj = {
                'number': '',
                'spans': [],
                'date': {'day': '', 'month': '', 'year': ''},
                'status': 'error',
                'confidence': 0.0,
                'rationale': msg,
                'type': 'error'
            }
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
            out.setdefault('number', '')
            out.setdefault('spans', [])
            out.setdefault('date', {'day': '', 'month': '', 'year': ''})
            out.setdefault('confidence', 0.0)
            out.setdefault('rationale', 'No rationale provided.')
            return out
        self.logger.error(f"[DROP QID:{qid}] Symbolic reasoner returned invalid format: {out}. Expected Dict.")
        return {
            'number': '',
            'spans': [],
            'date': {'day': '', 'month': '', 'year': ''},
            'status': 'error',
            'confidence': 0.0,
            'rationale': 'Invalid format from symbolic reasoner',
            'type': 'error'
        }

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
        number = sym_obj.get('number', '')
        spans = sym_obj.get('spans', [])
        date = sym_obj.get('date', {'day': '', 'month': '', 'year': ''})
        status = sym_obj.get('status', 'error')

        if rationale and conf > 0.3:
            guidance.append({'response': f"Symbolic rationale: {rationale}", 'confidence': conf})

        if status == 'success' and conf > 0.4:
            guidance_text = f"Symbolic operation result: Type={op_type}"
            if number:
                guidance_text += f", Number={number}"
            elif spans:
                guidance_text += f", Spans=[{', '.join(spans)}]"
            elif any(date.values()):
                guidance_text += f", Date={date.get('month','')}/{date.get('day','')}/{date.get('year','')}"
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
                                       neural_output_obj: Union[str, Dict[str, Any]],
                                       # Can be raw string or pre-parsed dict
                                       query_complexity: float,
                                       query_id: Optional[str]
                                       ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        qid = query_id or "unknown_qid_fusion"
        debug = {'fusion_strategy_drop': 'init', 'symbolic_input': symbolic_result_obj,
                 'neural_input_type': str(type(neural_output_obj))}
        self.logger.debug(
            f"[DROP QID:{qid}] Starting fusion. Symbolic status: {symbolic_result_obj.get('status')}, Neural input type: {type(neural_output_obj)}")

        # --- Configurable Fusion Thresholds (Consider moving to __init__ or config file) ---
        STRONG_SYM_CONF_THRESHOLD = 0.80  # Symbolic needs to be quite sure to be prioritized
        STRONG_NEU_CONF_THRESHOLD = 0.75  # Neural also needs good confidence
        MIN_USABLE_SYM_CONF = 0.35  # Minimum symbolic confidence to even be considered
        MIN_USABLE_NEU_CONF = 0.45  # Minimum neural confidence from its parser
        # How much more confident neural needs to be to override symbolic in type-match disagreement
        CONF_DELTA_FOR_NEURAL_OVERRIDE_IF_TYPES_MATCH = 0.10
        # How much more confident neural needs to be to override if types *mis-match*
        CONF_DELTA_FOR_NEURAL_OVERRIDE_IF_TYPES_MISMATCH = 0.20
        # Semantic similarity threshold for comparing spans if exact match fails
        SPAN_SIMILARITY_THRESHOLD_FOR_AGREEMENT = 0.75

        # --- Process Symbolic Result ---
        sym_conf = symbolic_result_obj.get('confidence', 0.0)
        sym_op_type = symbolic_result_obj.get('type')
        sym_status = symbolic_result_obj.get('status', 'error')
        sym_rationale = symbolic_result_obj.get('rationale', 'No symbolic rationale.')
        sym_number_val = self._normalize_drop_number_for_comparison(symbolic_result_obj.get('number'))
        sym_spans_val = [str(s).strip() for s in symbolic_result_obj.get('spans', []) if str(s).strip()]
        sym_date_val = symbolic_result_obj.get('date', {})
        sym_value_present_and_valid = (sym_number_val is not None) or \
                                      (sym_spans_val) or \
                                      (isinstance(sym_date_val, dict) and any(
                                          str(v).strip() for v in sym_date_val.values()))

        effective_sym_value_type = None
        if sym_status == 'success' and sym_value_present_and_valid:
            if sym_op_type in ['number', 'count', 'difference', 'extreme_value_numeric',
                               'temporal_difference'] and sym_number_val is not None:
                effective_sym_value_type = 'number'
            elif sym_op_type in ['spans', 'entity_span'] and sym_spans_val:
                effective_sym_value_type = 'spans'
            elif sym_op_type == 'extreme_value':  # Can be number or span
                if sym_number_val is not None:
                    effective_sym_value_type = 'number'
                elif sym_spans_val:
                    effective_sym_value_type = 'spans'
            elif sym_op_type == 'date' and isinstance(sym_date_val, dict) and any(
                    str(v).strip() for v in sym_date_val.values()):
                effective_sym_value_type = 'date'
        debug['symbolic_processed'] = {'conf': sym_conf, 'op_type': sym_op_type, 'status': sym_status,
                                       'value_type': effective_sym_value_type, 'num': sym_number_val,
                                       'spans': sym_spans_val, 'date': sym_date_val}

        # --- Process Neural Result ---
        neu_conf = 0.0
        effective_neu_value_type: Optional[str] = None
        neu_parsed_successfully = False
        neural_rationale = "Neural result processing initiated."
        neu_number_val: Optional[float] = None
        neu_spans_val: List[str] = []
        neu_date_val: Dict[str, str] = {}

        if isinstance(neural_output_obj, dict) and neural_output_obj.get(
                'status') == 'success':  # Already parsed by NeuralRetriever
            neu_conf = neural_output_obj.get('confidence', 0.0)
            effective_neu_value_type = neural_output_obj.get('type')  # This is 'number', 'spans', 'date'
            neural_rationale = neural_output_obj.get('rationale', 'Neural parsed result from structured input')
            neu_number_val = self._normalize_drop_number_for_comparison(neural_output_obj.get('number'))
            neu_spans_val = [str(s).strip() for s in neural_output_obj.get('spans', []) if str(s).strip()]
            raw_date_neu = neural_output_obj.get('date', {})
            if isinstance(raw_date_neu, dict) and any(str(v).strip() for v in raw_date_neu.values()):
                neu_date_val = {k: str(v).strip() for k, v in raw_date_neu.items() if str(v).strip()}
            neu_parsed_successfully = (neu_number_val is not None) or (neu_spans_val) or (
                        neu_date_val and any(neu_date_val.values()))

        elif isinstance(neural_output_obj, str):  # Raw string from LLM, needs parsing
            parsed_from_string = self.neural_retriever._parse_neural_for_drop(neural_output_obj, query,
                                                                              sym_op_type or effective_sym_value_type)
            if parsed_from_string:
                neu_conf = parsed_from_string.get('confidence', 0.6)
                effective_neu_value_type = parsed_from_string.get('type')
                parsed_value_content = parsed_from_string.get('value')
                neural_rationale = "Neural LLM output parsed by HybridIntegrator"

                if effective_neu_value_type == 'number':
                    neu_number_val = self._normalize_drop_number_for_comparison(parsed_value_content)
                elif effective_neu_value_type == 'spans' and isinstance(parsed_value_content, list):
                    neu_spans_val = [str(s).strip() for s in parsed_value_content if str(s).strip()]
                elif effective_neu_value_type == 'date' and isinstance(parsed_value_content, dict):
                    if any(str(v).strip() for v in parsed_value_content.values()):
                        neu_date_val = {k: str(v).strip() for k, v in parsed_value_content.items() if str(v).strip()}
                neu_parsed_successfully = (neu_number_val is not None) or (neu_spans_val) or (
                            neu_date_val and any(neu_date_val.values()))
            else:
                neural_rationale = "Failed to parse raw neural string output by HybridIntegrator's call"
        else:  # Error object or unexpected type
            neural_rationale = f"Unexpected neural output format or error: {str(neural_output_obj)[:100]}"
        debug['neural_processed'] = {'conf': neu_conf, 'parsed_success': neu_parsed_successfully,
                                     'value_type': effective_neu_value_type, 'num': neu_number_val,
                                     'spans': neu_spans_val, 'date': neu_date_val}

        # --- Fusion Decision Logic ---
        final_chosen_value_content: Any = None
        final_chosen_op_type = sym_op_type  # Default to symbolic op type if available
        final_confidence = 0.05  # Default to very low
        final_rationale = 'Fusion: No confident path or agreement found.'
        final_status = 'error'

        # Case 1: Symbolic is strong and provided a valid typed value
        if sym_status == 'success' and sym_conf >= STRONG_SYM_CONF_THRESHOLD and effective_sym_value_type:
            debug['fusion_strategy_drop'] = 'priority_strong_symbolic'
            final_status, final_chosen_op_type, final_chosen_value_content, final_confidence, final_rationale = \
                'success', sym_op_type, symbolic_result_obj.get(
                    effective_sym_value_type), sym_conf, f"Strong symbolic: {sym_rationale}"

        # Case 2: Neural is strong, parsed, and symbolic is weak or failed
        elif neu_parsed_successfully and neu_conf >= STRONG_NEU_CONF_THRESHOLD and \
                (sym_status != 'success' or sym_conf < MIN_USABLE_SYM_CONF or not effective_sym_value_type):
            debug['fusion_strategy_drop'] = 'priority_strong_neural_sym_weak'
            final_status, final_chosen_op_type, final_confidence, final_rationale = \
                'success', effective_neu_value_type, neu_conf, f"Strong neural: {neural_rationale}"
            if effective_neu_value_type == 'number':
                final_chosen_value_content = neu_number_val
            elif effective_neu_value_type == 'spans':
                final_chosen_value_content = neu_spans_val
            elif effective_neu_value_type == 'date':
                final_chosen_value_content = neu_date_val

        # Case 3: Both symbolic and neural have usable results; compare them
        elif sym_status == 'success' and effective_sym_value_type and neu_parsed_successfully and effective_neu_value_type:
            # Subcase 3a: Answer types match
            if effective_sym_value_type == effective_neu_value_type:
                debug['fusion_strategy_drop'] = 'types_match'
                current_comparison_type = effective_sym_value_type
                # Package values for _are_drop_values_equivalent
                sym_comp_dict = {'number': sym_number_val, 'spans': sym_spans_val, 'date': sym_date_val}
                neu_comp_dict = {'number': neu_number_val, 'spans': neu_spans_val, 'date': neu_date_val}

                values_agree_exactly = self._are_drop_values_equivalent(sym_comp_dict, neu_comp_dict,
                                                                        current_comparison_type, qid)

                # Enhanced span agreement: consider semantic similarity if exact fails
                values_agree_semantically_for_spans = False
                if current_comparison_type == 'spans' and not values_agree_exactly and self.encoder and util:
                    if sym_spans_val and neu_spans_val:  # Both must have spans
                        # Convert lists of spans to single strings for comparison
                        sym_span_text = " ".join(sorted(list(set(s.lower() for s in sym_spans_val))))
                        neu_span_text = " ".join(sorted(list(set(s.lower() for s in neu_spans_val))))
                        try:
                            emb1 = self.encoder.encode(sym_span_text, convert_to_tensor=True).to(self.device)
                            emb2 = self.encoder.encode(neu_span_text, convert_to_tensor=True).to(self.device)
                            span_similarity = util.cos_sim(emb1, emb2).item()
                            if span_similarity >= SPAN_SIMILARITY_THRESHOLD_FOR_AGREEMENT:
                                values_agree_semantically_for_spans = True
                                debug['fusion_strategy_drop'] += '_spans_sem_agree'
                        except Exception as sim_err:
                            self.logger.warning(
                                f"[DROP QID:{qid}] Error calculating span semantic similarity for fusion: {sim_err}")

                values_agree = values_agree_exactly or values_agree_semantically_for_spans

                if values_agree:
                    debug['fusion_strategy_drop'] += '_values_agree'
                    final_status, final_chosen_op_type, final_confidence = 'success', sym_op_type, max(sym_conf,
                                                                                                       neu_conf)
                    final_rationale = f"Sym & Neu agreed (Type: {current_comparison_type}, Agreement: {'exact' if values_agree_exactly else 'semantic'}). Conf: {final_confidence:.2f}."
                    # Prefer neural if agreement, as it might be cleaner or more complete
                    if current_comparison_type == 'number':
                        final_chosen_value_content = neu_number_val
                    elif current_comparison_type == 'spans':
                        final_chosen_value_content = neu_spans_val  # Neural spans often better formatted
                    elif current_comparison_type == 'date':
                        final_chosen_value_content = neu_date_val
                else:  # Types match, values disagree
                    debug['fusion_strategy_drop'] += '_values_disagree'
                    if neu_conf >= sym_conf + CONF_DELTA_FOR_NEURAL_OVERRIDE_IF_TYPES_MATCH:
                        final_status, final_chosen_op_type, final_confidence, final_rationale = \
                            'success', effective_neu_value_type, neu_conf, f"Neural preferred on disagreement (type {effective_neu_value_type}): {neural_rationale}"
                        if effective_neu_value_type == 'number':
                            final_chosen_value_content = neu_number_val
                        elif effective_neu_value_type == 'spans':
                            final_chosen_value_content = neu_spans_val
                        elif effective_neu_value_type == 'date':
                            final_chosen_value_content = neu_date_val
                    else:  # Symbolic preferred or confidences too close
                        final_status, final_chosen_op_type, final_confidence, final_rationale = \
                            'success', sym_op_type, sym_conf, f"Symbolic preferred on disagreement (type {effective_sym_value_type}): {sym_rationale}"
                        if effective_sym_value_type == 'number':
                            final_chosen_value_content = sym_number_val
                        elif effective_sym_value_type == 'spans':
                            final_chosen_value_content = sym_spans_val
                        elif effective_sym_value_type == 'date':
                            final_chosen_value_content = sym_date_val
            # Subcase 3b: Answer types mismatch
            else:
                debug['fusion_strategy_drop'] = 'types_mismatch'
                if sym_conf > neu_conf + CONF_DELTA_FOR_NEURAL_OVERRIDE_IF_TYPES_MISMATCH and sym_conf >= MIN_USABLE_SYM_CONF:  # Symbolic significantly more confident
                    final_status, final_chosen_op_type, final_confidence, final_rationale = \
                        'success', sym_op_type, sym_conf, f"Symbolic preferred on type mismatch (SymType: {effective_sym_value_type}, NeuType: {effective_neu_value_type}): {sym_rationale}"
                    if effective_sym_value_type == 'number':
                        final_chosen_value_content = sym_number_val
                    elif effective_sym_value_type == 'spans':
                        final_chosen_value_content = sym_spans_val
                    elif effective_sym_value_type == 'date':
                        final_chosen_value_content = sym_date_val
                elif neu_conf > sym_conf + CONF_DELTA_FOR_NEURAL_OVERRIDE_IF_TYPES_MISMATCH and neu_conf >= MIN_USABLE_NEU_CONF:  # Neural significantly more confident
                    final_status, final_chosen_op_type, final_confidence, final_rationale = \
                        'success', effective_neu_value_type, neu_conf, f"Neural preferred on type mismatch (SymType: {effective_sym_value_type}, NeuType: {effective_neu_value_type}): {neural_rationale}"
                    if effective_neu_value_type == 'number':
                        final_chosen_value_content = neu_number_val
                    elif effective_neu_value_type == 'spans':
                        final_chosen_value_content = neu_spans_val
                    elif effective_neu_value_type == 'date':
                        final_chosen_value_content = neu_date_val
                else:  # Confidence close or both low, unclear which type is correct
                    final_status = 'error'
                    final_rationale = f"Type mismatch and ambiguous confidence. Sym({effective_sym_value_type}, {sym_conf:.2f}), Neu({effective_neu_value_type}, {neu_conf:.2f})."
                    debug['fusion_strategy_drop'] += '_ambiguous_error'
                    final_chosen_op_type = 'error'  # Mark op_type as error

        # Case 4: Only Neural is usable (and not necessarily strong)
        elif neu_parsed_successfully and neu_conf >= MIN_USABLE_NEU_CONF and effective_neu_value_type:
            debug['fusion_strategy_drop'] = 'neural_only_usable'
            final_status, final_chosen_op_type, final_confidence, final_rationale = \
                'success', effective_neu_value_type, neu_conf, f"Neural only usable: {neural_rationale}"
            if effective_neu_value_type == 'number':
                final_chosen_value_content = neu_number_val
            elif effective_neu_value_type == 'spans':
                final_chosen_value_content = neu_spans_val
            elif effective_neu_value_type == 'date':
                final_chosen_value_content = neu_date_val

        # Case 5: Only Symbolic is usable (and not necessarily strong)
        elif sym_status == 'success' and sym_conf >= MIN_USABLE_SYM_CONF and effective_sym_value_type:
            debug['fusion_strategy_drop'] = 'symbolic_only_usable'
            final_status, final_chosen_op_type, final_confidence, final_rationale = \
                'success', sym_op_type, sym_conf, f"Symbolic only usable: {sym_rationale}"
            if effective_sym_value_type == 'number':
                final_chosen_value_content = sym_number_val
            elif effective_sym_value_type == 'spans':
                final_chosen_value_content = sym_spans_val
            elif effective_sym_value_type == 'date':
                final_chosen_value_content = sym_date_val

        # Case 6: Fallback if no path is clearly superior or usable
        else:
            debug['fusion_strategy_drop'] = 'fallback_no_clear_choice'
            final_status = 'error'
            final_rationale = f"Fusion failure: Neither symbolic nor neural provided a usable result above minimum thresholds. Sym: {sym_status}, {sym_conf:.2f}. Neu: parsed={neu_parsed_successfully}, {neu_conf:.2f}."
            final_chosen_op_type = sym_op_type or effective_neu_value_type or 'error'  # Best guess for type
            final_confidence = 0.05  # Lowest confidence

        # Final assembly of the answer object
        final_answer_obj = self._create_drop_answer_obj(final_chosen_op_type, final_chosen_value_content)
        final_answer_obj['status'] = final_status
        final_answer_obj['confidence'] = round(final_confidence, 3)
        final_answer_obj['rationale'] = final_rationale
        final_answer_obj['type'] = final_chosen_op_type if final_status == 'success' else 'error_fusion'

        # Add component timings to the debug info that will be passed to the aggregator
        debug['symbolic_time'] = self.component_times.get('symbolic', 0.0)
        debug['neural_time'] = self.component_times.get('neural', 0.0)  # This is pre-fusion neural time
        # Fusion time is calculated in the calling _handle_drop_path method

        self._update_fusion_metrics(final_confidence, query_complexity, debug)
        self.logger.info(
            f"[DROP QID:{qid}] Fusion Decision: Strategy='{debug['fusion_strategy_drop']}', "
            f"ChosenOpType='{final_answer_obj['type']}', Confidence={final_answer_obj['confidence']:.2f}, Status={final_answer_obj['status']}"
        )
        self.logger.debug(f"[DROP QID:{qid}] Final Fused Object: {final_answer_obj}")

        return final_answer_obj, final_confidence, debug

    def _create_drop_answer_obj(self, answer_type: Optional[str], value: Any) -> Dict[str, Any]:
        """
        Create a DROP answer object with validation.
        Ensures 'number' field contains native int/float (which is then stringified for schema)
        if applicable. Spans are cleaned. Date fields are strings.
        """
        obj = {
            'number': "",  # Standard: empty string for no number
            'spans': [],
            'date': {'day': "", 'month': "", 'year': ""},
            'status': 'success',
            'confidence': 0.5,
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
                obj['type'] = 'error_empty_result'  # More specific error type
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


    def _are_drop_values_equivalent(self,
                                    obj1: Dict[str, Any],
                                    obj2: Dict[str, Any],
                                    value_type: str,
                                    qid: str = "unknown"
                                    ) -> bool:
        """Compares DROP answer values, handling potential None values."""
        try:
            val1 = obj1.get(value_type) if value_type != 'date' else obj1.get('date')
            val2 = obj2.get(value_type) if value_type != 'date' else obj2.get('date')

            if value_type == "number":
                n1 = self._normalize_drop_number_for_comparison(val1)
                n2 = self._normalize_drop_number_for_comparison(val2)
                if n1 is None or n2 is None:
                    self.logger.debug(f"[QID:{qid}] Number comparison failed: One or both values are None (Pred: {val1}, GT: {val2})")
                    return False
                result = abs(n1 - n2) < 1e-6
                self.logger.debug(f"[QID:{qid}] Number comparison: {n1} == {n2} -> {result}")
                return result

            elif value_type == "spans":
                spans1 = set(str(s).strip().lower() for s in val1 if str(s).strip()) if isinstance(val1, list) else set()
                spans2 = set(str(s).strip().lower() for s in val2 if str(s).strip()) if isinstance(val2, list) else set()
                result = spans1 == spans2
                self.logger.debug(f"[QID:{qid}] Span comparison: {spans1} == {spans2} -> {result}")
                return result

            elif value_type == "date":
                if isinstance(val1, dict) and isinstance(val2, dict) and \
                   all(k in val1 for k in ['day','month','year']) and \
                   all(k in val2 for k in ['day','month','year']):
                    result = all(str(val1.get(k,'')).strip() == str(val2.get(k,'')).strip() for k in ['day','month','year'])
                    self.logger.debug(f"[QID:{qid}] Date comparison: {val1} == {val2} -> {result}")
                    return result
                return False

        except Exception as e:
            self.logger.warning(f"[QID:{qid}] Error during DROP value comparison for type '{value_type}': {e}")
            return False

        return False

    def _normalize_drop_number_for_comparison(self,
                                           value_str: Optional[Any]
                                           ) -> Optional[float]:
        """Normalizes numbers (string, int, float) to float for comparison, handles None and errors.
           Mirrors the one in neural_retriever for consistency.
        """
        if value_str is None or (isinstance(value_str, str) and not value_str.strip()):
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
                    match_clean_num = re.match(r'(-?\d+(?:\.\d+)?)', s)
                    if match_clean_num:
                        result = float(match_clean_num.group(1))
                    else:
                        self.logger.debug(f"Integrator: Could not normalize '{value_str}' to a number.")
                        return None
            return result
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Integrator: Error normalizing number '{value_str}': {e}")
            return None

    def _parse_neural_for_drop(self, neural_raw_output: Optional[str], query: str,
                               expected_type_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parses raw neural output string to extract a structured DROP answer.
        Enhanced to prioritize spans for 'who'/'which' queries and aggregate counts for 'both' queries.
        Updated to deduplicate spans, filter using semantic similarity, and whitelist entity types.
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
            # 1. Try Parsing SPANS (prioritize for "who"/"which" queries)
            if answer_type in ['spans', 'entity_span'] or (answer_type == 'extreme_value' and ('who' in query_lower or 'which' in query_lower)):
                # Use spaCy NER on the output if available
                if self.nlp:
                    doc = self.nlp(output)
                    # Define expected NER labels based on query type
                    expected_labels = set()
                    if 'who' in query_lower:
                        expected_labels = {'PERSON'}
                    elif 'team' in query_lower:
                        expected_labels = {'ORG'}
                    elif 'where' in query_lower:
                        expected_labels = {'GPE'}
                    else:
                        expected_labels = {'PERSON', 'ORG', 'GPE', 'NORP'}

                    # Extract spans and filter by expected labels
                    extracted_spans = [ent.text.strip() for ent in doc.ents if ent.label_ in expected_labels]
                    if extracted_spans:
                        # Filter spans using semantic similarity to query
                        if self.encoder and util:
                            query_embedding = self.encoder.encode(query_lower, convert_to_tensor=True)
                            filtered_spans = []
                            for span in extracted_spans:
                                span_embedding = self.encoder.encode(span, convert_to_tensor=True)
                                similarity = util.cos_sim(query_embedding, span_embedding).item()
                                if similarity >= 0.5:  # Threshold for relevance
                                    filtered_spans.append(span)
                            # Deduplicate spans (case-insensitive) while preserving order
                            seen = set()
                            parsed_value = [s for s in filtered_spans if not (s.lower() in seen or seen.add(s.lower()))]
                            confidence = 0.7
                            self.logger.debug(f"Parsed spans via spaCy NER with semantic filtering: {parsed_value}")
                        else:
                            # Deduplicate spans without semantic filtering
                            seen = set()
                            parsed_value = [s for s in extracted_spans if not (s.lower() in seen or seen.add(s.lower()))]
                            confidence = 0.6
                            self.logger.debug(f"Parsed spans via spaCy NER without semantic filtering: {parsed_value}")
                    else:
                        # Fallback: take first significant phrase/noun chunk if no priority NER
                        non_stop_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if not all(tok.is_stop for tok in chunk)]
                        if non_stop_chunks:
                            if self.encoder and util:
                                query_embedding = self.encoder.encode(query_lower, convert_to_tensor=True)
                                best_chunk = None
                                best_similarity = 0.0
                                for chunk in non_stop_chunks:
                                    chunk_embedding = self.encoder.encode(chunk, convert_to_tensor=True)
                                    similarity = util.cos_sim(query_embedding, chunk_embedding).item()
                                    if similarity >= 0.5 and similarity > best_similarity:
                                        best_chunk = chunk
                                        best_similarity = similarity
                                if best_chunk:
                                    parsed_value = [best_chunk]
                                    confidence = 0.5
                                    self.logger.debug(f"Parsed span via spaCy Noun Chunk with semantic filtering: {parsed_value}")
                            else:
                                parsed_value = [non_stop_chunks[0]]
                                confidence = 0.5
                                self.logger.debug(f"Parsed span via spaCy Noun Chunk without semantic filtering: {parsed_value}")
                # Fallback if no spaCy or no results: Take first line/sentence fragment
                if parsed_value is None:
                    first_line = output.split('\n')[0].strip('.,!? ')
                    if len(first_line) > 1 and len(first_line.split()) < 10:
                        if self.encoder and util:
                            query_embedding = self.encoder.encode(query_lower, convert_to_tensor=True)
                            first_line_embedding = self.encoder.encode(first_line, convert_to_tensor=True)
                            similarity = util.cos_sim(query_embedding, first_line_embedding).item()
                            if similarity >= 0.5:
                                parsed_value = [first_line]
                                confidence = 0.4
                                self.logger.debug(f"Parsed span via first line heuristic with semantic filtering: {parsed_value}")
                        else:
                            parsed_value = [first_line]
                            confidence = 0.4
                            self.logger.debug(f"Parsed span via first line heuristic without semantic filtering: {parsed_value}")

            # 2. Try Parsing NUMBER (including count, difference, extreme_value as number)
            elif answer_type in ['number', 'count', 'difference'] or (answer_type == 'extreme_value' and 'who' not in query_lower and 'which' not in query_lower):
                # Aggregate all numbers in the output (e.g., for "both teams" queries)
                numbers = []
                matches = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', output)
                for match in matches:
                    num_str = match.replace(',', '')
                    num = self._normalize_drop_number_for_comparison(num_str)
                    if num is not None:
                        numbers.append(num)
                        self.logger.debug(f"Parsed number via regex: {num}")

                if numbers:
                    # For "difference" queries, take the first number (assumed to be the difference)
                    if 'difference' in query_lower:
                        parsed_value = numbers[0]
                        confidence = 0.75
                        self.logger.debug(f"Extracted difference number: {parsed_value}")
                    else:
                        # For "how many" queries, sum the numbers (e.g., for "both teams")
                        parsed_value = sum(numbers)
                        confidence = 0.75
                        self.logger.debug(f"Aggregated numbers for count: {parsed_value}")
                else:
                    # Fallback: check word numbers if regex fails
                    first_word = output.split()[0].lower().strip('.,!?')
                    num_from_word = self._normalize_drop_number_for_comparison(first_word)
                    if num_from_word is not None:
                        parsed_value = num_from_word
                        confidence = 0.65
                        self.logger.debug(f"Parsed number via word: {parsed_value}")

            # 3. Try Parsing DATE
            elif answer_type == 'date':
                try:
                    # Use dateutil for robust parsing
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