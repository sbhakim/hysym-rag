# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from src.reasoners.rg_retriever import RuleGuidedRetriever

logger = logging.getLogger(__name__)


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
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto", load_in_8bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto"
            )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_input_length = 512
        self.guidance_template = (
            "Following these rules:\n{rules}\n\n"
            "Based on the context and rules, provide a precise answer."
        )
        self.prompt_templates = {
            'causal': {
                'prefix': "Let's analyze this step by step, considering the following principles:\n",
                'suffix': "\nBased on these principles and using step-by-step reasoning:\n"
            },
            'factual': {
                'prefix': "Consider these key points:\n",
                'suffix': "\nUsing the above information:\n"
            },
            'exploratory': {
                'prefix': "Taking into account these relevant facts:\n",
                'suffix': "\nLet's explore the answer:\n"
            }
        }
        print(f"Model {model_name} loaded successfully!")
        # Initialize Advanced RG-Retriever with adaptive threshold and paragraph filtering
        self.rg_retriever = RuleGuidedRetriever(
            encoder=self.encoder,
            similarity_threshold=0.4,  # Base similarity threshold
            adaptive_threshold=True,  # Enable adaptive threshold
            context_granularity="paragraph"  # Use paragraph-level filtering
        )

    def format_rule_guidance(self, rules, query_type='factual'):
        """Format rules for guidance."""
        if not rules:
            return ""
        template = self.prompt_templates.get(query_type, self.prompt_templates['factual'])
        formatted_rules = []
        for rule in rules[:3]:
            if isinstance(rule, dict):
                rule_text = rule.get("response", str(rule))
                confidence = rule.get("confidence", 1.0)
                prefix = "Critical principle:" if confidence > 0.8 else "Related principle:"
                formatted_rules.append(f"{prefix} {rule_text}")
            else:
                formatted_rules.append(f"- {str(rule)}")
        return template['prefix'] + "\n".join(formatted_rules) + template['suffix']

    def _determine_query_type(self, question):
        """Determine query type based on keywords."""
        question_lower = question.lower()
        if any(word in question_lower for word in ['why', 'how', 'cause', 'effect', 'lead to']):
            return 'causal'
        elif any(word in question_lower for word in ['what is', 'define', 'explain']):
            return 'exploratory'
        return 'factual'

    def _get_generation_params(self, query_type):
        """Return generation parameters based on query type."""
        params = {
            'causal': {'max_new_tokens': 100, 'temperature': 0.7, 'num_beams': 2},
            'factual': {'max_new_tokens': 80, 'temperature': 0.3, 'num_beams': 1},
            'exploratory': {'max_new_tokens': 90, 'temperature': 0.5, 'num_beams': 2}
        }
        return params.get(query_type, params['factual'])

    def retrieve_answer(self, context, question, symbolic_guidance=None, rule_guided_retrieval=True,
                        query_complexity=0.5):
        """
        Enhanced answer generation with consistent return format.

        Returns:
            str: The generated answer text
        """
        try:
            query_type = self._determine_query_type(question)
            guidance_text = self.format_rule_guidance(symbolic_guidance,
                                                      query_type=query_type) if symbolic_guidance else ""

            if rule_guided_retrieval and symbolic_guidance and hasattr(self, 'rg_retriever'):
                try:
                    context = self.rg_retriever.filter_context_by_rules(
                        context, symbolic_guidance, query_complexity=query_complexity
                    )
                except Exception as e:
                    logger.warning(f"RG-Retriever failed: {str(e)}. Using original context.")

            # Prepare input and generate response
            input_text = f"{guidance_text}\nContext: {context}\nQuestion: {question}\nAnswer:"
            generation_params = self._get_generation_params(query_type)

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, **generation_params)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return response  # Return just the string response

        except Exception as e:
            logger.error(f"Error in retrieve_answer: {str(e)}")
            return "Error generating response."

    def retrieve_answers_batched(self, contexts, questions, symbolic_guidances=None):
        """Batch version of retrieve_answer."""
        batch_inputs = []
        for i in range(len(questions)):
            guidance = symbolic_guidances[i] if symbolic_guidances and i < len(symbolic_guidances) else None
            guidance_text = ""
            if guidance:
                guidance_text = self.format_rule_guidance(guidance)
            if is_causal_query(questions[i]):
                cot_prompt = "Step-by-step causal reasoning:"
                input_text = f"{cot_prompt}\nContext: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:"
            else:
                input_text = f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:"
            if guidance_text:
                input_text = guidance_text + "\n" + input_text
            batch_inputs.append(input_text)
        inputs = self.tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
        decoded = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return decoded

    def encode(self, text):
        """Encode text using SentenceTransformer."""
        try:
            emb = self.encoder.encode(text, convert_to_tensor=True)
            if torch.cuda.is_available():
                emb = emb.to('cuda')
            return emb
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            raise
