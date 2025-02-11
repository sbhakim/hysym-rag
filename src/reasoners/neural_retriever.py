# src/reasoners/neural_retriever.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import logging

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
        # Set a maximum input length for prompt tokenization
        self.max_input_length = 512
        # Original guidance template (fallback)
        self.guidance_template = (
            "Following these rules:\n{rules}\n\n"
            "Based on the context and rules, provide a precise answer."
        )
        # --- New: Enhanced prompt templates for rule-guided generation ---
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

    def format_rule_guidance(self, rules, query_type='factual'):
        """
        Format rules with advanced template selection and rule prioritization.
        """
        if not rules:
            return ""

        # Determine template based on query type
        template = self.prompt_templates.get(query_type, self.prompt_templates['factual'])

        # Process and prioritize rules: limit to top 3 most relevant rules
        formatted_rules = []
        for rule in rules[:3]:
            if isinstance(rule, dict):
                rule_text = rule.get("response", str(rule))
                confidence = rule.get("confidence", 1.0)
                # Format based on confidence level
                if confidence > 0.8:
                    prefix = "Critical principle:"
                else:
                    prefix = "Related principle:"
                formatted_rules.append(f"{prefix} {rule_text}")
            else:
                formatted_rules.append(f"- {str(rule)}")

        # Combine with template parts
        return (
                template['prefix'] +
                "\n".join(formatted_rules) +
                template['suffix']
        )

    def _determine_query_type(self, question):
        """
        Determine query type for template selection.
        """
        question_lower = question.lower()
        if any(word in question_lower for word in ['why', 'how', 'cause', 'effect', 'lead to']):
            return 'causal'
        elif any(word in question_lower for word in ['what is', 'define', 'explain']):
            return 'exploratory'
        return 'factual'

    def _get_generation_params(self, query_type):
        """
        Get generation parameters based on query type.
        """
        params = {
            'causal': {
                'max_new_tokens': 100,
                'temperature': 0.7,
                'num_beams': 2,
            },
            'factual': {
                'max_new_tokens': 80,
                'temperature': 0.3,
                'num_beams': 1,
            },
            'exploratory': {
                'max_new_tokens': 90,
                'temperature': 0.5,
                'num_beams': 2,
            }
        }
        return params.get(query_type, params['factual'])

    def retrieve_answer(self, context, question, symbolic_guidance=None, rule_guided_retrieval=True, similarity_threshold=0.4): # Added similarity_threshold
        """
        Enhanced answer generation with improved rule guidance and RG-Retriever logic using semantic similarity.
        """
        # Determine query type based on question content
        query_type = self._determine_query_type(question)

        # Format guidance with appropriate template if symbolic guidance is provided
        guidance_text = self.format_rule_guidance(symbolic_guidance, query_type=query_type) if symbolic_guidance else ""

        # --- RG-Retriever Logic (Semantic Similarity) ---
        if rule_guided_retrieval and symbolic_guidance: # Apply rule-guided retrieval only if enabled and rules are present
            rule_embeddings = []
            for rule in symbolic_guidance[:3]: # Consider top 3 rules
                if isinstance(rule, dict):
                    rule_text = rule.get("response", str(rule)) # Use rule response for embedding
                elif isinstance(rule, str):
                    rule_text = rule # Use rule string directly
                else:
                    rule_text = ""
                if rule_text:
                    rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True)
                    rule_embeddings.append(rule_embedding)

            if rule_embeddings:
                context_sentences = context.split('.') # Split context into sentences
                filtered_context_sentences = []
                for sentence in context_sentences:
                    sentence_embedding = self.encoder.encode(sentence, convert_to_tensor=True)
                    similarities = [util.cos_sim(sentence_embedding, rule_emb).item() for rule_emb in rule_embeddings]
                    max_similarity = max(similarities) if similarities else 0.0 # Get max similarity to any rule

                    if max_similarity >= similarity_threshold: # Filter based on semantic similarity threshold
                        filtered_context_sentences.append(sentence)

                if filtered_context_sentences:
                    filtered_context = ". ".join(filtered_context_sentences) # Reconstruct filtered context
                    logger.info(f"RG-Retriever (Semantic): Context filtered based on rule similarity (threshold={similarity_threshold}). Original length: {len(context)}, Filtered length: {len(filtered_context)}")
                    context = filtered_context # Use filtered context for generation
                else:
                    logger.info("RG-Retriever (Semantic): No context filtering applied - no sentences above similarity threshold.")
            else:
                logger.info("RG-Retriever (Semantic): No rules embeddings available for context filtering.")


        else:
            logger.info("RG-Retriever: Rule-guided retrieval disabled or no rules provided.")
        # --- End RG-Retriever Logic (Semantic Similarity) ---


        # Construct complete prompt including guidance, context, and question
        input_text = (
            f"{guidance_text}\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # Get generation parameters based on query type
        generation_params = self._get_generation_params(query_type)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            **generation_params
        )

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
