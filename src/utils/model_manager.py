# src/utils/model_manager.py
import torch
import gc
import psutil
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        try:
            # Load the tokenizer first.
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
            if not self._check_memory_safe():
                self._emergency_cleanup()
            # Load the neural model with a stricter GPU memory limit.
            self.neural_model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m",
                device_map="auto",
                torch_dtype=torch.float16,
                max_memory={0: '8GB'}  # Adjusted limit to 8GB
            )
            if not self._check_memory_safe():
                self._emergency_cleanup()
            # Load the SentenceTransformer last.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            logger.info("ModelManager: Successfully loaded all models.")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            self._emergency_cleanup()
            raise

    def _check_memory_safe(self, threshold=0.85):
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            reserved = torch.cuda.memory_reserved(0)
            total = props.total_memory
            if reserved > threshold * total:
                return False
        process = psutil.Process()
        if process.memory_percent() / 100 > threshold:
            return False
        return True

    def _emergency_cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("ModelManager: Emergency cleanup performed.")
