# HySym-RAG: Hybrid Symbolic and Neural Reasoning for Advanced Knowledge Integration

**HySym-RAG** (Hybrid Symbolic Reasoning with Neural Assisted Graph) is a cutting-edge framework combining symbolic reasoning and neural networks for efficient, adaptive AI systems. Designed for complex knowledge integration, it optimizes resource usage while maintaining high reasoning fidelity.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Key Features

✨ **Hybrid Intelligence**  
- Combines graph-based symbolic rules with transformer neural models
- Dynamic path selection between symbolic/neural processing

⚡ **Energy Efficiency**  
- Real-time resource monitoring (CPU/GPU/Memory)
- Adaptive computation strategies
- Power-aware scheduling

🧠 **Knowledge Integration**  
- Bidirectional rule-embedding alignment
- Automatic rule generation from neural outputs
- Multi-hop reasoning with context preservation

📊 **Evaluation Metrics**  
- Energy-Reasoning Quality (ERQ) score
- Hybrid performance analytics
- Resource utilization tracking

## Architecture

```mermaid
graph TD
    A[User Query] --> B(Resource Manager)
    B --> C{Query Complexity}
    C -->|Low| D[Symbolic Reasoner]
    C -->|High| E[Neural Retriever]
    D --> F[Rule-Based Inference]
    E --> G[LLM Processing]
    F & G --> H{Result Confidence}
    H -->|High| I[Output Result]
    H -->|Low| J[Hybrid Verification]
    J --> K[Final Answer]
 ```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for neural components)
- CUDA 11.x (for GPU acceleration)

### Setup

1. Clone repository:
```
   git clone https://github.com/sbhakim/hysym-rag.git
   cd hysym-rag
```


2. Install dependencies:
```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm en_core_web_md
```

## Quick Start

1. Run the system:
   ```bash
   python src/main.py
   python src/main.py --query "What are the environmental effects of deforestation?"
    ```
## Project Structure

```bash

HySym-RAG/
├── src/                     # Main source code
│   ├── main.py              # Entry point for running the system
│   ├── app.py               # Core application logic
│   ├── config/              # Configuration files
│   ├── reasoners/           # Symbolic and neural reasoning modules
│   ├── integrators/         # Hybrid integration logic
│   ├── resources/           # Resource management utilities
│   ├── queries/             # Query expansion and logging utilities
│   └── utils/               # Helper scripts and evaluation utilities
├── data/                    # Input data and extracted rules
│   ├── deforestation.txt    # Sample knowledge base
│   ├── rules.json           # Extracted rules
│   └── ground_truths.json   # Ground truth data for evaluation
├── tests/                   # Unit and integration tests
├── examples/                # Example queries and outputs
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and usage instructions
└── LICENSE                  # Project license
    
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- spaCy for NLP pipelines
- Hugging Face for transformer models
- PyTorch for deep learning framework
- NetworkX for graph operations