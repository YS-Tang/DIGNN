# DIGNN

DIGNN (Deep Interaction Graph Neural Network) is a deep learning framework for molecular property prediction using graph neural networks. It leverages multi-level molecular representations including atoms, bonds, angles, and dihedrals to capture complex interactions in molecular systems.

## Features

- **Multi-scale Representation**: Captures atomic, bond, angle, and dihedral level information
- **Flexible Architecture**: Modular design with encoder, processor, and decoder components
- **Various GNN Backbones**: Supports GCN, GIN, EGAT, Graphormer, and more
- **Efficient Processing**: Optimized for molecular graph computations

## Installation

```bash
git clone https://github.com/yourusername/DIGNN.git
cd DIGNN
pip install -r requirements.txt
```

## Quick Start

```python
from nn.models.dignn import DIGNN
from nn.models.encoder import Encoder
from nn.models.processor import Processor
from nn.models.decoder import Decoder

# Initialize model components
encoder = Encoder(...)
processor = Processor(...)
decoder = Decoder(...)

# Create DIGNN model
model = DIGNN(encoder, processor, decoder)

# Forward pass
output = model(data)
```

## Project Structure

```
DIGNN/
├── calc/           # Calculation utilities
├── data/           # Data processing and atom representations
├── nn/             # Neural network modules
│   ├── convs/      # Graph convolution layers
│   └── models/     # Model architectures
├── utils/          # Utility functions
└── examples/       # Usage examples
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DIGNN in your research, please cite:
```bibtex
@software{dignn2026,
  title={DIGNN: Deep Interaction Graph Neural Network},
  author={Tang, Yushan},
  year={2026}
}
```