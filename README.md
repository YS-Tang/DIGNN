# DIGNN

**Deconstruction-Integration Graph Neural Networks for data-efficient and out-of-distribution prediction of atomistic material properties**

DIGNN is a graph neural network framework for modeling structure–property relationships of atomistic systems (crystals, molecules, and clusters). It is designed to overcome two long-standing bottlenecks of conventional GNNs: (1) the inability to **explicitly** capture high-order geometric features such as bond angles and dihedral angles, and (2) the lack of a synergistic mechanism for **cross-scale** information fusion. These properties make DIGNN especially strong in small-data learning and out-of-distribution (OOD) extrapolation.

> This repository accompanies the (unpublished) manuscript *"Deconstruction-Integration Graph Neural Networks for data-Efficient and out-of-distribution prediction of atomistic material properties"* by Yushan Tang, Haoqi Chen, Jiale Cao, Bo Li, and Zhe Li.

## Key ideas

DIGNN follows an **encoding → deconstruction → integration → decoding** pipeline built from three core components (see `nn/models/dignn.py`):

1. **Preprocessing / Encoder** (`nn.models.Encoder`)
   Converts an atomic structure into multi-order features:
   - `At` — atomic element embedding (`nn.Embedding`)
   - `Bd` — bond length, encoded by a radial basis function (RBF) with cosine decay
   - `Ag` — bond angle (RBF over `[0, π]`)
   - `Dh` — dihedral angle (RBF over `[0, π]`)
   - `BondI` — a second bond-length encoding used by the long-range module under an expanded cutoff

2. **Serial information interaction / Processor** (`nn.models.*_Processor`)
   - **HGC — Hierarchical Geometric Coupler.** A stack of **progressive message-passing layers (PMLs)** operating on multi-level line graphs (AtomG / BondG / AngleG). Information is propagated bidirectionally following the *dihedral → angle → bond → atom* hierarchy, explicitly deconstructing high-order geometry back into atom embeddings within a short-range cutoff (`pml_rcut`).
   - **LCP — Local Coordination Propagator.** A stack of **interaction message-passing layers (IMLs)** that integrate the geometry-aware atom features over a larger cutoff (`iml_rcut`), coordinating fragmented local environments and capturing longer-range interactions.

3. **Task-adaptive decoder** (`nn.models.Decoder` / `Global_Decoder`)
   - Per-atom decoding via an MLP (e.g. charges, magnetic moments), or
   - System-level decoding via pooling (`mean` for intensive quantities, `sum` for extensive quantities) followed by an MLP.
   - Atomic **forces** are obtained by automatic differentiation of the predicted energy w.r.t. atomic positions (energy-conserving force field, see `pl.TrainModule_FF` and `calc.Calculator`).

The dual-cutoff HGC + LCP design gives the model a natural "divide-and-conquer" generalization capability across scales.

## Project structure

```
DIGNN/
├── calc/
│   └── ase.py                  # ASE Calculator (energy & forces) for MD / structure search
├── data/
│   ├── Atoms.py                # AtomsData: PyG Data subclass; topology & geometry construction (PBC-aware)
│   └── utils.py                # ase<->AtomsData conversion, batching, graph-completion utilities
├── nn/
│   ├── convs/                  # Message-passing layers: GatedGCN, GINE, GATv2, EGAT, ALIGNN, MGN, Graphormer, ...
│   └── models/
│       ├── dignn.py            # DIGNN = Encoder + Processor + Decoder
│       └── modules/
│           ├── encoder.py      # Encoder (embedding + RBF encodings)
│           ├── processor.py    # BaseProcessor + GCN/GINE/GATv2/EGAT processors
│           ├── processor_components.py  # HGC & LCP
│           └── decoder.py      # Decoder, Global_Decoder, pooling
├── pl/
│   ├── dataset.py              # DataModule (PyTorch Lightning): split, preprocess, batch
│   └── trainer.py              # TrainModule (property) & TrainModule_FF (energy+force)
├── utils/
│   ├── utils.py                # AtomIndexMapper, differentiable clamp, index helpers
│   ├── line_graph.py           # multi-level line-graph construction
│   ├── sign.py                 # sign bookkeeping for angle/dihedral directions
│   └── feature_collect.py      # latent-feature hook collector
├── visualize/                  # plot_comparison, plot_tsne, plot_umap
├── examples/
│   ├── simple_train.ipynb      # property & force-field training walkthrough (QM7 subset)
│   ├── jarvis.py               # end-to-end JARVIS-DFT property training script
│   └── qm7_1000samples.xyz     # small demo dataset
└── requirements.txt
```

## Installation

DIGNN is a research codebase; import it as a package (the examples add its parent directory to `sys.path`).

```bash
git clone https://github.com/YS-Tang/DIGNN.git
```

### Dependencies

Core requirements (Python ≥ 3.10 recommended):

- `torch` (≤ 2.8.x)
- `torch_geometric`
- `pytorch_lightning`
- `ase`
- `fairchem-core` (used for PBC radius graphs: `radius_graph_pbc_v2`, `get_pbc_distances`)
- `numpy`, `pandas`, `scikit-learn`, `tqdm`, `joblib`
- `matplotlib`, `scikit-learn` (t-SNE), `umap-learn` (UMAP) — for `visualize/`

> Note: preprocessing is much faster on GPU. Set the environment variable `DIGNN_ENV=cuda` (or `cuda:0`) before importing DIGNN; otherwise a CPU warning is printed and preprocessing will be slow.

## Quick start

### 1. Build a dataset

Convert ASE `Atoms` into `AtomsData`. `check_rcut` should match the short-range (PML) cutoff so that connectivity is validated.

```python
import os
os.environ["DIGNN_ENV"] = "cuda:0"   # set before importing DIGNN

import numpy as np
from ase.io import read
from DIGNN.data import ase2AtomsData
from DIGNN.utils import AtomIndexMapper
from DIGNN.pl import DataModule, TrainModule
from DIGNN.nn import models as dgm
import pytorch_lightning as pl

# hyper-parameters
HP_atomsdata = {"pml_rcut": 3.0, "pml_mnn": 12, "iml_rcut": 6.0, "iml_mnn": 24}
HP_feat_dim  = {"atom_dim": 64, "bond_dim": 64, "ang_dim": 32, "dih_dim": 16}
HP_nn        = {"init": 1, "pml": 2, "iml": 4, "decoder": [64, 1], "pooling": "sum"}

frames = read("examples/qm7_1000samples.xyz", index=":", format="extxyz")
for a in frames:
    a.arrays["energy"] = a.get_potential_energy()
atomsdata = [ase2AtomsData(a, check_rcut=HP_atomsdata["pml_rcut"], properties=["energy"])
             for a in frames]
```

- `pml_rcut` / `pml_mnn`: cutoff radius and max neighbors for the **HGC** (short-range, high-order geometry).
- `iml_rcut` / `iml_mnn`: cutoff radius and max neighbors for the **LCP** (long-range interaction).
- For crystals, provide `cell` / `pbc` and set `if_pbc=True`; keep crystal and molecular systems in separate batches. Systems with fewer than 4 atoms are not supported.

### 2. Create the DataModule

`DataModule` splits, preprocesses, and pre-batches the graphs. `return_type='cplt'` produces fully expanded interaction graphs (best for property prediction); `return_type='basic'` keeps a compact form and rebuilds graphs on the fly (used for force-field training so that positions can carry gradients).

```python
data = DataModule(atomsdata,
                  **HP_atomsdata,
                  test_size=0.2, val_size=0.1,
                  batch_size=8, num_workers=1, store_device="cpu",
                  mapper=AtomIndexMapper(),
                  return_type="cplt")
data.setup()
```

### 3. Assemble the model

```python
model = dgm.DIGNN(
    encoder=dgm.Encoder(num_species=data.mapper.num_embeddings,
                        **HP_feat_dim,
                        pml_rcut=HP_atomsdata["pml_rcut"] + 0.2,
                        bondI_dim=HP_feat_dim["bond_dim"],
                        iml_rcut=HP_atomsdata["iml_rcut"] + 0.2),
    processor=dgm.GCN_Processor(**HP_feat_dim,
                                pml=HP_nn["pml"], iml=HP_nn["iml"],
                                residual=True, dropout=0.0,
                                bondI_dim=HP_feat_dim["bond_dim"],
                                init_nn_layer=HP_nn["init"]),
    decoder=dgm.Decoder(dim=[HP_feat_dim["atom_dim"]] + HP_nn["decoder"],
                        reduce_method=HP_nn["pooling"], dropout=0.0),
).to("cuda:0")
```

Available processor backbones (interchangeable): `GCN_Processor` (GatedGCN, default), `GINE_Processor`, `GATv2_Processor`, `EGAT_Processor`.

### 4. Train (property prediction)

```python
tm = TrainModule(model, prop="energy", lr=1e-2,
                 onecycle_total_steps=20 * len(data.train_dataloader()),
                 compile_model=False)  # set True on Linux for speed

trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=[0],
                     precision="16-mixed", gradient_clip_val=1.0)
trainer.fit(tm, train_dataloaders=data.train_dataloader(),
                val_dataloaders=data.val_dataloader())
trainer.test(tm, dataloaders=data.test_dataloader())
```

`examples/jarvis.py` is a complete script for training on any of the ~50 JARVIS-DFT properties.

### 5. Train a force field (energy + forces)

Use `return_type='basic'` and `TrainModule_FF`, which recomputes graphs with `pos.requires_grad=True` and derives forces from `-∂E/∂r`.

```python
from DIGNN.pl import TrainModule_FF

data = DataModule(atomsdata, **HP_atomsdata, return_type="basic",
                  batch_size=8, mapper=AtomIndexMapper())
data.setup()

tm = TrainModule_FF(model, lr=1e-3, energy_weight=0.1, force_weight=1.0,
                    onecycle_total_steps=5 * len(data.train_dataloader()))
trainer = pl.Trainer(max_epochs=5, accelerator="gpu",
                     inference_mode=False)   # required: forces need gradients at eval
trainer.fit(tm, train_dataloaders=data.train_dataloader(),
                val_dataloaders=data.val_dataloader())
```

### 6. Use as an ASE calculator / structure search

A trained DIGNN force field can be wrapped as an ASE `Calculator` for relaxations, MD, or coupling with global structure search engines (e.g. CALYPSO), as done for the Au cluster study in the paper.

```python
from DIGNN.calc.ase import Calculator

calc = Calculator(model, mapper=data.mapper,
                  pml_rcut=3.0, pml_mnn=12, iml_rcut=6.0, iml_mnn=24,
                  device="cuda:0")
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### 7. Visualization & latent features

```python
from DIGNN.visualize import plot_comparison, plot_tsne, plot_umap

preds, targets = tm.test_results.values()
plot_comparison(target=targets, pred=preds)

features, labels = tm.extract_features(
    dataloader=data.train_batch + data.val_batch + data.test_batch)
plot_tsne(features, labels, perplexity=10)
```

## Datasets used in the paper

- **JARVIS-DFT** — general crystal benchmark (25 property tasks): https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699
- **Cu(II) aqua complexes** — structure-sensitive optical response: https://archive.materialscloud.org/record/2022.66
- **OOD Materials Benchmark** — dielectric / elastic / perovskite splits: https://github.com/sadmanomee/OOD_Materials_Benchmark
- **Gold clusters (Au₄–Au₂₀)** — computed with VASP (GGA-PBE, PAW); used for data-efficient extrapolation and CALYPSO-based global structure search.

## Citation

If you use DIGNN, please cite the paper (update once published):

```bibtex
@article{tang2026dignn,
  title  = {Deconstruction-Integration Graph Neural Networks for data-efficient and out-of-distribution prediction of atomistic material properties},
  author = {Tang, Yushan and Chen, Haoqi and Cao, Jiale and Li, Bo and Li, Zhe},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
