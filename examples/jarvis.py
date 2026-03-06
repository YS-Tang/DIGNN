import os
device = "cuda"
os.environ["DIGNN_ENV"] = device

import sys
sys.path.append(r'/home/user/tys/DIGNN')

from DIGNN.data import ase2AtomsData, AtomsData
from DIGNN.utils import AtomIndexMapper, plot_comparison
from DIGNN.pl import DataModule, TrainModule_FF, TrainModule
from DIGNN.nn import models as dgm

import time
import torch
import numpy as np
import pandas as pd
from ase.atoms import Atoms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

hyperparams = {"pml_rcut": 5.0, "pml_mnn": 12, "iml_rcut": 10.0, "iml_mnn": 24}
DIGNN_feat_dim = {'atom_dim': 256, 'bond_dim': 256, 'ang_dim': 128, 'dih_dim': 64}

def generate_Graphs_from_jarvis(file_path:str, property:str, r_cut:float) -> list[AtomsData]:
    """
    basic_graphs: list of AtomsData, 只包含元素、坐标、力、能量和拓扑结构
    """
    jarvis = pd.read_json(file_path)
    jarvis[property] = jarvis[property].replace('na', np.nan)
    mask = jarvis[property].notna()
    jarvis = jarvis[mask]

    basic_graphs = []
    start_time = time.time()
    
    for i,mol in enumerate(jarvis.iloc):
        atoms = Atoms(symbols=mol['atoms']['elements'],
                      positions=mol['atoms']['coords'],
                      cell=mol['atoms']['lattice_mat'],
                      pbc=True)
        atoms.arrays[property] = np.array(mol[property])
        try:
            data = ase2AtomsData(atoms, check_rcut=r_cut, properties=[property])
        except:
            continue
        basic_graphs.append(data)

        if i % 1000 == 0:
            print(f'sample:{i},time_cost:{time.time() - start_time}')
            start_time = time.time()
    
    return basic_graphs

property = 'max_efg'
"""
['jid', 'spg_number', 'spg_symbol', 'formula',
        'formation_energy_peratom', 'func', 'optb88vdw_bandgap', 'atoms',
        'slme', 'magmom_oszicar', 'spillage', 'elastic_tensor',
        'effective_masses_300K', 'kpoint_length_unit', 'maxdiff_mesh',
        'maxdiff_bz', 'encut', 'optb88vdw_total_energy', 'epsx', 'epsy', 'epsz',
        'mepsx', 'mepsy', 'mepsz', 'modes', 'magmom_outcar', 'max_efg',
        'avg_elec_mass', 'avg_hole_mass', 'icsd', 'dfpt_piezo_max_eij',
        'dfpt_piezo_max_dij', 'dfpt_piezo_max_dielectric',
        'dfpt_piezo_max_dielectric_electronic',
        'dfpt_piezo_max_dielectric_ionic', 'max_ir_mode', 'min_ir_mode',
        'n-Seebeck', 'p-Seebeck', 'n-powerfact', 'p-powerfact', 'ncond',
        'pcond', 'nkappa', 'pkappa', 'ehull', 'dimensionality', 'efg',
        'xml_data_link', 'typ', 'exfoliation_energy', 'spg', 'crys', 'density',
        'poisson', 'raw_files', 'nat', 'bulk_modulus_kv', 'shear_modulus_gv',
        'mbj_bandgap', 'hse_gap', 'reference', 'search']
"""
atomsdata = generate_Graphs_from_jarvis(r'data/jdft_3d-8-18-2021.json',
                                        property=property, r_cut=hyperparams['pml_rcut'])

data = DataModule(atomsdata,
                    **hyperparams,
                    test_size=0.1, val_size=0.1,
                    batch_size=40, num_workers=-2, store_device='cpu',
                    mapper=AtomIndexMapper(),
                    return_type='cplt',
                    )
data.setup()


model = dgm.DIGNN(encoder=dgm.Encoder(num_species=data.mapper.num_embeddings,
                                            **DIGNN_feat_dim,
                                            pml_rcut=hyperparams["pml_rcut"]+0.2,
                                            bondI_dim=DIGNN_feat_dim['bond_dim'],
                                            iml_rcut=hyperparams["iml_rcut"]+0.2),
                processor=dgm.GCN_Processor(**DIGNN_feat_dim,
                                            pml=4,
                                            iml=6,
                                            residual=True,
                                            dropout=0.0,
                                            bondI_dim=DIGNN_feat_dim['bond_dim'],
                                            init_nn_layer=1,
                                            ), 
                decoder=dgm.Decoder(dim=[DIGNN_feat_dim['atom_dim'],128,1], 
                                    reduce_method='mean', 
                                    dropout=0.0),
                ).to(device)
                
from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger("tb_logs", name='max_efg')
tb_logger.log_hyperparams({**hyperparams, 
                           **DIGNN_feat_dim, 
                            'n_layer':(4,6),
                            'decoder':(128,1),  
                            'init_nn_layer':1,
                            'epochs': 300,
                            'weight_decay': 1e-4,
                            'final_div_factor': 1e+4,
                            'lr': 1e-3,
                            })

max_epoch = 300

checkpoint_callback = ModelCheckpoint(
    monitor='val_mae_prop',
    dirpath=f'{tb_logger.root_dir}/version_{tb_logger.version}/checkpoints/',
    filename='best_{epoch:03d}-{val_mae_prop:.4f}',
    save_top_k=1,
    mode='min',
    save_last=True,
)

train_module = TrainModule(model, 
                           compile_model=True,
                           lr=1e-3,
                           prop=property,
                           adamw_weight_decay=1e-4,
                           adamw_betas=(0.9, 0.999),
                           onecycle_total_steps=max_epoch*len(data.train_dataloader()), 
                           onecycle_final_div_factor=1e+4,
                           empty_cache_every_epoch=False,
                           enable_embed_decay=True,
                           )
trainer = pl.Trainer(max_epochs=max_epoch,
                    accelerator="gpu",
                    check_val_every_n_epoch=1,
                    log_every_n_steps=100,
                    precision='16-mixed',
                    benchmark=True,
                    logger=tb_logger,
                    callbacks=[checkpoint_callback],
                    )

trainer.fit(train_module, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
trainer.test(train_module, dataloaders=data.test_dataloader())

best_train_module = TrainModule.load_from_checkpoint(checkpoint_callback.best_model_path, model=model)
best_train_module.test_prefix = 'best_'
trainer.test(best_train_module, dataloaders=data.test_dataloader())
