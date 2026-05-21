import os
device = "cuda:0"
os.environ["DIGNN_ENV"] = device
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.append(r'/home/user/tys/DIGNN')

from DIGNN.data import ase2AtomsData, AtomsData
from DIGNN.utils import AtomIndexMapper
from DIGNN.pl import DataModule, TrainModule
from DIGNN.nn import models as dgm

import time
import torch
import numpy as np
import pandas as pd
from ase.atoms import Atoms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from joblib import Parallel, delayed

HP_atomsdata = {"pml_rcut": 5.0, "pml_mnn": 12, "iml_rcut": 10.0, "iml_mnn": 24} # HP指hyperparams
HP_feat_dim = {'atom_dim': 512, 'bond_dim': 512, 'ang_dim': 256, 'dih_dim': 128}
HP_nn = {'init': 1, 'pml': 4, 'iml': 8, 'decoder': [64,1], 'pooling': 'mean'}
HP_train = {'batch_size': 32, 'max_epochs': 300, 'lr': 1e-2, 'adamw_weight_decay': 1e-2,
            'adamw_betas': (0.9, 0.999), '1cycle_final_div_factor': 1e+5, 'gradient_clip_val': 1.0}

def generate_Graphs_from_jarvis(file_path:str, property:str, r_cut:float, num_workers:int=1) -> list[AtomsData]:
    """
    basic_graphs: list of AtomsData, 只包含元素、坐标、力、能量和拓扑结构
    """
    jarvis = pd.read_json(file_path)
    jarvis[property] = jarvis[property].replace('na', np.nan)
    mask = jarvis[property].notna()
    jarvis = jarvis[mask]
    
    def _jarvis_process(mol, property:str, r_cut:float):
        atoms = Atoms(symbols=mol['atoms']['elements'],
                      positions=mol['atoms']['coords'],
                      cell=mol['atoms']['lattice_mat'],
                      pbc=True)
        atoms.arrays[property] = np.array(mol[property])
        try:
            data = ase2AtomsData(atoms, check_rcut=r_cut, properties=[property])
        except:
            continue
        return data
    
    basic_graphs = Parallel(n_jobs=num_workers, prefer="Preprocess")(
        delayed(_jarvis_process)(mol, property, r_cut)
        for mol in tqdm(jarvis.iloc, desc="Processing molecules", unit="molecule")
    )
    
    return basic_graphs

property = 'mbj_bandgap'
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
atomsdata, special_atomsdata = generate_Graphs_from_jarvis(r'jarvis/data/jdft_3d-8-18-2021.json',
                                         property=property, r_cut=HP_atomsdata['pml_rcut'], num_workers=32)

data = DataModule(atomsdata,
                    **HP_atomsdata,
                    test_size=0.1, val_size=0.1,
                    batch_size=HP_train['batch_size'], num_workers=32, store_device='cpu',
                    mapper=AtomIndexMapper(),
                    return_type='cplt',
                    )
data.setup()

# special_data = DataModule(special_atomsdata,
#                     **{"pml_rcut": 22.0, "pml_mnn": 12, "iml_rcut": 22.0, "iml_mnn": 24},
#                     test_size=0, val_size=0.1,
#                     batch_size=40, num_workers=-2, store_device='cpu',
#                     mapper=data.mapper,
#                     return_type='cplt',
#                     )
# special_data.setup()

# data.train_batch += special_data.train_batch


model = dgm.DIGNN(encoder=dgm.Encoder(num_species=data.mapper.num_embeddings,
                                            **HP_feat_dim,
                                            pml_rcut=HP_atomsdata["pml_rcut"]+0.2,
                                            bondI_dim=HP_feat_dim['bond_dim'],
                                            iml_rcut=HP_atomsdata["iml_rcut"]+0.2),
                processor=dgm.GCN_Processor(**HP_feat_dim,
                                            pml=HP_nn['pml'],
                                            iml=HP_nn['iml'],
                                            residual=True,
                                            dropout=0.0,
                                            bondI_dim=HP_feat_dim['bond_dim'],
                                            init_nn_layer=HP_nn['init'],
                                            ), 
                decoder=dgm.Decoder(dim=[HP_feat_dim['atom_dim']] + HP_nn['decoder'],
                                    reduce_method=HP_nn['pooling'],
                                    dropout=0.0),
                ).to(device)
                
from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger("tb_logs", name=property)
tb_logger.log_hyperparams({**HP_atomsdata, 
                           **HP_feat_dim, 
                            'n_layer':(HP_nn['pml'], HP_nn['iml']),
                            'decoder':HP_nn['decoder'],  
                            'init_nn_layer':HP_nn['init'],
                            'pooling':HP_nn['pooling'],
                            **HP_train,
                            })

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
                           lr=HP_train['lr'],
                           prop=property,
                           adamw_weight_decay=HP_train['adamw_weight_decay'],
                           adamw_betas=HP_train['adamw_betas'],
                           onecycle_total_steps=HP_train['max_epochs']*len(data.train_dataloader()), 
                           onecycle_final_div_factor=HP_train['1cycle_final_div_factor'],
                           empty_cache_every_epoch=False,
                           enable_embed_decay=True,
                           )
trainer = pl.Trainer(max_epochs=HP_train['max_epochs'],
                    accelerator="gpu",
                    devices=[int(device.split(":")[-1])], # 单卡训练
                    check_val_every_n_epoch=1,
                    log_every_n_steps=100,
                    precision='16-mixed',
                    gradient_clip_val=HP_train['gradient_clip_val'],
                    benchmark=True,
                    logger=tb_logger,
                    callbacks=[checkpoint_callback],
                    )

trainer.fit(train_module, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())

trainer.test(train_module, dataloaders=data.test_dataloader())

best_train_module = TrainModule.load_from_checkpoint(checkpoint_callback.best_model_path, model=model)
best_train_module.test_prefix = 'best_'
trainer.test(best_train_module, dataloaders=data.test_dataloader())
