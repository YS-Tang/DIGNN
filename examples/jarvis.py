import os
device = "cuda:0"
os.environ["DIGNN_ENV"] = device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.append(r'../..')

from DIGNN.data import ase2AtomsData, AtomsData
from DIGNN.utils import AtomIndexMapper
from DIGNN.pl import DataModule, TrainModule
from DIGNN.nn import models as dgm

import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from ase.atoms import Atoms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from joblib import Parallel, delayed

HP_atomsdata = {"pml_rcut": 5.0, "pml_mnn": 4, "iml_rcut": 10.0, "iml_mnn": 8} # HP指hyperparams
HP_feat_dim = {'atom_dim': 64, 'bond_dim': 64, 'ang_dim': 32, 'dih_dim': 16}
HP_nn = {'init': 1, 'pml': 1, 'iml': 4, 'decoder': [32,1], 'pooling': 'mean',
         # 长程/全局增强: “数量>0 即开启”约定(无需单独布尔开关)。num_global_tokens=0 即关闭全局模块。
         # 详见 docs/长程全局增强方法说明.md; 晶体能量任务推荐 global + phase(recip,n_max=1)。
         'num_global_tokens': 2,
         # 相位 token(需 num_global_tokens>0 方生效)。晶体用倒格矢(recip)优于各向同性;
         # phase_use_reciprocal=True 时 phase_n_k 被倒格矢数量(由 phase_n_max 决定)覆盖, n_max=1 -> 3个k。
         'phase_n_k': 4,
         'phase_use_reciprocal': True, 'phase_n_max': 1,
         # LES 长程能量项(Decoder 内): 消融证伪(小数据无力监督下发散), 默认 les_n_k=0 关闭。
         'les_n_k': 0,
         'les_use_reciprocal': False, 'les_n_max': 2}
HP_train = {'batch_size': 4, 'max_epochs': 10, 'lr': 1e-3, 'adamw_weight_decay': 1e-3,
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
        except Exception:
            return None
        return data
    
    results = Parallel(n_jobs=num_workers, prefer="processes")(
        delayed(_jarvis_process)(mol, property, r_cut)
        for mol in tqdm(jarvis.iloc, desc="Processing molecules", unit="molecule")
    )
    basic_graphs = [r for r in results if r is not None]
    
    return basic_graphs

property = 'optb88vdw_total_energy'
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
# 正式训练使用完整 JARVIS 数据集; 快速测试可改用同目录的 1000 样本文件:
#   atomsdata = generate_Graphs_from_jarvis('jarvis_1000samples.json',
#                                           property=property, r_cut=HP_atomsdata['pml_rcut'], num_workers=8)
atomsdata = generate_Graphs_from_jarvis(r'jarvis_1000samples.json',
                                         property=property, r_cut=HP_atomsdata['pml_rcut'], num_workers=8)

data = DataModule(atomsdata,
                    **HP_atomsdata,
                    test_size=0.1, val_size=0.1,
                    batch_size=HP_train['batch_size'], num_workers=8, store_device='cpu',
                    mapper=AtomIndexMapper(),
                    return_type='cplt',
                    )
data.setup()


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
                                            num_global_tokens=HP_nn['num_global_tokens'],
                                            n_k=HP_nn['phase_n_k'],
                                            phase_use_reciprocal=HP_nn['phase_use_reciprocal'],
                                            phase_n_max=HP_nn['phase_n_max'],
                                            ), 
                decoder=dgm.Decoder(dim=[HP_feat_dim['atom_dim']] + HP_nn['decoder'],
                                    reduce_method=HP_nn['pooling'],
                                    dropout=0.0,
                                    les_n_k=HP_nn['les_n_k'],
                                    les_use_reciprocal=HP_nn['les_use_reciprocal'],
                                    les_n_max=HP_nn['les_n_max']),
                ).to(device)
                
from pytorch_lightning.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger("tb_logs", name=property)
tb_logger.log_hyperparams({**HP_atomsdata, 
                           **HP_feat_dim, 
                            'n_layer':(HP_nn['pml'], HP_nn['iml']),
                            'decoder':HP_nn['decoder'],  
                            'init_nn_layer':HP_nn['init'],
                            'pooling':HP_nn['pooling'],
                            'num_global_tokens':HP_nn['num_global_tokens'],
                            'phase_n_k':HP_nn['phase_n_k'],
                            'phase_use_reciprocal':HP_nn['phase_use_reciprocal'],
                            'les_n_k':HP_nn['les_n_k'],
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
                           compile_model=False,
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
