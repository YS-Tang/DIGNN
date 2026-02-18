import ast
import json
import time

import numpy as np
import pandas as pd

from ase.data import atomic_numbers
from ase import Atoms
from torch_geometric.loader import DataLoader

import torch
import tqdm
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from torch_geometric.data import Dataset, InMemoryDataset

from .Atoms import AtomsData

import os
ENV_DEVICE = os.environ.get('DIGNN_ENV') or os.environ.get('DEVICE') or 'cpu'


def _check_SimplyConnected(pos, r_cut):
    device = torch.device(ENV_DEVICE)
    
    pos_tensor = torch.from_numpy(pos).float().to(device)
    dm = torch.cdist(pos_tensor, pos_tensor)
    
    threshold = r_cut
    adjacency_matrix = (dm <= threshold).int() - torch.eye(len(pos), dtype=torch.int, device=device)
    
    adjacency_matrix_np = adjacency_matrix.cpu().numpy()
    sparse_adjacency_matrix = csr_matrix(adjacency_matrix_np)
    n_components, labels = connected_components(sparse_adjacency_matrix, directed=False)
    size_components = np.bincount(labels)
    
    if np.max(size_components) >= 4: return True # 要存在4原子的连通子图
    else: return False

def ase2AtomsData(ase_atoms, check_rcut: float, properties: list[str]=None, if_MonoatomicChain_check: bool=True):
    """
    property的值存储在atoms.arrays中，且为numpy数组
    check_rcut: 检查体系是否在check_rcut下满足至少4原子连通
    properties: 存储在atoms.arrays中的属性
    if_MonoatomChain_check: 单原子链检查
    """
    if if_MonoatomicChain_check: 
        ase_atoms = MonoatomicChain_check(ase_atoms)
    atom_num = torch.from_numpy(ase_atoms.get_atomic_numbers()).long()
    
    pbc = ase_atoms.get_pbc()
    if any(pbc):
        periodic_expand = np.array([1,1,1]) + pbc * 2 # 在周期方向扩展至3倍, 单原子链情况需谨慎处理。
        atoms_expand = ase_atoms * periodic_expand
        if not _check_SimplyConnected(atoms_expand.positions, check_rcut):
            print(ase_atoms)
            raise ValueError(f"该晶体在当前check_rcut={check_rcut}下不满足至少4原子连通，请检查check_rcut取值。")
    else: 
        if not _check_SimplyConnected(ase_atoms.positions, check_rcut):
            print(ase_atoms)
            raise ValueError(f"该分子在当前check_rcut={check_rcut}下不满足至少4原子连通，请检查分子数和check_rcut取值，如果是晶体注意开启pbc。")
    
    data = AtomsData(pos=torch.from_numpy(ase_atoms.positions).float(),atom=atom_num)
    if any(pbc):
        # 先添加cell，添加if_pbc时触发晶体格式检查
        data.cell = torch.from_numpy(ase_atoms.cell.array).float()
        data.pbc = torch.from_numpy(pbc)
        data.if_pbc = True
    
    if properties is None: properties = []
    data.properties = properties
    for prop in properties:
        data[prop] = torch.from_numpy(np.array(ase_atoms.arrays[prop])).float()
    
    return data

def ase_db2AtomsData_list(ase_db, properties: list[str], r_cut: float):
    """
    将ase_db中的所有结构转换为AtomsData格式，返回一个list
    """
    data_list = []
    for mol in tqdm.tqdm(ase_db.select()):
        try:
            atoms = mol.toatoms()
            for prop in properties:
                atoms.arrays[prop] = mol.data[prop]
            data_list.append(ase2AtomsData(atoms, properties, r_cut))
        except ValueError as e:
            print(e)
            continue
    return data_list

def AtomsData2ase(data: AtomsData, properties: list[str]=None) -> Atoms:
    """
    将AtomsData转换为ASE Atoms
    不支持经过DataLoader处理后的batch数据
    """
    data.to('cpu')
    atom_nums = data.atom.numpy()
    pos = data.pos.detach().numpy()
    
    if data.if_pbc:
        cell = data.cell.numpy()
        pbc = data.pbc.numpy()
        atoms = Atoms(atom_nums, pos, cell=cell, pbc=pbc)
    else:
        atoms = Atoms(atom_nums, pos)

    properties = properties or getattr(data, 'properties', None) or []
    if properties is not None:
        for prop in properties:
            atoms.arrays[prop] = data[prop].numpy()
    
    return atoms

def AtomsData2ase_list(data_batch: AtomsData, properties: list[str]=None) -> list[Atoms]:
    raise NotImplementedError("AtomsData2ase_list 未实现")
    
def update_basic_batch(loader: DataLoader, pml_rcut, pml_mnn, iml_rcut, iml_mnn, store_device='cpu', logger=None) -> list[AtomsData]:
    """
    用以批量更新topo
    """

    basic_batch = []
    
    for batch in tqdm.tqdm(loader, desc="Updating topo", unit="batch"):
        batch.update_topo(pml_rcut=pml_rcut, pml_mnn=pml_mnn, 
                          iml_rcut=iml_rcut, iml_mnn=iml_mnn)
        basic_batch.append(batch.to(store_device))
        
    return basic_batch

def update_cplt_graph(basic_graph: AtomsData, store_device='cpu', pos_grad=False, if_strip=True):
    """
    同AtomsData.update_geo()
    添加常用功能。
    store_device: complete graph的转换和保存位置
    """
    basic_graph.to(store_device)
    basic_graph.pos.requires_grad = pos_grad
    basic_graph.update_geo()
    if if_strip:
        basic_graph.strip_topo()
    return basic_graph


def MonoatomicChain_check(data: Atoms):
    """
    将单原子链中的单原子晶胞扩至双原子
    """
    assert isinstance(data, Atoms)
    if len(data.get_atomic_numbers()) == 1 and data.get_pbc().sum() == 1:
        data = data.repeat(data.get_pbc() + np.array([1,1,1])) # 扩至2原子
        raise ValueError("单原子链仍有问题，待修改。")
    return data
            
def batch_iterator(batch_list):
    for batch in batch_list:
        yield batch