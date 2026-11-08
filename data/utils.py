import numpy as np
from joblib import Parallel, delayed

from ase import Atoms
from torch_geometric.loader import DataLoader

import torch
import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from .Atoms import AtomsData

import os
ENV_DEVICE = os.environ.get('DIGNN_ENV') or os.environ.get('DEVICE') or 'cpu'


def _check_SimplyConnected(pos, r_cut):
    device = torch.device(ENV_DEVICE)
    
    pos_tensor = torch.from_numpy(pos).float().to(device)
    dm = torch.cdist(pos_tensor, pos_tensor)
    
    threshold = r_cut
    adjacency_matrix = (dm <= threshold)
    adjacency_matrix.fill_diagonal_(False)  # 原地去自环, 避免额外分配 torch.eye(N)
    adjacency_matrix = adjacency_matrix.int()
    
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
    将ase_db中的所有结构转换为AtomsData格式，返回一个list（串行）

    等价于 ase_db2AtomsData_parallel(ase_db, check_rcut=r_cut,
                                     properties=properties, n_jobs=1)。
    """
    return ase_db2AtomsData_parallel(ase_db, check_rcut=r_cut,
                                     properties=properties, n_jobs=1)


def _convert_chunk_worker(atoms_list, keys, check_rcut, properties):
    """多进程 worker：把一小批 ASE Atoms 转成 AtomsData。

    单帧失败只记录到 bad，不让整批中断；异常类型一并保留，
    避免串行写法里 except ValueError 之外的错误被 print 后静默吞掉。
    """
    good, bad = [], []
    for atoms, key in zip(atoms_list, keys):
        try:
            good.append(ase2AtomsData(atoms, check_rcut=check_rcut,
                                      properties=list(properties)))
        except ValueError as exc:          # 预期异常：连通性检查不通过
            bad.append((key, f'ValueError: {exc}'))
        except Exception as exc:           # 其余异常同样暴露出来
            bad.append((key, f'{type(exc).__name__}: {exc}'))
    return good, bad


def _iter_db_atoms(source, subset=None, chunk=128, frame_stride=1,
                   limit_per_system=None, group_key='system',
                   energy_key='energy', force_key='force'):
    """从 ase.db 流式读取并构造 ASE Atoms，按 chunk 产出 (atoms_list, keys)。

    sqlite 连接只在调用方进程内使用，不跨进程传递。
    frame_stride: 同一体系内每 N 帧取 1 帧，用于降采样控制内存。
    limit_per_system: 同一体系最多取多少帧。
    """
    if hasattr(source, 'select'):          # 已经 connect 好的 db 对象
        db = source
    else:
        from ase.db import connect
        db = connect(source)
    rows = db.select(subset=subset) if subset is not None else db.select()

    counts, buf_atoms, buf_keys = {}, [], []
    for mol in rows:
        group = getattr(mol, group_key, '') if group_key else ''
        frame = int(getattr(mol, 'frame', len(buf_keys)))
        if frame_stride > 1 and frame % frame_stride != 0:
            continue
        if limit_per_system is not None and counts.get(group, 0) >= limit_per_system:
            continue

        atoms = mol.toatoms()
        atoms.arrays[energy_key] = np.array(mol.energy)
        if force_key is not None:
            try:                            # 没有力数据的行由 worker 报缺失键
                atoms.arrays[force_key] = np.asarray(mol.forces)
            except Exception:
                pass

        buf_atoms.append(atoms)
        buf_keys.append(f'{group}#{frame}')
        counts[group] = counts.get(group, 0) + 1

        if len(buf_atoms) >= chunk:
            yield buf_atoms, buf_keys
            buf_atoms, buf_keys = [], []
    if buf_atoms:
        yield buf_atoms, buf_keys


def ase_db2AtomsData_parallel(source, check_rcut: float,
                              properties: list = ('energy', 'force'),
                              subset: str = None,
                              n_jobs: int = 1, chunk: int = 128,
                              frame_stride: int = 1,
                              limit_per_system: int = None,
                              group_key: str = 'system',
                              return_failures: bool = False):
    """把 ase.db 并行转换为 AtomsData list。

    与逐帧串行写法的区别：

    1. 每帧的连通性检查（3x3x3 扩胞 cdist）分散到 worker 进程，支持 n_jobs 并行；
    2. 失败结构不再 print(atoms) 后静默丢弃，而是统一收集返回；
    3. frame_stride / limit_per_system 可降采样——力场训练必须用
       return_type='basic'，一个 48 原子的合金结构约占 9.6 MB 索引，
       两万结构需要上百 GB 内存。

    参数
    ----
    source : str | SQLite3Database
        db 路径，或已 connect 的 ase.db 对象。
    check_rcut : float
        连通性检查半径，通常取 pml_rcut。
    properties : list
        需要写入 AtomsData 的属性名，同时也是 atoms.arrays 的键名。
        ASE db 的 energy / forces 分别映射到 arrays['energy'] / arrays['force']。
    subset : str, None
        按 db 的 kvp ``subset`` 过滤，如 'train' / 'val' / 'test'。
    n_jobs : int
        进程数；<=1 时退化为串行（避免小数据集上的进程启动开销）。
    chunk : int
        每个任务包含的结构数，越大通信开销越小。
    frame_stride, limit_per_system, group_key
        降采样控制，见 :func:`_iter_db_atoms`。
    return_failures : bool
        为 True 时返回 ``(data_list, failures)``。

    注意
    ----
    子进程会重新 import DIGNN，因此 DIGNN_ENV 由环境变量决定。若父进程设为
    cuda，每个 worker 都会建立独立 CUDA context，应把 n_jobs 控制在显存允许的
    范围内；这一步的瓶颈是 scipy 的 connected_components 而非 cdist，用 CPU
    并行通常更划算。
    """
    properties = tuple(properties or ())
    force_key = 'force' if 'force' in properties else None

    def _chunk_factory():
        """chunks 是生成器只能消费一次，每个分支各自新建一份。"""
        return _iter_db_atoms(source, subset=subset, chunk=chunk,
                              frame_stride=frame_stride,
                              limit_per_system=limit_per_system,
                              group_key=group_key, force_key=force_key)

    data_list, failures = [], []
    if n_jobs is None or n_jobs <= 1:
        for atoms_list, keys in tqdm.tqdm(_chunk_factory(),
                                         desc='ase2AtomsData(serial)', unit='chunk'):
            good, bad = _convert_chunk_worker(atoms_list, keys, check_rcut, properties)
            data_list.extend(good)
            failures.extend(bad)
    else:
        # 不能把持有 db 游标的懒生成器直接交给 Parallel：joblib 的
        # dispatch_one_batch 会在内部回调线程里 islice() 预取任务，而
        # ASE 的 db.select() 游标绑定在主线程创建的 sqlite 连接上，跨线程
        # 使用会报 "SQLite objects created in a thread can only be used in
        # that same thread"。因此在这里按窗口自己取 chunk，交给 Parallel
        # 的只是 list（list 的迭代器无副作用，线程安全）。
        from itertools import islice
        chunk_iter = _chunk_factory()
        window = max(int(n_jobs) * 4, 16)
        with tqdm.tqdm(desc='ase2AtomsData(parallel)', unit='chunk') as pbar:
            while True:
                batch = list(islice(chunk_iter, window))   # 主线程消费 sqlite 游标
                if not batch:
                    break
                results = Parallel(n_jobs=n_jobs, prefer='processes')(
                    delayed(_convert_chunk_worker)(atoms_list, keys, check_rcut, properties)
                    for atoms_list, keys in batch)
                for good, bad in results:
                    data_list.extend(good)
                    failures.extend(bad)
                pbar.update(len(batch))

    if failures:
        print(f'[ase_db2AtomsData_parallel] {len(failures)} 个结构转换失败，前 5 条：')
        for key, msg in failures[:5]:
            print(f'    {key}: {msg}')

    return (data_list, failures) if return_failures else data_list

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

def _process_single_basic_batch(batch, pml_rcut, pml_mnn, iml_rcut, iml_mnn, store_device,
                                compact_topo=None):
    """单个batch的处理函数

    compact_topo: None 不处理；'dtype' 只压缩索引/位移的 dtype；
        'full' 额外删除 update_geo() 会重算的派生量（仅 return_type='basic' 安全）。
    """
    batch.update_topo(pml_rcut=pml_rcut, pml_mnn=pml_mnn, 
                      iml_rcut=iml_rcut, iml_mnn=iml_mnn)
    batch = batch.to(store_device)
    if compact_topo:
        batch.compact_topo(drop_derived=(compact_topo == 'full'))
    return batch

def update_basic_batch(loader: DataLoader, pml_rcut, pml_mnn, iml_rcut, iml_mnn, 
                       store_device='cpu', num_workers=None, compact_topo=None) -> list[AtomsData]:
    """
    用以批量更新topo（多进程版本）

    num_workers: 建议 1。多进程的结果会在每个 worker 里各留一份不归还 OS，
        实测 16 进程的常驻内存是单进程的 5 倍以上。
    compact_topo: 见 _process_single_basic_batch，默认 None 保持原有行为。
    """
    if num_workers is None:
        num_workers = 1
    
    batches = list(loader)
    
    results = Parallel(n_jobs=num_workers, prefer="processes")(
        delayed(_process_single_basic_batch)(batch, pml_rcut, pml_mnn, iml_rcut, iml_mnn, store_device, compact_topo)
        for batch in tqdm.tqdm(batches, desc="Updating topo", unit="batch")
    )
    
    return results

def update_cplt_graph(basic_graph: AtomsData, store_device='cpu', pos_grad=False, if_strip=True):
    """
    同AtomsData.update_geo()
    添加常用功能。
    store_device: complete graph的转换和保存位置
    """
    basic_graph.to(store_device)
    # update_topo 存的是单图的 n_graphs=int，但 PyG 的 collate 会把 int 属性
    # stack 成 Tensor（值全为 1），而 decoder 的 scatter(dim_size=...) 必须是 int。
    # 这里按 batch 的结构边界重设；单图（无 ptr）保持原值。
    ptr = getattr(basic_graph, 'ptr', None)
    if ptr is not None:
        basic_graph.n_graphs = int(ptr.numel() - 1)
    # compact_topo(drop_derived=True) 会删掉 update_geo() 能重算的派生量
    basic_graph.pos.requires_grad = pos_grad
    basic_graph.update_geo()
    if if_strip:
        basic_graph.strip_topo()
    return basic_graph

def update_cplt_batch(basic_batch: list[AtomsData], store_device: str = 'cpu', num_workers=None):
    if num_workers is None:
        num_workers = 1
        
    results = Parallel(n_jobs=num_workers, prefer="processes")(
        delayed(update_cplt_graph)(b, store_device=store_device, pos_grad=False, if_strip=True)
        for b in tqdm.tqdm(basic_batch, desc='updating geo', unit='batch')
    )
    return results


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