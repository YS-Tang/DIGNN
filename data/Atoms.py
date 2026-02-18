import os
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected, sort_edge_index, degree

from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2 as radius_graph_pbc
from fairchem.core.graph.compute import get_pbc_distances

from ..utils.line_graph import line_graph
from ..utils.sign import Sign, Calc_Sign
from ..utils.utils import DifferentiableClamp, NeboEdge2OpstEdge, radius_graph

ENV_DEVICE = os.environ.get('DIGNN_ENV') or os.environ.get('DEVICE') or 'cpu'
if ENV_DEVICE == 'cpu':
    print("ENV_DEVICE为CPU, 这会导致预处理速度缓慢. 请考虑设置环境变量: DIGNN_ENV=cuda")

class AtomsData(Data):
    """
    AtomsData是DIGNN中原子构型数据结构，用于存储原子构型信息。实例化之后可以通过:
        - update_topo()：更新拓扑结构，用于计算键、角、二面角等特征的index。
        - update_geo()：更新几何特征，包括键长、角度和二面角。
    
    注意：
        1. 模型不支持小于4个原子的分子体系；
        2. 对于周期性系统，一维原子链的情况需谨慎处理，rcut较小时可能有错误，建议使用data.utils.MonoatomicChain_check扩胞至2原子链；
        3. 晶体传入时需要先添加cell和pbc（pbc默认三维），再设置if_pbc=True
        4. 晶体体系和分子体系分开出理，避免混合batch计算
    """
    EPS = 1e-6
    
    def __init__(self, pos: Optional[Tensor] = None, atom: Optional[Tensor] = None, 
                 if_pbc: Union[bool, Tensor] = False, **kwargs):
        super(AtomsData, self).__init__()
        self.pos = pos
        self.atom = atom

        self.if_pbc = if_pbc
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        super(AtomsData, self).__setattr__(key, value)
        if key == 'if_pbc' and value == True:
            self._crystal_check()
    
    # 该方法无用，原因未知
    # def __cat_dim__(self, key: str, value, *args, **kwargs):
    #     if key in ['if_pbc', 'properties', 'pbc]:
    #         return None
    #     return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.atom.device or self.pos.device or None
    
    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        
    def update_topo(self, pml_rcut: float = None, pml_mnn: int = None, 
                    iml_rcut: float = None, iml_mnn: int = None) -> None:
        """
        更新拓扑结构，用于辅助计算键、角、二面角等特征。
        为节省内存，避免高维线图的重复生成，无向图将做有向化处理。
        使用后缀_reo表示re-organization之后的index，其表示单向图，顺序排布，满足row<col
        
        Args:
            pml_rcut: PML层的截断半径
            pml_mnn: PML层的最大邻居数
            iml_rcut: IML层的截断半径
            iml_mnn: IML层的最大邻居数
        """
        # 对单个结构也添加atom_batch, 以便global_decoder统一计算
        if not hasattr(self,'atom_batch'):
            self.atom_batch = torch.zeros_like(self.atom, dtype=torch.long, device=self.device)
        if isinstance(self.properties[0], list):
            self.properties = self.properties[0]
            if hasattr(self, 'pbc'):
                self.pbc = self.pbc[:3]
        """
        ------------------PML层拓扑结构-------------------
        当实际可能的邻居数大于max_num_neighbors时, 生成的bond_index可能是单向的, 这在后续计算中会报错。
        因此在检查结束后再无向化处理
        """
        if torch.all(self.if_pbc):
            (bond_index, cell_offset, _) = radius_graph_pbc(data=self, 
                                                            radius=pml_rcut, 
                                                            max_num_neighbors_threshold=pml_mnn,
                                                            pbc=self.pbc)
            (self.bond_index_reo, self.cell_offset_reo, self.bond_index) = self._crystal_topo(bond_index, cell_offset)
            
            offset_hash = self._offset_hash(self.cell_offset_reo, torch.max(self.bond_index_reo))
            bond_index_reo_bias = self.bond_index_reo.clone()
            bond_index_reo_bias[1] += offset_hash # 下排表示dst, 添加bias之后保证不同晶胞内的对应原子有不同的索引, 以便于bondG的连接计算, 以及符号计算不会混乱
            
            self.angle_index = to_undirected(line_graph(bond_index_reo_bias))
            
        else:
            self.bond_index = to_undirected(radius_graph(self.pos, r=pml_rcut, max_num_neighbors=pml_mnn, batch=self.atom_batch))
            self.bond_index, self.bond_index_reo = _index_reorg(self.bond_index)
        
            self.angle_index = to_undirected(line_graph(self.bond_index))
        
        self.angle_index, self.angle_index_reo = _index_reorg(self.angle_index)
        
        self.dihedral_index = to_undirected(line_graph(self.angle_index))
        # 去除三角形环形子图计算的二面角
        mask = self._rm_TriangleLoopSubgraphDih_mask()
        self.dihedral_index = self.dihedral_index[:, mask]
        self.dihedral_index, self.dihedral_index_reo = _index_reorg(self.dihedral_index)
        
        
        # 计算位置对齐的符号
        if torch.all(self.if_pbc):
            self.bond2angle_AliSign = Sign(bond_index_reo_bias, self.angle_index_reo).AlignmentSign()
        else:
            self.bond2angle_AliSign = Sign(self.bond_index_reo, self.angle_index_reo).AlignmentSign()
        self.angle2dihedral_AliSign = Sign(self.angle_index_reo, self.dihedral_index_reo).AlignmentSign()


        """
        ------------------IML层拓扑结构-------------------
        """
        if torch.all(self.if_pbc):
            (bondI_index, cell_offset_I, _) = radius_graph_pbc(data=self, 
                                                                radius=iml_rcut, 
                                                                max_num_neighbors_threshold=iml_mnn,
                                                                pbc=self.pbc)
            (self.bondI_index_reo, self.cell_offset_I_reo, self.bondI_index) = self._crystal_topo(bondI_index, cell_offset_I)
            
        else:
            self.bondI_index = to_undirected(radius_graph(self.pos, r=iml_rcut, max_num_neighbors=iml_mnn, batch=self.atom_batch))
            self.bondI_index, self.bondI_index_reo = _index_reorg(self.bondI_index)
            
        # 计算映射关系
        self._calc_map()
        
        
    def update_geo(self) -> None:
        """
        更新几何特征，包括键长、角度和二面角
        
        """
        self._calc_bond()
        self._calc_ang()
        self._calc_dihedral()
        
        
    def strip_topo(self) -> None:
        """
        清除辅助拓扑属性，释放内存
        """
        aux_attr = ['bond_index_reo', 'bond2angle_AliSign', 
                    'angle_index_reo', 'angle2dihedral_AliSign',
                    'dihedral_index_reo',
                    'BondVec_reo_uni', 'CrossAng_AliSign', 'CrossVec_PosSignFm',
                    'bondI_index_reo', 'cell_offset_I_reo']
        for attr in aux_attr:
            if hasattr(self, attr):
                delattr(self, attr)

    def to_ase(self):
        """
        单个AtomsData转换为 ASE Atoms 对象
        
        Returns:
            ASE Atoms 对象
        """
        from .utils import AtomsData2ase
        return AtomsData2ase(self)
    
    def to_ase_list(self):
        """
        将DataLoader处理后的batch数据转换为 ASE Atoms 对象列表
        """
        raise NotImplementedError("to_ase_list 方法未实现")
    
    
    def _calc_map(self):
        self.BndReo4Bnd = torch.hstack([torch.arange(self.bond_index_reo.shape[1]), torch.arange(self.bond_index_reo.shape[1])]).to(self.device)
        self.AngReo4Ang = torch.hstack([torch.arange(self.angle_index_reo.shape[1]), torch.arange(self.angle_index_reo.shape[1])]).to(self.device)
        self.DihReo4Dih = torch.hstack([torch.arange(self.dihedral_index_reo.shape[1]), torch.arange(self.dihedral_index_reo.shape[1])]).to(self.device)
        
        self.BndReo4BndI = torch.hstack([torch.arange(self.bondI_index_reo.shape[1]), torch.arange(self.bondI_index_reo.shape[1])]).to(self.device)
    
    def _calc_bond(self) -> None:
        """
        计算键长和键向量
        """
        self.bond_batch = self.atom_batch[self.bond_index_reo[0]]
        if torch.all(self.if_pbc):
            cell_neighbors = degree(self.bond_batch).long()
            out = get_pbc_distances(self.pos, self.bond_index_reo.flip(0), self.cell, self.cell_offset_reo, cell_neighbors, False, True)
            _, self.BondLength_reo, BondVec_reo = out.values()
            self.BondLength_reo = self.BondLength_reo.view(-1, 1)
        else:
            BondVec_reo = self.pos[self.bond_index_reo[1]] - self.pos[self.bond_index_reo[0]]
            self.BondLength_reo = torch.linalg.norm(BondVec_reo, dim=-1, keepdim=True)
        self.BondVec_reo_uni = BondVec_reo / (self.BondLength_reo + self.EPS)
        
        self.bondI_batch = self.atom_batch[self.bondI_index_reo[0]]
        if torch.all(self.if_pbc):
            cell_neighbors_I = degree(self.bondI_batch).long()
            out = get_pbc_distances(self.pos, self.bondI_index_reo, self.cell, self.cell_offset_I_reo, cell_neighbors_I, False, True)
            _, self.BondLengthI_reo, BondIVec_reo = out.values()
            self.BondLengthI_reo = self.BondLengthI_reo.view(-1, 1)
        else:
            BondIVec_reo = self.pos[self.bondI_index_reo[1]] - self.pos[self.bondI_index_reo[0]]
            self.BondLengthI_reo = torch.linalg.norm(BondIVec_reo, dim=-1, keepdim=True)        
        
    def _calc_ang(self) -> None:
        """
        计算角度特征
        """
        CrossSign = Sign(self.bond_index_reo, self.angle_index_reo)
        self.CrossAng_AliSign = CrossSign.AlignmentSign()
        self.CrossVec_PosSignFm = CrossSign.PositionSign('former')

        ang_row, ang_col = self.angle_index_reo
        CosAng_reo = (self.BondVec_reo_uni[ang_row] * self.BondVec_reo_uni[ang_col]).sum(dim=-1, keepdim=True)
        CosAng_reo = DifferentiableClamp.apply(CosAng_reo, -1+self.EPS, 1-self.EPS)
        self.CosAng_reo = torch.acos(CosAng_reo.squeeze(1) * self.bond2angle_AliSign)
        
        self.angle_batch = self.bond_batch[ang_row]

    
    def _calc_dihedral(self) -> None:
        """
        计算二面角特征
        """
        ang_row, ang_col = self.angle_index_reo
        CrossVec_reo = torch.cross(self.BondVec_reo_uni[ang_row], self.BondVec_reo_uni[ang_col], dim=-1)
        CrossVec_uni = CrossVec_reo / (torch.linalg.norm(CrossVec_reo, dim=-1, keepdim=True) + self.EPS)

        dih_row, dih_col = self.dihedral_index_reo
        DihAng_PosSignFm, DihAng_PosSignLt = Sign(self.angle_index_reo, self.dihedral_index_reo).PositionSign('all')

        ang_fm_sign = Calc_Sign(Pre_AliSign = self.CrossAng_AliSign[dih_row],
                                Pre_PosSignFm = self.CrossVec_PosSignFm[dih_row], 
                                Post_PosSign = DihAng_PosSignFm)
        ang_lt_sign = Calc_Sign(Pre_AliSign = self.CrossAng_AliSign[dih_col], 
                                Pre_PosSignFm = self.CrossVec_PosSignFm[dih_col], 
                                Post_PosSign = DihAng_PosSignLt)
        dih_sign = (ang_fm_sign * ang_lt_sign * self.angle2dihedral_AliSign).long()
        
        CosDih_reo = (CrossVec_uni[dih_row] * CrossVec_uni[dih_col]).sum(dim=-1, keepdim=True)
        CosDih_reo = DifferentiableClamp.apply(CosDih_reo, -1+self.EPS, 1-self.EPS)
        
        self.CosDih_reo = torch.acos(CosDih_reo.squeeze(1) * dih_sign)
        
        self.dihedral_batch = self.angle_batch[dih_row]
        
    
    def _crystal_check(self) -> None:
        """
        检查晶体结构是否完整
        
        Raises:
            ValueError: 当cell属性不存在或为None时
        """
        if len(self.atom) == 1 and self.pbc.sum() == 1:
            print('一维单原子链需谨慎处理，建议先使用data.utils.MonoatomicChain_check扩胞至2原子链')
            raise ValueError("单原子链仍有问题，待修改。")
        if not hasattr(self, 'cell') or self.cell is None:
            raise ValueError("晶体结构必须提供cell属性（晶胞参数）")
        
        if not hasattr(self, 'atom_ptr'):
            self.atom_ptr = torch.tensor([0, self.atom.shape[0]])
        
        if not hasattr(self, 'pbc'):
            # 若未提供pbc属性，默认开启三维PBC
            self.pbc = torch.tensor([True, True, True])
            
        self.natoms = self.atom_ptr.diff()
        self.cell = self.cell.view(len(self.natoms), 3, 3)
    
    
        
    def _rm_TriangleLoopSubgraphDih_mask(self) -> Tensor:
        """
        生成去除三角形环形子图计算的二面角的mask
        如果不去除，每个三角形环形子图都会计算出三个数值为0的二面角
        Returns:
            Tensor: 布尔mask，True表示保留该二面角，False表示去除
        """
        # 生成每个角度对边的ids
        AngleOppositeBond_index = NeboEdge2OpstEdge(self.bond_index_reo, self.angle_index_reo)
        AngleOppositeBond_ids = _index_mapping(AngleOppositeBond_index, self.bond_index_reo)
        
        # dih_index_reo中第一排的angle对面的opposite_bond
        dih_fm_AOB = AngleOppositeBond_ids[self.dihedral_index[0, :]]
        # dih_index_reo中第一排的角度的pair_bond
        dih_lt_PB = self.angle_index_reo[:, self.dihedral_index[1, :]]
        
        # 如果dih_fm_AOB存在于dih_lt_PB之中, 说明是三角形环形子图
        mask = (dih_fm_AOB == dih_lt_PB[0, :]) | (dih_fm_AOB == dih_lt_PB[1, :])
        return ~mask
    
    @staticmethod
    def _crystal_topo(index: Tensor, offset: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        处理晶体拓扑结构，确保单向且有序（row <= col）
        
        Args:
            index: 边索引，形状为 [2, num_edges]
            offset: 晶胞偏移，形状为 [num_edges, 3]
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (index_reo, offset_reo, index)
                - index_reo: 重组后的单向边索引
                - offset_reo: 对应的晶胞偏移
                - index: 双向边索引
        """
        index = index.flip(0) # 切换至上排为src, 下排为dst, 以便于后续为dst添加offset_hash_bias之后依然保持row <= cal (注意等号表示允许跨晶胞的self loop)
        index_sorted, offset_sorted = sort_edge_index(index, offset)
        row, col = index_sorted
        mask_reo = row <= col
        
        index_reo, offset_reo = index_sorted[:, mask_reo], offset_sorted[mask_reo]
        index = torch.hstack([index_reo, index_reo.flip(0)]) # 确保双向
        
        return index_reo, offset_reo, index
    
    @staticmethod
    def _offset_hash(offsets: Tensor, base_bias: int) -> Tensor:
        """
        计算晶胞偏移的哈希值，用于区分不同晶胞内的对应原子
        
        Args:
            offsets: 晶胞偏移，形状为 [3] 或 [N, 3]
            base_bias: 哈希基数偏移量
            
        Returns:
            Tensor: 哈希值，用于区分不同晶胞内的原子
        """
        max_offset = torch.max(torch.abs(offsets))
        base = 2 * max_offset + 1
        
        if offsets.dim() == 1:
            if torch.all(offsets == 0):
                magnif = torch.tensor(0, device=self.device)
            else:
                shift = offsets + max_offset
                magnif = (shift[0] * base**2 + shift[1] * base + shift[2] + 1).long()
        else:
            zero_mask = torch.all(offsets == 0, dim=1)
            shift = offsets + max_offset
            magnif = (shift[:, 0] * base**2 + shift[:, 1] * base + shift[:, 2] + 1).long()
            magnif[zero_mask] = 0
        hash = (base_bias + 1) * magnif
        return hash
    
    
def _index_reorg(edge_index: Tensor) -> Tuple[Tensor, Tensor]:
    """
    对边索引进行重组，确保单向且有序（row < col）
    
    Args:
        edge_index: 原始边索引，形状为 [2, num_edges]
        
    Returns:
        Tuple[Tensor, Tensor]: (index, index_reo)，其中index是双向边索引，index_reo是单向边索引
    """
    index_sort = sort_edge_index(edge_index)
    row, col = index_sort
    mask = row < col
    index_reo = index_sort[:, mask]
    index = torch.hstack([index_reo, index_reo.flip(0)])
    return index, index_reo

def _index_mapping(index: Tensor, index_reo: Tensor) -> Tensor:
    """
    为了避免图节点的指数膨胀，键长到角度、角度到二面角都经过了re-organization去重和重排。
    在信息传递时，需要将节点信息恢复至原本两倍量的边信息，即进行reo的逆操作。
    该程序接受reo前后的index，并生成norm前index关于norm后的索引。
    如果index中有index_reo中不存在的索引，则返回-1
    
    Args:
        index: 原始索引，形状为 [2, N]
        index_reo: 重组后的索引，形状为 [2, M]
        
    Returns:
        Tensor: 映射后的索引，未找到的标记为-1
    """
    input_device = index.device
    calc_device = ENV_DEVICE
    
    index = index.to(calc_device)
    index_reo = index_reo.to(calc_device)
    
    index_sort = torch.sort(index, dim=0)[0]
    index_reo_sort = torch.sort(index_reo, dim=0)[0]
    
    # 计算哈希基数（确保无冲突）
    num_nodes = max(index.max(), index_reo.max()).item() + 1
    
    # 哈希编码
    def encode(pairs):
        return pairs[0] * num_nodes + pairs[1]
    
    # 生成去重排序后的哈希数组
    reo_hashes = encode(index_reo_sort).flatten()  # 形状 (M,)
    # 生成原始索引的哈希数组
    original_hashes = encode(index_sort).flatten()  # 形状 (N,)
    
    # 使用二分查找确定每个原始哈希在reo_hashes中的位置
    pos = torch.searchsorted(reo_hashes, original_hashes)
    
    # 检查位置是否有效且哈希值匹配
    valid_pos_mask = pos < reo_hashes.size(0)
    matched = torch.zeros_like(valid_pos_mask, device=calc_device)
    
    if valid_pos_mask.any():
        valid_pos = pos[valid_pos_mask]
        matched_valid = reo_hashes[valid_pos] == original_hashes[valid_pos_mask]
        matched[valid_pos_mask] = matched_valid
    
    result = torch.where(matched, pos, torch.tensor(-1, device=calc_device))
    
    return result.to(input_device)