import torch
from torch import nn
from torch_geometric.utils import scatter

class HGC(nn.Module):
    def __init__(self,
                 atm_bnd_pmls: nn.ModuleList,
                 bnd_ang_pmls: nn.ModuleList,
                 ang_dih_pmls: nn.ModuleList,
                 pml_node_only: bool = False):
        super().__init__()
        self.atm_bnd_pmls = atm_bnd_pmls
        self.bnd_ang_pmls = bnd_ang_pmls
        self.ang_dih_pmls = ang_dih_pmls
        self.pml_node_only = pml_node_only
        self.pml = len(atm_bnd_pmls)
    
    def forward(self,
                h_atm: torch.Tensor,
                h_bnd: torch.Tensor,
                h_ang: torch.Tensor,
                h_dih: torch.Tensor,
                edge_index_bnd: torch.Tensor,
                edge_index_ang: torch.Tensor,
                edge_index_dih: torch.Tensor,
                index_bond_map,
                index_ang_map,
                index_dih_map) -> torch.Tensor:
        """PML 层前向传播"""
        if self.pml_node_only:
            for ang_dih_pml, bnd_ang_pml, atm_bnd_pml in zip(self.ang_dih_pmls, 
                                                            self.bnd_ang_pmls,
                                                            self.atm_bnd_pmls):
                h_ang = ang_dih_pml(h_ang, edge_index_dih, h_dih[index_dih_map])
                h_bnd = bnd_ang_pml(h_bnd, edge_index_ang, h_ang[index_ang_map])
                h_atm = atm_bnd_pml(h_atm, edge_index_bnd, h_bnd[index_bond_map])
        else:
            for ang_dih_pml, bnd_ang_pml, atm_bnd_pml in zip(self.ang_dih_pmls, 
                                                            self.bnd_ang_pmls,
                                                            self.atm_bnd_pmls):
                h_ang, h_dih_cplt = ang_dih_pml(h_ang, edge_index_dih, h_dih[index_dih_map])
                if self.pml > 1:
                    h_dih = scatter(h_dih_cplt, index_dih_map, dim=0, reduce='mean', dim_size=h_dih.shape[0])
                    
                h_bnd, h_ang_cplt = bnd_ang_pml(h_bnd, edge_index_ang, h_ang[index_ang_map])
                if self.pml > 1:
                    h_ang = scatter(h_ang_cplt, index_ang_map, dim=0, reduce='mean', dim_size=h_ang.shape[0])
                    
                h_atm, h_bnd_cplt = atm_bnd_pml(h_atm, edge_index_bnd, h_bnd[index_bond_map])
                if self.pml > 1:
                    h_bnd = scatter(h_bnd_cplt, index_bond_map, dim=0, reduce='mean', dim_size=h_bnd.shape[0])

        return h_atm


class LCP(nn.Module):
    def __init__(self,
                 atm_bnd_imls: nn.ModuleList,
                 iml_node_only: bool = False):
        super().__init__()
        self.atm_bnd_imls = atm_bnd_imls
        self.iml_node_only = iml_node_only
        self.iml = len(atm_bnd_imls)
        self.global_processor = None
    
    def forward(self,
                h_atm: torch.Tensor,
                h_bndI: torch.Tensor,
                edge_index_bndI: torch.Tensor,
                index_bondI_map) -> torch.Tensor:
        self.global_processor.preprocess(h_atm)
        """IML 层前向传播"""
        if self.iml_node_only:
            for atm_bnd_iml in self.atm_bnd_imls:
                h_atm = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
        else:
            for atm_bnd_iml, global_net in zip(self.atm_bnd_imls, self.global_processor.global_atm_bnd):
                h_atm, h_bndI_cplt = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                h_bndI = scatter(h_bndI_cplt, index_bondI_map, dim=0, reduce='mean', dim_size=h_bndI.shape[0])
                
                h_atm_with_global = torch.cat([h_atm, self.global_processor.atom_global], dim=0)
                h_atm_with_global, self.global_processor.bond_global = \
                    global_net(h_atm_with_global, 
                               self.global_processor.edge_index_global, 
                               self.global_processor.bond_global)
                
                h_atm = h_atm_with_global[:h_atm.size(0)]
                self.global_processor.atom_global = h_atm_with_global[h_atm.size(0):]
                
        return h_atm
    

from ...utils import MLP
from typing import List
from torch_geometric.utils import to_undirected
# from torch_geometric.nn import GATv2Conv
from ...convs.gcn import GatedGCN

class Global_node_nn(nn.Module):
    def __init__(self,
                 atom_dim: int,
                 bond_dim: int, # 可以不跟atomsdata中的bnd_dim相同
                 global_node_num: int=4,
                 proj_hid_dim: List[int]=[64, 64],
                 layer_num: int=2,
                 residual: bool=True):
        super().__init__()
        self.global_node_num = global_node_num
        self.residual = residual
        self.atom_batch = None  # [batch*num_atm]
        
        # 数量投影
        # 输入[batch, atom_dim, 1]，输出[batch, atom_dim, global_node_num], (1,2)经过转置
        # 输入的1是batch内的atom池化后的结果
        self.proj_mlp_atom = nn.Sequential(
            MLP([1]+proj_hid_dim+[global_node_num],  act=nn.SiLU(), batch_norm=False, dropout=0),
            nn.LayerNorm((atom_dim, global_node_num))
        ) 
        
        # 权重投影
        # 输入[total_atom_num, atom_dim]，输出[total_atom_num, bond_dim]
        self.proj_mlp_bond = nn.Sequential(
            MLP([atom_dim]+proj_hid_dim+[bond_dim],  act=nn.SiLU(), batch_norm=False, dropout=0),
            nn.LayerNorm(bond_dim)
        ) 
        
        # 可以考虑传入edge
        # edge初始化由原子近邻边池化投影而来
        self.global_atm_bnd = nn.ModuleList([
            GatedGCN(atom_dim, bond_dim, residual=self.residual) for _ in range(layer_num)
        ])
    
    # 用pml传递的h_atm来初始化atom_global和bond_global
    @torch.compiler.disable    
    def preprocess(self,
                h_atm: torch.Tensor,       # [batch*num_atm, atom_dim]
                ) -> torch.Tensor:
        device = h_atm.device
        batch_size = self.atom_batch.max() + 1
        channel_dim = h_atm.size(1)
        real_atom_num = h_atm.size(0)
        num_global_per_batch = self.global_node_num

        atom_global_mean = scatter(h_atm, self.atom_batch, dim=0, reduce='mean', dim_size=batch_size).unsqueeze(1)
        atom_global = self.proj_mlp_atom(atom_global_mean.transpose(1, 2)).transpose(1, 2)
        atom_global = atom_global.reshape(-1, channel_dim) # [Batch * G, Dim]

        atom_batch_global = torch.arange(batch_size, device=device).repeat_interleave(num_global_per_batch)

        start_global_id = real_atom_num
        global_ids = torch.arange(start_global_id, start_global_id + batch_size * num_global_per_batch, device=device)

        count_real_per_batch = torch.bincount(self.atom_batch, minlength=batch_size)
        # max_real_in_batch = count_real_per_batch.max().item()


        repeats_for_dst = count_real_per_batch.repeat_interleave(num_global_per_batch)
        dst = global_ids.repeat_interleave(repeats_for_dst)

        real_ids = torch.arange(real_atom_num, device=device)

        sorted_real_ids = torch.arange(real_atom_num, device=device)[self.atom_batch.argsort()]
        sections = sorted_real_ids.split(count_real_per_batch.tolist())
        src = torch.cat([s.tile((num_global_per_batch,)) for s in sections])

        bond_global = self.proj_mlp_bond(h_atm) # [atom_num, bond_dim]
        bond_global = bond_global[src] # [真实虚拟投影数, bond_dim]
        
        edge_index_global = torch.stack([src, dst], dim=0)
        
        edge_index_global, bond_global = to_undirected(edge_index_global, bond_global)

        self.atom_global = atom_global
        self.atom_batch_global = atom_batch_global
        self.edge_index_global = edge_index_global
        self.bond_global = bond_global

""" 用bondI来初始化bond_global
    @torch.compiler.disable    
    def preprocess(self,
                h_atm: torch.Tensor,       # [batch*num_atm, atom_dim]
                atom_batch: torch.Tensor,  # [batch*num_atm]
                h_bnd: torch.Tensor,       # [batch*num_bnd, bond_dim]
                bond_index: torch.Tensor,  # [2, batch*num_bnd]
                ) -> torch.Tensor:
        
        device = h_atm.device
        batch_size = atom_batch.max() + 1
        channel_dim = h_atm.size(1)
        real_atom_num = h_atm.size(0)
        num_global_per_batch = self.global_node_num

        atom_global_mean = scatter(h_atm, atom_batch, dim=0, reduce='mean', dim_size=batch_size).unsqueeze(1)
        atom_global = self.proj_mlp_atom(atom_global_mean.transpose(1, 2)).transpose(1, 2)
        atom_global = atom_global.reshape(-1, channel_dim) # [Batch * G, Dim]

        atom_batch_global = torch.arange(batch_size, device=device).repeat_interleave(num_global_per_batch)

        start_global_id = real_atom_num
        global_ids = torch.arange(start_global_id, start_global_id + batch_size * num_global_per_batch, device=device)

        count_real_per_batch = torch.bincount(atom_batch, minlength=batch_size)
        # max_real_in_batch = count_real_per_batch.max().item()


        repeats_for_dst = count_real_per_batch.repeat_interleave(num_global_per_batch)
        dst = global_ids.repeat_interleave(repeats_for_dst)

        real_ids = torch.arange(real_atom_num, device=device)

        sorted_real_ids = torch.arange(real_atom_num, device=device)[atom_batch.argsort()]
        sections = sorted_real_ids.split(count_real_per_batch.tolist())
        src = torch.cat([s.tile((num_global_per_batch,)) for s in sections])

        bond_index = bond_index[:,:bond_index.shape[1]//2] # 先做有向图，与h_bnd一致
        bond_index, h_bnd = to_undirected(bond_index, h_bnd) # 由于真实节点和虚拟节点不对等，因此不用reo的对称形式
        local_bnd_mean = scatter(h_bnd, bond_index[0], dim=0, reduce='mean', dim_size=real_atom_num) # [atom_num, bond_dim]
        bond_global = self.proj_mlp_bond(local_bnd_mean) # [atom_num, bond_dim]
        bond_global = bond_global[src] # [真实虚拟投影数, bond_dim]
        
        edge_index_global = torch.stack([src, dst], dim=0)
        
        edge_index_global, bond_global = to_undirected(edge_index_global, bond_global)

        # atom_with_global = torch.cat([h_atm, atom_global], dim=0)
        # atom_batch_with_global = torch.cat([atom_batch, atom_batch_global], dim=0)
        
        # return atom_global, atom_batch_global, edge_index_global, real_atom_num
        self.atom_global = atom_global
        self.atom_batch_global = atom_batch_global
        self.edge_index_global = edge_index_global
        self.bond_global = bond_global
        # self.real_atom_num = real_atom_num
"""
