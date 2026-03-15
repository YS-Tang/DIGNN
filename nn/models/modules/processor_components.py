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
        # raise ValueError("test")
        """IML 层前向传播"""
        if self.iml_node_only:
            for atm_bnd_iml in self.atm_bnd_imls:
                h_atm = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
        else:
            for atm_bnd_iml, aggr, broad in zip(self.atm_bnd_imls, 
                                                self.global_processor.aggr,
                                                self.global_processor.broad):
                h_atm, h_bndI_cplt = atm_bnd_iml(h_atm, edge_index_bndI, h_bndI[index_bondI_map])
                h_bndI = scatter(h_bndI_cplt, index_bondI_map, dim=0, reduce='mean', dim_size=h_bndI.shape[0])
                
                # 不更新global_bond
                self.global_processor.atom_global = aggr(h_atm, 
                                                        self.global_processor.atom_global, 
                                                        self.global_processor.real2virt_edge_index,
                                                        self.global_processor.real2virt_bond,
                                                        )
                h_atm = broad(self.global_processor.atom_global, 
                              h_atm, 
                              h_atm,
                              self.global_processor.virt2real_edge_index,
                              self.global_processor.virt2real_bond,
                              self.global_processor.reg_weights,
                              )
                
        return h_atm
    

from ...utils import MLP
from typing import List
from torch_geometric.utils import to_undirected, add_self_loops
from ...range.blocks import AggregationBlock, BroadcastBlock
from ...range.regularization import LinearReg

class Global_node_nn(nn.Module):
    def __init__(self,
                 atom_dim: int,
                 bond_dim: int, # 可以不跟atomsdata中的bnd_dim相同
                 global_node_num: int=4,
                 proj_hid_dim: List[int]=[64, 64],
                 layer_num: int=2,
                 residual: bool=True,
                 global_heads: int=4):
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.global_node_num = global_node_num
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
        self.aggr = nn.ModuleList([
            AggregationBlock(in_channels=atom_dim, 
                             out_channels=atom_dim, 
                             n_heads=global_heads,
                             basis_dim=bond_dim,
                             ) for _ in range(layer_num)
        ])
        for aggr in self.aggr:
            aggr.reset_parameters()
        
        self.broad = nn.ModuleList([
            BroadcastBlock(in_channels=atom_dim, 
                             out_channels=atom_dim, 
                             n_heads=global_heads,
                             basis_dim=bond_dim,
                             ) for _ in range(layer_num)
        ])
        for broad in self.broad:
            broad.reset_parameters()

        self.linreg = LinearReg(num_virt_nodes=self.global_node_num, 
                                min_num_atoms=1, max_num_atoms=1000)
    
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

        # 计算atom_global和atom_batch_global
        atom_global_mean = scatter(h_atm, self.atom_batch, dim=0, reduce='mean', dim_size=batch_size).unsqueeze(1)
        atom_global = self.proj_mlp_atom(atom_global_mean.transpose(1, 2)).transpose(1, 2)
        self.atom_global = atom_global.reshape(-1, channel_dim) # [Batch * G, Dim]

        self.atom_batch_global = torch.arange(batch_size, device=device).repeat_interleave(num_global_per_batch)

        # 计算edge_index_global和bond_global
        # src为real， dst为virt
        """版本1，将global_node放于real_node最后，global_node_id从real_atom_num开始
        start_global_id = real_atom_num
        global_ids = torch.arange(start_global_id, start_global_id + batch_size * num_global_per_batch, device=device)
        """
        
        # 版本2，将global_node与real_node分离，global_node_id从0开始额外计数
        real_ids = torch.arange(real_atom_num, device=device)
        global_ids = torch.arange(batch_size * num_global_per_batch, device=device)
        
        # 每个batch的real_node数量
        count_real_per_batch = torch.bincount(self.atom_batch, minlength=batch_size)

        repeats_for_dst = count_real_per_batch.repeat_interleave(num_global_per_batch)
        dst = global_ids.repeat_interleave(repeats_for_dst)


        sorted_real_ids = torch.arange(real_atom_num, device=device)[self.atom_batch.argsort()]
        sections = sorted_real_ids.split(count_real_per_batch.tolist())
        src = torch.cat([s.tile((num_global_per_batch,)) for s in sections])

        bond_global = self.proj_mlp_bond(h_atm) # [atom_num, bond_dim]
        bond_global = bond_global[src] # [真实虚拟投影数, bond_dim]
        
        self.real2virt_edge_index = torch.stack([src, dst], dim=0)
        self.real2virt_bond = bond_global
        
        # 错理解为virt做self loop
        # self.real2virt_edge_index, self.real2virt_bond = to_undirected(real2virt_edge_index, bond_global)
        # self.virt2real_edge_index, self.virt2real_bond = add_self_loops(self.real2virt_edge_index.flip(0), 
        #                                                                 self.real2virt_bond,
        #                                                                 fill_value=0)
        
        # broadcast需要做real_node的self loop
        self.virt2real_edge_index = torch.cat([self.real2virt_edge_index.flip(0),
                                               torch.cat([(self.global_node_num+torch.arange(real_atom_num, device=device)).unsqueeze(0),
                                                             torch.arange(real_atom_num, device=device).unsqueeze(0)], dim=0)],
                                              dim=1)
        self.virt2real_bond = torch.cat([self.real2virt_bond, torch.zeros(real_atom_num, self.bond_dim, device=device)], dim=0)
        
        self.reg_weights = self.linreg(torch.tensor([real_atom_num], device=device))