"""周期边规范化、距离和映射的独立回归测试。"""
from collections import Counter

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from torch_geometric.utils import scatter

from DIGNN.data.Atoms import AtomsData, radius_graph_pbc


def make_data(atoms):
    data = AtomsData(pos=torch.tensor(atoms.positions, dtype=torch.float32),
                     atom=torch.tensor(atoms.numbers), properties=[])
    data.cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
    data.pbc = torch.tensor(atoms.pbc)
    data.if_pbc = True
    data.atom_batch = torch.zeros(len(atoms), dtype=torch.long)
    return data


def edge_keys(index, offset):
    return [(int(i), int(j), *map(int, s))
            for (i, j), s in zip(index.T.tolist(), offset.tolist())]


def reverse(key):
    return (key[1], key[0], *(-s for s in key[2:]))


def reorganize(keys):
    # 输入采用 FairChem 的 [source, target] 约定。
    index = torch.tensor([[k[1], k[0]] for k in keys], dtype=torch.long).reshape(-1, 2).T
    offset = torch.tensor([k[2:] for k in keys], dtype=torch.float32).reshape(-1, 3)
    return AtomsData._crystal_topo(index, offset)


@pytest.mark.parametrize('keys', [
    [(4, 2, 1, -1, 0)], [(2, 4, -1, 1, 0)],
    [(4, 2, 1, -1, 0), (2, 4, -1, 1, 0)],
])
def test_single_direction_is_normalized_not_discarded(keys):
    reo, shift, full = reorganize(keys)
    assert edge_keys(reo, shift) == [(2, 4, -1, 1, 0)]
    assert torch.equal(full, torch.cat([reo, reo.flip(0)], dim=1))


@pytest.mark.parametrize('keys', [
    [(0, 0, 0, -1, 0)], [(0, 0, 0, 1, 0)],
    [(0, 0, 0, -1, 0), (0, 0, 0, 1, 0)],
])
def test_self_image_has_one_representative(keys):
    reo, shift, full = reorganize(keys)
    assert edge_keys(reo, shift) == [(0, 0, 0, 1, 0)]
    assert full.shape == (2, 2)


def test_multiple_images_are_not_merged():
    keys = [(0, 1, 0, 0, 0), (0, 1, 1, 0, 0),
            (1, 0, -1, 0, 0), (0, 1, 1, 0, 0)]
    reo, shift, _ = reorganize(keys)
    assert set(edge_keys(reo, shift)) == {keys[0], keys[1]}


def test_empty_periodic_edges():
    reo, shift, full = reorganize([])
    assert reo.shape == (2, 0)
    assert shift.shape == (0, 3)
    assert full.shape == (2, 0)


@pytest.mark.parametrize('atoms,cutoff,mnn', [
    (bulk('Cu', 'fcc', a=3.6), 2.7, 1000),
    (bulk('Si', 'diamond', a=5.43), 4.0, 1000),
    (Atoms('Si4', scaled_positions=[[.02,.1,.1],[.97,.12,.1],[.2,.4,.2],[.8,.6,.3]],
           cell=[[3,0,0],[.7,3.4,0],[.3,.4,4]], pbc=True), 1.8, 1000),
    (Atoms('Si6', scaled_positions=np.random.default_rng(19).random((6,3)),
           cell=[[3.3,0,0],[.5,3.6,0],[.2,.3,4]], pbc=True), 2.0, 2),
])
def test_each_edge_distance_and_feature_mapping(atoms, cutoff, mnn):
    data = make_data(atoms)
    raw, offsets, _ = radius_graph_pbc(data, cutoff, mnn, pbc=data.pbc)
    reo, shift, full = data._crystal_topo(raw, offsets)
    raw_keys = edge_keys(raw.flip(0), offsets)
    reo_keys = edge_keys(reo, shift)
    physical = reo_keys + [reverse(k) for k in reo_keys]
    expected_set = set(raw_keys) | {reverse(k) for k in raw_keys}
    assert set(physical) == expected_set
    assert all(count == 1 for count in Counter(physical).values())
    i, j, s, distances = neighbor_list('ijSd', atoms, cutoff)
    oracle = {(int(a), int(b), *map(int, v)): d for a,b,v,d in zip(i,j,s,distances)}
    data.bond_index_reo = data.bondI_index_reo = reo
    data.cell_offset_reo = data.cell_offset_I_reo = shift
    data.bond_index = data.bondI_index = full
    data.angle_index_reo = data.dihedral_index_reo = torch.empty((2,0), dtype=torch.long)
    data._calc_map()
    data._calc_bond()
    expected = torch.tensor([oracle[k] for k in reo_keys], dtype=torch.float32)
    torch.testing.assert_close(data.BondLength_reo.flatten(), expected, atol=2e-5, rtol=0)
    torch.testing.assert_close(data.BondLengthI_reo.flatten(), expected, atol=2e-5, rtol=0)
    for mapping in [data.BndReo4Bnd, data.BndReo4BndI]:
        feature = torch.arange(len(reo_keys), dtype=torch.float32)
        expanded = feature[mapping]
        torch.testing.assert_close(scatter(expanded, mapping, reduce='mean'), feature)
        for n, k in enumerate(physical):
            assert k in {reo_keys[mapping[n]], reverse(reo_keys[mapping[n]])}


def test_neighbor_limit_is_invariant_to_atom_relabelling():
    atoms = Atoms('Si6', scaled_positions=np.random.default_rng(19).random((6,3)),
                  cell=[[3.3,0,0],[.5,3.6,0],[.2,.3,4]], pbc=True)
    physical_sets = []
    for permutation in [np.arange(6), np.arange(6)[::-1]]:
        data = make_data(atoms[permutation])
        raw, offset, _ = radius_graph_pbc(data, 2.0, 2, pbc=data.pbc)
        reo, shift, _ = data._crystal_topo(raw, offset)
        keys = [(int(permutation[k[0]]), int(permutation[k[1]]), *k[2:])
                for k in edge_keys(reo, shift)]
        physical_sets.append({min(k, reverse(k)) for k in keys})
    assert physical_sets[0] == physical_sets[1]


def test_last_batch_graph_without_edges():
    atoms = Atoms('H4', positions=[[.1,.1,.1],[1.9,.1,.1],[.3,.4,.2],[1.7,.4,.2]],
                  cell=[2,4,4], pbc=True)
    data = make_data(atoms)
    raw, offset, _ = radius_graph_pbc(data, .9, 1000, pbc=data.pbc)
    reo, shift, _ = data._crystal_topo(raw, offset)
    data.pos = torch.cat([data.pos, torch.tensor([[0.,0.,0.]])])
    data.atom_batch = torch.tensor([0,0,0,0,1])
    data.cell = torch.cat([data.cell, 20*torch.eye(3)[None]])
    data.bond_index_reo = data.bondI_index_reo = reo
    data.cell_offset_reo = data.cell_offset_I_reo = shift
    data._calc_bond()
    assert data.BondLength_reo.shape == (6,1)
    torch.testing.assert_close(data.BondLengthI_reo, data.BondLength_reo)


def test_complete_geometry_and_second_derivatives():
    atoms = Atoms('Si4', scaled_positions=[[.02,.1,.1],[.97,.12,.1],[.2,.4,.2],[.8,.6,.3]],
                  cell=[[3,0,0],[.7,3.4,0],[.3,.4,4]], pbc=True)
    data = make_data(atoms)
    data.update_topo(1.8, 1000, 1.8, 1000)
    data.pos.requires_grad_(True)
    data.update_geo()
    torch.testing.assert_close(data.BondLengthI_reo, data.BondLength_reo)
    for feature in [data.BondLength_reo, data.BondAngle_reo, data.DihedralAngle_reo]:
        assert torch.isfinite(feature).all()
    loss = data.BondLength_reo.square().sum() + data.BondAngle_reo.square().sum()
    force = torch.autograd.grad(loss, data.pos, create_graph=True)[0]
    second = torch.autograd.grad(force.square().sum(), data.pos)[0]
    assert torch.isfinite(force).all()
    assert torch.isfinite(second).all()


def canonical_path(atoms, shifts):
    def normalized(ids, offsets):
        origin = offsets[1]
        return tuple((int(i), *(int(s)-int(o) for s,o in zip(t, origin)))
                     for i,t in zip(ids, offsets))
    forward = normalized(atoms, shifts)
    backward = normalized(atoms[::-1], shifts[::-1])
    return min(forward, backward)


def independent_paths(atoms, cutoff):
    from itertools import combinations
    i,j,s = neighbor_list('ijS', atoms, cutoff)
    neighbors = [[] for _ in atoms]
    for a,b,t in zip(i,j,s):
        neighbors[a].append((int(b), tuple(map(int,t))))
    angles, dihedrals = set(), set()
    zero = (0,0,0)
    for center, edges in enumerate(neighbors):
        for left,right in combinations(edges,2):
            angles.add(canonical_path([left[0],center,right[0]], [left[1],zero,right[1]]))
        for middle,shift in edges:
            for left in edges:
                if left == (middle,shift):
                    continue
                for right in neighbors[middle]:
                    if right == (center,tuple(-v for v in shift)):
                        continue
                    final = tuple(a+b for a,b in zip(shift,right[1]))
                    ids = [left[0],center,middle,right[0]]
                    offsets = [left[1],zero,shift,final]
                    if len(set(zip(ids,offsets))) == 4:
                        dihedrals.add(canonical_path(ids,offsets))
    return angles,dihedrals


def actual_paths(data, kind):
    ids = getattr(data, 'periodic_'+kind+'_atoms').cpu().tolist()
    shifts = getattr(data, 'periodic_'+kind+'_shifts').cpu().tolist()
    return [canonical_path(a,s) for a,s in zip(ids,shifts)]


@pytest.mark.parametrize('atoms,cutoff', [
    (bulk('Cu','fcc',a=3.6),2.7),
    (bulk('Si','diamond',a=5.43),2.6),
    (Atoms('H4',positions=[[.1,.1,.1],[1.9,.1,.1],[.3,.4,.2],[1.7,.4,.2]],cell=[2,4,4],pbc=True),.9),
    (Atoms('Si4',scaled_positions=[[.02,.1,.1],[.97,.12,.1],[.2,.4,.2],[.8,.6,.3]],
           cell=[[3,0,0],[.7,3.4,0],[.3,.4,4]],pbc=True),1.8),
    (Atoms('C4',scaled_positions=[[.03,.1,.5],[.95,.1,.5],[.3,.4,.5],[.7,.5,.5]],
           cell=[3,3,15],pbc=[True,True,False]),1.8),
])
def test_periodic_high_order_paths_and_geometry(atoms,cutoff):
    data=make_data(atoms)
    data.update_topo(cutoff,1000,cutoff,1000)
    data.update_geo()
    expected=independent_paths(atoms,cutoff)
    for kind,oracle in zip(['angle','dihedral'],expected):
        paths=actual_paths(data,kind)
        assert len(paths)==len(set(paths))
        assert set(paths)==oracle
        feature=data.BondAngle_reo if kind=='angle' else data.DihedralAngle_reo
        ids=getattr(data,'periodic_'+kind+'_atoms').cpu().numpy()
        shifts=getattr(data,'periodic_'+kind+'_shifts').cpu().numpy()
        points=atoms.positions[ids]+shifts@atoms.cell.array
        if kind=='angle':
            a,b=points[:,0]-points[:,1],points[:,2]-points[:,1]
        else:
            first,middle,last=points[:,1]-points[:,0],points[:,2]-points[:,1],points[:,3]-points[:,2]
            a,b=np.cross(first,middle),np.cross(middle,last)
        lengths=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        valid=lengths>1e-8
        reference=np.arccos(np.clip(np.sum(a*b,axis=-1)[valid]/lengths[valid],-1,1))
        # 0/π 附近的 EPS 正则产生可预期偏差，另用同约定独立计算核对。
        np.testing.assert_allclose(feature.detach().numpy()[valid],reference,atol=5e-3,rtol=0)
        if kind=='dihedral':
            first=first/(np.linalg.norm(first,axis=-1,keepdims=True)+data.EPS)
            middle=middle/(np.linalg.norm(middle,axis=-1,keepdims=True)+data.EPS)
            last=last/(np.linalg.norm(last,axis=-1,keepdims=True)+data.EPS)
            a,b=np.cross(first,middle),np.cross(middle,last)
            a=np.where(np.linalg.norm(a,axis=-1,keepdims=True)>data.EPS,a,0.)
            b=np.where(np.linalg.norm(b,axis=-1,keepdims=True)>data.EPS,b,0.)
        a=a/(np.linalg.norm(a,axis=-1,keepdims=True)+data.EPS)
        b=b/(np.linalg.norm(b,axis=-1,keepdims=True)+data.EPS)
        regulated=np.arccos(np.clip(np.sum(a*b,axis=-1),-1+data.EPS,1-data.EPS))
        np.testing.assert_allclose(feature.detach().numpy(),regulated,atol=2e-4,rtol=0)
        index=getattr(data,kind+'_index_reo')
        full=getattr(data,kind+'_index')
        mapping=data.AngReo4Ang if kind=='angle' else data.DihReo4Dih
        assert torch.equal(full,torch.cat([index,index.flip(0)],dim=1))
        if kind=='dihedral':
            angle_paths=actual_paths(data,'angle')
            for n,(first,last) in enumerate(index.T.tolist()):
                first_key=canonical_path(ids[n,:3].tolist(),shifts[n,:3].tolist())
                last_key=canonical_path(ids[n,1:].tolist(),shifts[n,1:].tolist())
                assert angle_paths[first]==first_key
                assert angle_paths[last]==last_key
        values=torch.arange(len(paths),dtype=torch.float32)
        torch.testing.assert_close(scatter(values[mapping],mapping,dim_size=len(paths),reduce='mean'),values)


def test_empty_high_order_and_strip():
    atoms=Atoms('H2',positions=[[0,0,0],[.5,0,0]],cell=[10,10,10],pbc=True)
    data=make_data(atoms)
    data.update_topo(1,1000,1,1000)
    data.update_geo()
    assert data.BondAngle_reo.shape==(0,)
    assert data.DihedralAngle_reo.shape==(0,)
    data.update_geo()
    data.strip_topo()
    assert 'periodic_angle_atoms' not in data
    assert 'periodic_dihedral_shifts' not in data


def test_zero_shift_dihedral_matches_legacy_convention():
    positions=[[0,0,0],[1,0,0],[1.5,.8,0],[2.1,1.3,.7]]
    periodic=make_data(Atoms('C4',positions=positions,cell=[20,20,20],pbc=True))
    molecule=AtomsData(pos=periodic.pos.clone(),atom=periodic.atom.clone(),properties=[])
    for data in [periodic,molecule]:
        data.update_topo(1.1,1000,1.1,1000)
        data.update_geo()
    assert len(periodic.DihedralAngle_reo)==len(molecule.DihedralAngle_reo)==1
    torch.testing.assert_close(periodic.DihedralAngle_reo,molecule.DihedralAngle_reo,atol=1e-5,rtol=0)


def test_high_order_relabelling_and_lattice_gauge():
    atoms=Atoms('Si6',scaled_positions=np.random.default_rng(19).random((6,3)),
                cell=[[3.3,0,0],[.5,3.6,0],[.2,.3,4]],pbc=True)
    original=make_data(atoms)
    original.update_topo(2,2,2,2)
    original.update_geo()
    perm=np.arange(6)[::-1]
    other=make_data(atoms[perm])
    other.update_topo(2,2,2,2)
    other.update_geo()
    for kind in ['angle','dihedral']:
        ids=getattr(other,'periodic_'+kind+'_atoms')
        shifts=getattr(other,'periodic_'+kind+'_shifts')
        mapped=[canonical_path(perm[a],s) for a,s in zip(ids.numpy(),shifts.numpy())]
        assert set(mapped)==set(actual_paths(original,kind))
        left=original.BondAngle_reo if kind=='angle' else original.DihedralAngle_reo
        right=other.BondAngle_reo if kind=='angle' else other.DihedralAngle_reo
        values=dict(zip(actual_paths(original,kind),left.tolist()))
        np.testing.assert_allclose(right.numpy(),[values[k] for k in mapped],atol=2e-4)
    # 固定物理拓扑，单原子换用相邻晶胞坐标，偏移同时补偿。
    translations=torch.tensor([[1,0,0],[0,-1,0],[0,0,0],[0,0,1],[0,0,0],[0,0,0]])
    saved=[original.BondLength_reo.clone(),original.BondAngle_reo.clone(),original.DihedralAngle_reo.clone()]
    original.pos=original.pos+translations.float()@original.cell[0]
    for indices,offsets in [(original.bond_index_reo,original.cell_offset_reo),
                            (original.bondI_index_reo,original.cell_offset_I_reo)]:
        offsets+=translations[indices[0]]-translations[indices[1]]
    for kind in ['angle','dihedral']:
        ids=getattr(original,'periodic_'+kind+'_atoms')
        shifts=getattr(original,'periodic_'+kind+'_shifts')
        shifts+=translations[ids[:,1]][:,None,:]-translations[ids]
    original.update_geo()
    for before,after in zip(saved,[original.BondLength_reo,original.BondAngle_reo,original.DihedralAngle_reo]):
        torch.testing.assert_close(before,after,atol=2e-4,rtol=0)


def test_two_cells_high_order_batch():
    atoms=Atoms('Si4',scaled_positions=[[.02,.1,.1],[.97,.12,.1],[.2,.4,.2],[.8,.6,.3]],
                cell=[[3,0,0],[.7,3.4,0],[.3,.4,4]],pbc=True)
    other=atoms.copy()
    other.set_cell(other.cell.array*1.05,scale_atoms=True)
    parts=[make_data(a) for a in [atoms,other]]
    batch=AtomsData(pos=torch.cat([d.pos for d in parts]),atom=torch.cat([d.atom for d in parts]),properties=[])
    batch.cell=torch.cat([d.cell for d in parts])
    batch.pbc=torch.ones(3,dtype=torch.bool)
    batch.atom_ptr=torch.tensor([0,4,8])
    batch.atom_batch=batch.batch=torch.tensor([0]*4+[1]*4)
    batch.if_pbc=True
    batch.update_topo(1.8,1000,1.8,1000)
    batch.update_geo()
    for data in parts:
        data.update_topo(1.8,1000,1.8,1000)
        data.update_geo()
    for kind,field in [('angle','BondAngle_reo'),('dihedral','DihedralAngle_reo')]:
        ids=getattr(batch,'periodic_'+kind+'_atoms')
        assert torch.all(batch.atom_batch[ids]==batch.atom_batch[ids[:,1]][:,None])
        expected=torch.cat([getattr(d,field) for d in parts])
        torch.testing.assert_close(getattr(batch,field),expected)


@pytest.mark.parametrize('node_only',[False,True])
def test_full_model_multilayer_and_force_gradients(node_only):
    from torch import nn
    from DIGNN.nn.models import DIGNN, Encoder, GCN_Processor, Decoder
    class NodeOnly(nn.Module):
        def __init__(self,layer):
            super().__init__()
            self.layer=layer
        def forward(self,*args):
            return self.layer(*args)[0]
    torch.manual_seed(12)
    data=make_data(Atoms('C4',positions=[[0,0,0],[1,0,0],[1.5,.8,0],[2.1,1.3,.7]],cell=[20,20,20],pbc=True))
    data.update_topo(1.1,1000,2,1000)
    data.pos.requires_grad_(True)
    data.update_geo()
    processor=GCN_Processor(atom_dim=8,bond_dim=8,ang_dim=8,dih_dim=8,bondI_dim=8,pml=2,iml=1)
    if node_only:
        processor.hgc.pml_node_only=True
        for field in ['atm_bnd_pmls','bnd_ang_pmls','ang_dih_pmls']:
            setattr(processor.hgc,field,nn.ModuleList([NodeOnly(layer) for layer in getattr(processor.hgc,field)]))
    model=DIGNN(Encoder(num_species=10,atom_dim=8,bond_dim=8,ang_dim=8,dih_dim=8,bondI_dim=8,pml_rcut=1.1,iml_rcut=2),
                processor,Decoder([8,1],reduce_method='sum'))
    energy=model(data)
    assert torch.isfinite(energy).all()
    force=torch.autograd.grad(energy.sum(),data.pos,create_graph=True)[0]
    loss=force.square().sum()
    loss.backward()
    assert torch.isfinite(data.pos.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
