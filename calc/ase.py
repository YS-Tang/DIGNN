from ..data.Atoms import AtomsData
from ..data.utils import ase2AtomsData

import torch
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator as Calculator_base


class Calculator(Calculator_base):
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, mapper, pml_rcut, pml_mnn, iml_rcut, iml_mnn, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()
        self.mapper = mapper
        self.pml_rcut = pml_rcut
        self.pml_mnn = pml_mnn
        self.iml_rcut = iml_rcut
        self.iml_mnn = iml_mnn
        self.device = device
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        # 调用父类方法，确保原子状态更新
        super().calculate(atoms, properties, system_changes)
        
        # 调用外部函数计算能量（单位：eV）和受力（单位：eV/Å）
        energy, forces = self.DIGNN_calculate()
        
        # 存储结果
        self.results = {
            'energy': energy,
            'forces': forces,
        }

    def DIGNN_calculate(self):
        data = ase2AtomsData(self.atoms, check_rcut=self.pml_rcut)
        data.atom = self.mapper(data.atom)
        data.to(self.device)
        data.update_topo(pml_rcut=self.pml_rcut, 
                         pml_mnn=self.pml_mnn,
                         iml_rcut=self.iml_rcut, 
                         iml_mnn=self.iml_mnn)
        data.pos.requires_grad = True
        data.update_geo()
        data.strip_topo()
        
        energy = self.model(data)
        force = -torch.autograd.grad(outputs=energy, 
                                    inputs=data.pos, 
                                    grad_outputs=torch.ones_like(energy),
                                    create_graph=False,
                                    )[0]
        
        energy = energy.detach().cpu().view(-1,1).numpy()
        force = force.detach().cpu().view(-1,3).numpy()
        return energy, force