from typing import List
import numpy as np
import tqdm
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from ..data.Atoms import AtomsData
from ..data.utils import update_basic_batch, update_cplt_graph
from ..utils.utils import AtomIndexMapper


class IterData:
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        for batch in self.data:
            yield batch
    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):
    def __init__(self, atomsdata:List[AtomsData], 
                 pml_rcut=3.0,
                 pml_mnn=16,
                 iml_rcut=5.0,
                 iml_mnn=32,
                 test_size=0.1, 
                 val_size=0.1, 
                 batch_size=32, 
                 num_workers=0,
                 store_device='cpu',
                 random_state=42,
                 mapper:AtomIndexMapper=None,
                 return_type:str='cplt'):
        super().__init__()
        self.atomsdata = atomsdata
        self.pml_rcut = pml_rcut
        self.pml_mnn = pml_mnn
        self.iml_rcut = iml_rcut
        self.iml_mnn = iml_mnn
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.store_device = store_device
        self.random_state = random_state
        self.mapper = mapper
        self.return_type = return_type
        assert self.return_type in ['basic', 'cplt'], "return_type must be 'basic' or 'cplt'"
        
        self.train_batch = []
        self.val_batch = []
        self.test_batch = []
    def setup(self, stage=None):
        val_test_size = self.test_size + self.val_size
        train_data, val_test_data = train_test_split(self.atomsdata, test_size=val_test_size, random_state=self.random_state) 
        
        if self.val_size == 0.0:
            self.val_batch = None
            test_data = val_test_data
        if self.test_size == 0.0:
            self.test_batch = None
            val_data = val_test_data
        if self.val_size > 0.0 and self.test_size > 0.0:
            val_data, test_data = train_test_split(val_test_data, test_size=self.test_size/val_test_size, random_state=self.random_state)
        
        if self.mapper is not None:
            if self.mapper.known_atomic_numbers == []:
                all_atom_nums = np.unique(np.concatenate([data.atom.numpy() for data in self.atomsdata]))
                self.mapper.set_known_atomic_numbers(all_atom_nums)
            train_data = self._map_process(train_data)
            if self.val_batch is not None:
                val_data = self._map_process(val_data)
            if self.test_batch is not None:
                test_data = self._map_process(test_data)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, follow_batch=['atom'])
        if self.val_batch is not None:
            val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, follow_batch=['atom'])
        if self.test_batch is not None:
            test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, follow_batch=['atom'])
        
        train_batch = update_basic_batch(train_loader, self.pml_rcut, self.pml_mnn, self.iml_rcut, self.iml_mnn, self.store_device)
        if self.val_batch is not None:
            val_batch = update_basic_batch(val_loader, self.pml_rcut, self.pml_mnn, self.iml_rcut, self.iml_mnn, self.store_device)
        if self.test_batch is not None:
            test_batch = update_basic_batch(test_loader, self.pml_rcut, self.pml_mnn, self.iml_rcut, self.iml_mnn, self.store_device)

        if self.return_type == 'basic':
            self.train_batch = train_batch
            if self.val_batch is not None:
                self.val_batch = val_batch
            if self.test_batch is not None:
                self.test_batch = test_batch
        
        elif self.return_type == 'cplt':
            self.train_batch = self._get_cplt(train_batch)
            if self.val_batch is not None:
                self.val_batch = self._get_cplt(val_batch)
            if self.test_batch is not None:
                self.test_batch = self._get_cplt(test_batch)
            
    def train_dataloader(self):
        return IterData(self.train_batch)
    def val_dataloader(self):
        return IterData(self.val_batch)
    def test_dataloader(self):
        return IterData(self.test_batch)
    
    def _map_process(self, data_list):
        for d in data_list:
            d.atom = self.mapper(d.atom)
        return data_list
    
    def _get_cplt(self, basic):
        cplt = []
        for b in tqdm.tqdm(basic, desc='cplt graph'):
            c = update_cplt_graph(b,
                                store_device=self.store_device, 
                                pos_grad=False, 
                                if_strip=True)
            cplt.append(c)
        return cplt