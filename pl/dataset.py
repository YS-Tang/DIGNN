from typing import List
import numpy as np
import tqdm
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from ..data.Atoms import AtomsData
from ..data.utils import update_basic_batch, update_cplt_batch
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
        
    def setup(self, stage=None, input_TrainValTestList_directly: List[List[AtomsData]]=None, no_train:bool=False):
        # ---------- 1. 划分 train / val / test (None 表示该子集不存在) ----------
        # 注意: 两处 train_test_split 的调用顺序与参数必须保持不变, 以确保划分结果的可复现性
        if input_TrainValTestList_directly is None:
            val_test_size = self.test_size + self.val_size
            train_data, val_test_data = train_test_split(self.atomsdata, test_size=val_test_size, random_state=self.random_state)

            if self.val_size > 0.0 and self.test_size > 0.0:
                val_data, test_data = train_test_split(val_test_data, test_size=self.test_size/val_test_size, random_state=self.random_state)
            elif self.val_size > 0.0:   # test_size == 0.0, 不划分测试集
                val_data, test_data = val_test_data, None
            else:                       # val_size == 0.0, 不划分验证集
                val_data, test_data = None, val_test_data
        else:
            train_data, val_data, test_data = input_TrainValTestList_directly
            self.atomsdata = train_data + (val_data or []) + (test_data or [])

        # ---------- 2. 原子序号 -> 嵌入索引映射 ----------
        if self.mapper is not None:
            if self.mapper.known_atomic_numbers == []:
                all_atom_nums = np.unique(np.concatenate([data.atom.numpy() for data in self.atomsdata]))
                self.mapper.set_known_atomic_numbers(all_atom_nums)
            train_data = self._map_process(train_data)
            if val_data:  val_data = self._map_process(val_data)
            if test_data: test_data = self._map_process(test_data)

        # ---------- 3. 预处理 (basic/cplt); no_train 时保留原 train_batch, 空子集置 None ----------
        if not no_train:
            self.train_batch = self._preprocess(train_data, shuffle=True)
        self.val_batch = self._preprocess(val_data, shuffle=False) if val_data else None
        self.test_batch = self._preprocess(test_data, shuffle=False) if test_data else None

    def train_dataloader(self):
        return IterData(self.train_batch)
    def val_dataloader(self):
        return IterData(self.val_batch)
    def test_dataloader(self):
        return IterData(self.test_batch)

    def _preprocess(self, data_list, shuffle: bool):
        """将 AtomsData 列表预处理为 basic 或 cplt 批数据"""
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle, follow_batch=['atom'])
        batch = update_basic_batch(loader, self.pml_rcut, self.pml_mnn, self.iml_rcut, self.iml_mnn, self.store_device, self.num_workers)
        if self.return_type == 'cplt':
            batch = update_cplt_batch(batch, self.store_device, self.num_workers)
        return batch

    def _map_process(self, data_list):
        for d in data_list:
            d.atom = self.mapper(d.atom)
        return data_list