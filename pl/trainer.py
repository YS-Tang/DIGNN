import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import gc

from ..nn.models.dignn import DIGNN
from ..data.utils import update_cplt_graph
from ..utils.feature_collect import FeatureCollector

class TrainModule(pl.LightningModule):
    def __init__(self,
                model,
                compile_model: bool = False,
                lr: float = 1e-3,
                prop: str = 'prop',
                adamw_weight_decay: float = 1e-2,
                adamw_betas: tuple = (0.9, 0.999),
                onecycle_total_steps: int = None,
                onecycle_final_div_factor: float = 1e+5,
                empty_cache_every_epoch: bool = False,
                test_prefix: str = '',
                enable_embed_decay: bool = True,
                ):
        super().__init__()
        self.model = model
        if compile_model:
            self.model = torch.compile(self.model)
        self.lr = lr
        self.adamw_weight_decay = adamw_weight_decay
        self.adamw_betas = adamw_betas
        self.prop = prop

        self.onecycle_total_steps = onecycle_total_steps
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.empty_cache_every_epoch = empty_cache_every_epoch
        self.test_prefix = test_prefix
        self.enable_embed_decay = enable_embed_decay
        
        self.criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()
        self.save_hyperparameters(ignore=["model"])
        
        self.test_results = {'preds': [], 'targets': []}

    def forward(self, cplt):
        return self.model(cplt)

    def training_step(self, batch, batch_idx):
        prop = self(batch) # cplt
        loss = self.criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # cplt = batch#.to(self.device)
        prop = self(batch)
        loss = self.criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        mae_prop = self.mae_criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae_prop", mae_prop, prog_bar=True, on_epoch=True)
        
        return {"val_loss": loss.detach(), "val_mae_prop": mae_prop.detach()}

    def test_step(self, batch, batch_idx):
        # cplt = batch#.to(self.device)
        prop = self(batch)
        loss = self.criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        mae_prop = self.mae_criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        
        self.log(f"{self.test_prefix}test_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{self.test_prefix}test_mae_prop", mae_prop, prog_bar=True, on_epoch=True)
        
        self.test_results['preds'].append(prop.view(-1, 1).detach().cpu().numpy())
        self.test_results['targets'].append(batch[self.prop].view(-1, 1).detach().cpu().numpy())
        
        return {"test_loss": loss.detach(), "test_mae_prop": mae_prop.detach()}
    
    def on_train_epoch_end(self):
        if self.empty_cache_every_epoch and self.trainer.is_global_zero:
            gc.collect()
            torch.cuda.empty_cache()
    
    def on_test_epoch_end(self):
        self.test_results['preds'] = np.concatenate(self.test_results['preds'], axis=0)
        self.test_results['targets'] = np.concatenate(self.test_results['targets'], axis=0)
    
    def configure_optimizers(self):
        if not self.enable_embed_decay:
            decay, no_decay = set(), set()
            for n, p in self.model.named_parameters():
                if n.startswith('encoder.embed_atm'):
                    no_decay.add(p)
                else:
                    decay.add(p)
            
            param_groups = [
                {"params": list(decay), "weight_decay": self.adamw_weight_decay},
                {"params": list(no_decay), "weight_decay": 0.0},
            ]
        else:
            param_groups = [
                {"params": list(self.parameters()), "weight_decay": self.adamw_weight_decay},
            ]

        optimizer = AdamW(param_groups,
                        lr=self.lr,
                        weight_decay=self.adamw_weight_decay,
                        betas=self.adamw_betas
                        )
        
        scheduler = OneCycleLR(optimizer,
                                max_lr=self.lr,
                                total_steps=self.onecycle_total_steps,
                                final_div_factor=self.onecycle_final_div_factor,
                            )
        
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step",},
                }

    @torch.no_grad()
    def extract_features(self, dataloader, hook_module: nn.Module=None):
        """返回 (features, labels) 两个 numpy 数组"""
        collector = FeatureCollector()
        if hook_module is None: hook_module = self.model.decoder.pooling # 默认提取池化层的特征
        else: hook_module = hook_module
        collector.register_hook(hook_module)

        for batch in dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            _ = self(batch)
            collector.collect_label(batch[self.prop])
            
        collector.remove_hook()

        return collector.get_features(), collector.get_labels()

class TrainModule_FF(pl.LightningModule):
    def __init__(self,
                model: DIGNN,
                compile_model: bool = False,
                compiled_autograd: bool = False,
                lr: float = 1e-3,
                energy_weight: float = 0.1,
                force_weight: float = 1.0,
                adamw_weight_decay: float = 1e-2,
                adamw_betas: tuple = (0.9, 0.999),
                onecycle_total_steps: int = None,
                onecycle_final_div_factor: float = 1e+5,
                empty_cache_every_epoch: bool = False,
                test_prefix: str = '',
                enable_embed_decay: bool = True,
                gradient_clip_val: float = 0.0,
                compile_dynamic: bool = True,
                ):
        super().__init__()
        self.model = model
        # 力场训练中 force = -dE/dpos, loss.backward() 需穿过它 -> double backward。
        # torch.compile(model) 的 AOTAutograd 不支持编译区内 double backward (会报错),
        # 因此 FF 场景不对 model 做 torch.compile, 而是用 compiled_autograd 编译反向图。
        self.use_compiled_autograd = compiled_autograd
        if compile_model and not compiled_autograd:
            # 保留向后兼容: 显式提示用户 FF 场景应用 compiled_autograd 而非 compile_model
            import warnings
            warnings.warn(
                "TrainModule_FF: compile_model=True 在力场训练中会因 double backward 报错, "
                "已自动忽略; 请改用 compiled_autograd=True 以获得编译加速。"
            )
        # compiled_autograd 需要将 forward(含 force 求导)与 backward 包在同一 context 内,
        # 而 Lightning 自动优化会将二者分开, 故切换为手动优化模式。
        if self.use_compiled_autograd:
            self.automatic_optimization = False
        self.lr = lr
        self.adamw_weight_decay = adamw_weight_decay
        self.adamw_betas = adamw_betas
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.onecycle_total_steps = onecycle_total_steps
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.empty_cache_every_epoch = empty_cache_every_epoch
        self.enable_embed_decay = enable_embed_decay
        self.gradient_clip_val = gradient_clip_val
        # DIGNN 每个 batch 的原子/键/角数不同, 属动态形状。compiled_autograd 默认(static)
        # 会为每个新形状重新编译, 导致启动极慢; dynamic=True 让 dynamo 一次编译出通用图,
        # 编译图数降为 1, 大幅减少重编译。代价是单次编译时间略长(仅首步)。
        self.compile_dynamic = compile_dynamic
        
        self.criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()
        self.save_hyperparameters(ignore=["model"])
        
        self.test_results = {'preds_ene': [], 'targets_ene': [],
                             'preds_force': [], 'targets_force': [], 
                             'atom_num': []}

    def forward(self, basic):
        torch.set_grad_enabled(True)
        cplt = update_cplt_graph(basic.clone(),
                                store_device=self.device,
                                pos_grad=True,
                                if_strip=True
                                )
        energy = self.model(cplt)
        force = -torch.autograd.grad(outputs=energy,
                                    inputs=cplt.pos,
                                    grad_outputs=torch.ones_like(energy),
                                    create_graph=True,
                                    retain_graph=True,
                                    )[0]
        return energy, force

    def training_step(self, batch, batch_idx):        
        if self.use_compiled_autograd:
            return self._training_step_compiled_autograd(batch, batch_idx)

        energy, force = self(batch)
        
        loss_ene = self.criterion(energy.view(-1, 1), batch.energy.view(-1, 1))
        loss_force = self.criterion(force.view(-1, 3), batch.force.view(-1, 3)) 
        loss = self.energy_weight * loss_ene + self.force_weight * loss_force
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_ene", loss_ene, prog_bar=False, on_epoch=True)
        self.log("train_loss_force", loss_force, prog_bar=False, on_epoch=True)
        
        return loss

    def _training_step_compiled_autograd(self, batch, batch_idx):
        """手动优化 + compiled_autograd 路径。

        将 forward(含 force 的一阶求导)与 backward 包在同一 compiled_autograd
        context 内, 以编译含二阶链路的完整反向图, 绕开 AOTAutograd 不支持
        编译区 double backward 的限制。实测与 eager 数值一致且提速约 1.5x。
        """
        opt = self.optimizers()
        opt.zero_grad()
        with torch._dynamo.compiled_autograd._enable(torch.compile(dynamic=self.compile_dynamic)):
            energy, force = self(batch)
            loss_ene = self.criterion(energy.view(-1, 1), batch.energy.view(-1, 1))
            loss_force = self.criterion(force.view(-1, 3), batch.force.view(-1, 3))
            loss = self.energy_weight * loss_ene + self.force_weight * loss_force
            self.manual_backward(loss)

        if self.gradient_clip_val and self.gradient_clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val,
                                gradient_clip_algorithm="norm")
        opt.step()

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_ene", loss_ene, prog_bar=False, on_epoch=True)
        self.log("train_loss_force", loss_force, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        energy, force = self(batch)
        atom_num = batch.atom_ptr.diff().view(-1, 1)
        
        mae_ene = self.mae_criterion(energy.view(-1, 1), batch.energy.view(-1, 1))
        mae_ene_peratom = self.mae_criterion(energy.view(-1, 1)/atom_num, batch.energy.view(-1, 1)/atom_num)
        mae_force = self.mae_criterion(force.view(-1, 3), batch.force.view(-1, 3))  
        
        self.log("val_mae_ene", mae_ene, prog_bar=True, on_epoch=True)
        self.log("val_mae_ene_peratom", mae_ene_peratom, prog_bar=True, on_epoch=True)
        self.log("val_mae_force", mae_force, prog_bar=True, on_epoch=True)
        
        return {"val_mae_ene": mae_ene, "val_mae_ene_peratom": mae_ene_peratom, "val_mae_force": mae_force}

    def test_step(self, batch, batch_idx):
        energy, force = self(batch)
        atom_num = batch.atom_ptr.diff().view(-1, 1)
        
        mae_ene = self.mae_criterion(energy.view(-1, 1), batch.energy.view(-1, 1))
        mae_ene_peratom = self.mae_criterion(energy.view(-1, 1)/atom_num, batch.energy.view(-1, 1)/atom_num)
        mae_force = self.mae_criterion(force.view(-1, 3), batch.force.view(-1, 3))  
        
        self.log("test_mae_ene", mae_ene, prog_bar=True, on_epoch=True)
        self.log("test_mae_ene_peratom", mae_ene_peratom, prog_bar=True, on_epoch=True)
        self.log("test_mae_force", mae_force, prog_bar=True, on_epoch=True)
        
        self.test_results['preds_ene'].append(energy.view(-1, 1).detach().cpu().numpy())
        self.test_results['targets_ene'].append(batch.energy.view(-1, 1).detach().cpu().numpy())
        self.test_results['preds_force'].append(force.view(-1, 3).detach().cpu().numpy())
        self.test_results['targets_force'].append(batch.force.view(-1, 3).detach().cpu().numpy())
        self.test_results['atom_num'].append(atom_num.cpu().numpy())
        
        return {"test_mae_ene": mae_ene, "test_mae_ene_peratom": mae_ene_peratom, "test_mae_force": mae_force}
    
    def on_train_epoch_end(self):
        if self.empty_cache_every_epoch and self.trainer.is_global_zero:
            torch.cuda.empty_cache()
    def on_test_epoch_end(self):
        self.test_results['preds_ene'] = np.concatenate(self.test_results['preds_ene'], axis=0)
        self.test_results['targets_ene'] = np.concatenate(self.test_results['targets_ene'], axis=0)
        self.test_results['preds_force'] = np.concatenate(self.test_results['preds_force'], axis=0)
        self.test_results['targets_force'] = np.concatenate(self.test_results['targets_force'], axis=0)
        self.test_results['atom_num'] = np.concatenate(self.test_results['atom_num'], axis=0)
    
    def configure_optimizers(self):
        if not self.enable_embed_decay:
            decay, no_decay = set(), set()
            for n, p in self.model.named_parameters():
                if n.startswith('encoder.embed_atm'):
                    no_decay.add(p)
                else:
                    decay.add(p)
            
            param_groups = [
                {"params": list(decay), "weight_decay": self.adamw_weight_decay},
                {"params": list(no_decay), "weight_decay": 0.0},
            ]
        else:
            param_groups = [
                {"params": list(self.parameters()), "weight_decay": self.adamw_weight_decay},
            ]

        optimizer = AdamW(param_groups,
                        lr=self.lr,
                        weight_decay=self.adamw_weight_decay,
                        betas=self.adamw_betas
                        )
        
        scheduler = OneCycleLR(optimizer,
                                max_lr=self.lr,
                                total_steps=self.onecycle_total_steps,
                                final_div_factor=self.onecycle_final_div_factor,
                            )
        
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step",},
                }
    
    @torch.no_grad()
    def extract_features(self, dataloader, hook_module: nn.Module=None):
        """返回 (features, labels) 两个 numpy 数组"""
        collector = FeatureCollector()
        if hook_module is None: hook_module = self.model.decoder.pooling # 默认提取池化层的特征
        else: hook_module = hook_module
        collector.register_hook(hook_module)

        for batch in dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            _ = self(batch)
            collector.collect_label(batch[self.prop])
            
        collector.remove_hook()

        return collector.get_features(), collector.get_labels()