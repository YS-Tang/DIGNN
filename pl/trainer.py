import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

from ..nn.models.dignn import DIGNN
from ..data.utils import update_cplt_graph

class TrainModule(pl.LightningModule):
    def __init__(self,
                model,
                lr: float = 1e-3,
                prop: str = 'prop',
                adamw_weight_decay: float = 1e-2,
                adamw_betas: tuple = (0.9, 0.999),
                onecycle_total_steps: int = None,
                onecycle_final_div_factor: float = 1e+5,
                empty_cache_every_epoch: bool = False,
                ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.adamw_weight_decay = adamw_weight_decay
        self.adamw_betas = adamw_betas
        self.prop = prop

        self.onecycle_total_steps = onecycle_total_steps
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.empty_cache_every_epoch = empty_cache_every_epoch
        
        self.criterion = torch.nn.MSELoss()
        self.mae_criterion = torch.nn.L1Loss()
        self.save_hyperparameters(ignore=["model"])
        
        self.test_results = {'preds': [], 'targets': []}

    def forward(self, cplt):
        return self.model(cplt)

    def training_step(self, batch, batch_idx):
        # cplt = batch#.to(self.device)
        prop = self(batch)
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
        
        return {"val_loss": loss, "val_mae_prop": mae_prop}

    def test_step(self, batch, batch_idx):
        # cplt = batch#.to(self.device)
        prop = self(batch)
        loss = self.criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        mae_prop = self.mae_criterion(prop.view(-1, 1), batch[self.prop].view(-1, 1))
        
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_mae_prop", mae_prop, prog_bar=True, on_epoch=True)
        
        self.test_results['preds'].append(prop.view(-1, 1).detach().cpu().numpy())
        self.test_results['targets'].append(batch[self.prop].view(-1, 1).detach().cpu().numpy())
        
        return {"test_loss": loss, "test_mae_prop": mae_prop}
    
    def on_train_epoch_end(self):
        if self.empty_cache_every_epoch and self.trainer.is_global_zero:
            torch.cuda.empty_cache()
    
    def on_test_epoch_end(self):
        self.test_results['preds'] = np.concatenate(self.test_results['preds'], axis=0)
        self.test_results['targets'] = np.concatenate(self.test_results['targets'], axis=0)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
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


class TrainModule_FF(pl.LightningModule):
    def __init__(self,
                model: DIGNN,
                lr: float = 1e-3,
                energy_weight: float = 0.1,
                force_weight: float = 1.0,
                adamw_weight_decay: float = 1e-2,
                adamw_betas: tuple = (0.9, 0.999),
                onecycle_total_steps: int = None,
                onecycle_final_div_factor: float = 1e+5,
                empty_cache_every_epoch: bool = False,
                ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.adamw_weight_decay = adamw_weight_decay
        self.adamw_betas = adamw_betas
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.onecycle_total_steps = onecycle_total_steps
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.empty_cache_every_epoch = empty_cache_every_epoch
        
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
        energy, force = self(batch)
        
        loss_ene = self.criterion(energy.view(-1, 1), batch.energy.view(-1, 1))
        loss_force = self.criterion(force.view(-1, 3), batch.force.view(-1, 3)) 
        loss = self.energy_weight * loss_ene + self.force_weight * loss_force
        
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
        optimizer = AdamW(self.model.parameters(),
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