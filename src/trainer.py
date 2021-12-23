import os
import time
import numpy as np
#from accelerate import accelerator 
# TODO: check accelerator
from typing import Tuple

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

class Trainer:
    def __init__(
        self,
        training_args, 
        model, 
        device, 
        loss_meter, 
        score_meter, 
        optimizers: Tuple[torch.optim.Optimizer,
                    torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        self.training_args = training_args
        self.device = device
        self.model = model.to(self.device)
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        self.optimizer, self.lr_scheduler = optimizers
        self.messages = {
            "epoch" : "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint" : "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
        }
        self.wandb = has_wandb and self.training_args['use_wandb']
        self.total_samples_trained_on = 0
        self.best_val_score = -np.inf
        self.dummy_input = None
    
    def create_optimizer(self):
        lr = self.training_args['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def create_lr_scheduler(self, dataloaders):
        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.training_args['learning_rate'],
            epochs=self.training_args['num_epochs'],
            steps_per_epoch=len(dataloaders['train'])
        )

    def fit(self, epochs, dataloaders, save_path):
        if self.optimizer is None:
            self.create_optimizer()

        if self.lr_scheduler is None:
            self.create_lr_scheduler(dataloaders)

        wandb.watch(self.model, log="all", log_freq=10)
        print("Training for", epochs, "epochs.")
        for epoch in range(1,epochs+1):
            self.info_message("Epoch: {}", epoch)
            train_loss, train_score, train_time = self.run_epoch(epoch, dataloaders, mode="train")
            val_loss, val_score, val_time = self.run_epoch(epoch, dataloaders, mode="val")

            self.info_message(self.messages["epoch"], "Train", epoch,
                              train_loss, train_score, train_time)
            self.info_message(self.messages["epoch"], "Val", epoch,
                              val_loss, val_score, val_time)

            if val_score > self.best_val_score:
                self.info_message(self.messages["checkpoint"], self.best_val_score,
                                  val_score, save_path)
                self.best_val_score = val_score
                self.save_model(epoch, save_path)
        
    def run_epoch(self, epoch, dataloaders, mode):
        is_train = mode == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        
        t = time.time()
        running_loss = self.loss_meter()
        running_score = self.score_meter()
        
        for _, batch in enumerate(dataloaders[mode]):
            with torch.set_grad_enabled(is_train):
                X = batch["X"].to(self.device)
                y_true = batch["y"].to(self.device)
                if is_train:
                    self.optimizer.zero_grad()
                if self.dummy_input == None:
                    self.dummy_input = X
                
                outputs = self.model(X)
                loss = F.cross_entropy(outputs, y_true) #reduce mean
                if is_train:
                    loss.backward()
                running_loss.update(loss.detach().item())
                running_score.update(outputs.detach(), y_true)
                #if self.device=='cuda':
                #    current_score.update(outputs.detach(), y_true)
                #else:
                #    current_score.update(outputs.detach(), y_true)
                if is_train:
                    self.optimizer.step()
                _loss, _score = running_loss.get(), running_score.get()
                if is_train:
                    self.total_samples_trained_on += X.shape[0]
                    wandb.log({
                        f"train_accuracy_digit{i}": _score[i] for i in range(len(_score))},
                        step=self.total_samples_trained_on)
                    wandb.log({
                        "epoch" : epoch, "loss" : _loss},
                        step=self.total_samples_trained_on)
        if not is_train:
            wandb.log({
                f"val_accuracy_digit{i}": _score[i] for i in range(len(running_score.get()))},
                step=self.total_samples_trained_on)
            wandb.log({
                "epoch": epoch}, step=self.total_samples_trained_on)
        return running_loss.get(), running_score.get().mean(), int(time.time() - t)

    def save_model(self, epoch, save_path):
        torch.save(
            {"model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "n_epoch" : epoch},
            save_path
        )
        torch.onnx.export(self.model, self.dummy_input, "model.onnx")
        wandb.save("model.onnx")
    
    def info_message(self, message, *args, end="\n"):
        print(message.format(*args), end=end)