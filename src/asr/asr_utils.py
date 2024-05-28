import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight

from transformers import Seq2SeqTrainer

""" Trainer Class """
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, balancing, **kwargs):
        super().__init__(**kwargs)
        self.balancing = balancing
    
    def compute_loss(self, model, inputs, return_outputs=False):

        weights = inputs.get("weights")
        if weights is not None:
            inputs.pop("weights")    
        
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if weights is not None:
            if not self.balancing:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels), 
                    labels.view(-1)
                    )
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss_1 = loss_fct(
                    logits.view(-1, self.model.config.num_labels), 
                    labels.view(-1)
                    )
                loss_weighted = nn.CrossEntropyLoss(reduction="none")
                loss_2 = loss_weighted(
                    logits.view(-1, self.model.config.num_labels), 
                    labels.view(-1)
                    )
                loss_2 = loss_2 * weights
                loss_2 = loss_2.mean()
                loss = loss_1 + loss_2
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), 
                labels.view(-1)
                )
        return (loss, outputs) if return_outputs else loss