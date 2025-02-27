"""
Python script with the loss functions for the `chronos-pdm` project
"""

import torch
import torch.nn as nn
import ipdb
from typing import Tuple

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Root Mean Squared Error (RMSE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()
        masked_preds=[y_pred[i][mask[i]].unsqueeze(0) for i in range(y_pred.shape[0])]
        masked_true=[y_true[i][mask[i]].unsqueeze(0) for i in range(y_true.shape[0])]
        y_pred_unpadded=torch.cat(masked_preds)
        y_true_unpadded=torch.cat(masked_true)
        return torch.sqrt(torch.mean((y_pred_unpadded - y_true_unpadded) ** 2))

class SSMRMSELoss(nn.Module):
    def __init__(self):
        super(SSMRMSELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Root Mean Squared Error (RMSE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        y_pred=y_pred[mask.bool()]
        y_true=y_true[mask.bool()]
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()
        masked_preds=[y_pred[i][mask[i]].unsqueeze(0) for i in range(y_pred.shape[0])]
        masked_true=[y_true[i][mask[i]].unsqueeze(0) for i in range(y_true.shape[0])]
        y_pred_unpadded=torch.cat(masked_preds)
        y_true_unpadded=torch.cat(masked_true)
        return torch.mean((y_pred_unpadded - y_true_unpadded) ** 2)

class SSMSELoss(nn.Module):
    def __init__(self):
        super(SSMSELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        y_pred=y_pred[mask.bool()]
        y_true=y_true[mask.bool()]
        return torch.mean((y_pred - y_true) ** 2)

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Absolute Error (MAE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()
        masked_preds=[y_pred[i][mask[i]].unsqueeze(0) for i in range(y_pred.shape[0])]
        masked_true=[y_true[i][mask[i]].unsqueeze(0) for i in range(y_true.shape[0])]
        y_pred_unpadded=torch.cat(masked_preds)
        y_true_unpadded=torch.cat(masked_true)
        return torch.mean(torch.abs((y_pred_unpadded - y_true_unpadded)))

class SSMAELoss(nn.Module):
    def __init__(self):
        super(SSMAELoss, self).__init__()

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Absolute Error (MAE) loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The RMSE loss
        """

        y_pred=y_pred[mask.bool()]
        y_true=y_true[mask.bool()]
        return torch.mean(torch.abs((y_pred - y_true)))

class PinballLoss(nn.Module):
    def __init__(self, tau:float):
        super(PinballLoss, self).__init__()
        self.mae_loss=MAELoss()
        self.tau=tau

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Pinball loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The Pinball loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()
        masked_preds=[y_pred[i][mask[i]].unsqueeze(0) for i in range(y_pred.shape[0])]
        masked_true=[y_true[i][mask[i]].unsqueeze(0) for i in range(y_true.shape[0])]
        y_pred_unpadded=torch.cat(masked_preds)
        y_true_unpadded=torch.cat(masked_true)
        d = y_pred_unpadded - y_true_unpadded

        if d>0: # Overestimation
            return (1-self.tau) * self.mae_loss(y_pred, y_true, mask)
        else: # Underestimation
            return self.tau *  self.mae_loss(y_pred, y_true, mask)

class SSMPinballLoss(nn.Module):
    def __init__(self, tau:float):
        super(SSMPinballLoss, self).__init__()
        self.mae_loss=MAELoss()
        self.tau=tau

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Pinball loss between the predicted and the true values

        Args:
            yhat (torch.Tensor): The predicted values
            y (torch.Tensor): The true values

        Returns:
            torch.Tensor: The Pinball loss
        """


        y_pred=y_pred[mask.bool()]
        y_true=y_true[mask.bool()]
        d = y_pred - y_true

        if d>0: # Overestimation
            return (1-self.tau) * self.mae_loss(y_pred, y_true, mask)
        else: # Underestimation
            return self.tau *  self.mae_loss(y_pred, y_true, mask)

def load_loss_functions(loss_name:str,
                        model_name:str,
                        eval_loss_name:str,
                        tau:float=0.5) -> Tuple[nn.Module, nn.Module]:
    """
    Load the loss functions for the training and evaluation phases

    Args:
        loss_name (str): The name of the loss function
        model_name (str): The name of the model
        eval_loss_name (str): The name of the evaluation loss function
        tau (float): The quantile for the Pinball loss

    Returns:
        criterion (nn.Module): The loss function for the training phase
        eval_loss (nn.Module): The loss function for the evaluation phase
    """

    if loss_name=="mae":
        if model_name.startswith("chronos"):
            criterion=MAELoss()
        else:
            criterion=SSMAELoss()
    elif loss_name=="mse":
        criterion=MSELoss()
    elif loss_name=="rmse":
        if model_name.startswith("chronos"):
            criterion=RMSELoss()
        else:
            criterion=SSMRMSELoss()
    elif loss_name=="pinball":
        criterion=PinballLoss(tau=tau)

    if eval_loss_name=="mae":
        eval_loss=MAELoss()
    elif eval_loss_name=="mse":
        eval_loss=MSELoss()
    elif eval_loss_name=="rmse":
        eval_loss=RMSELoss()
    elif eval_loss_name=="pinball":
        eval_loss=PinballLoss(tau=tau)

    return criterion, eval_loss
