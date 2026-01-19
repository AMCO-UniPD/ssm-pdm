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
            mask: torch.Tensor: The mask for the padded values

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

        loss = torch.where(
            d>0,
            (1-self.tau) * torch.abs(d), # Overestimation
            self.tau *  torch.abs(d) # Underestimation
        )

        return torch.mean(loss)

class SSMPinballLoss(nn.Module):
    def __init__(self, tau:float):
        super(SSMPinballLoss, self).__init__()
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

        loss = torch.where(
            d>0,
            (1-self.tau) * torch.abs(d), # Overestimation
            self.tau *  torch.abs(d) # Underestimation
        )

        return torch.mean(loss)

class QuantileLoss(nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self,
                y_pred:torch.Tensor,
                y_true:torch.Tensor,
                mask:torch.Tensor,
                tau: float = 0.5
    ) -> torch.Tensor:
        """
        Compute the Pinball loss between the predicted and the true values

        Args:
            y_pred (torch.Tensor): The predicted values
            y_true (torch.Tensor): The true values
            mask (torch.Tensor): The mask for the padded values
            tau (float): The quantile level

        Returns:
            torch.Tensor: The Pinball loss
        """

        y_pred=y_pred[mask.bool()]
        y_true=y_true[mask.bool()]
        d = y_pred - y_true

        loss = torch.where(
            d>0,
            (1-tau) * torch.abs(d), # Overestimation
            tau *  torch.abs(d) # Underestimation
        )

        return torch.mean(loss)


def load_loss_functions(
    loss_name:str,
    eval_loss_name:str,
    tau:float=0.5,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load the loss functions for the training and evaluation phases

    Args:
        loss_name (str): The name of the loss function
        eval_loss_name (str): The name of the evaluation loss function
        tau (float): The quantile for the Pinball loss

    Returns:
        criterion (nn.Module): The loss function for the training phase
        eval_loss (nn.Module): The loss function for the evaluation phase
    """

    if loss_name=="mae":
        criterion=SSMAELoss()
    elif loss_name=="mse":
        criterion=MSELoss()
    elif loss_name=="rmse":
        criterion=SSMRMSELoss()
    elif loss_name=="pinball":
        criterion=SSMPinballLoss(tau=tau)
    elif loss_name=="quantile_reg":
        criterion=QuantileLoss()

    if eval_loss_name=="mae":
        eval_loss=MAELoss()
    elif eval_loss_name=="mse":
        eval_loss=MSELoss()
    elif eval_loss_name=="rmse":
        eval_loss=SSMRMSELoss()
    elif eval_loss_name=="pinball":
        eval_loss=SSMPinballLoss(tau=tau)

    return criterion, eval_loss
