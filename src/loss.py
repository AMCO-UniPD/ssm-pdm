"""
Python script with the loss functions for the `chronos-pdm` project
"""

from typing import Tuple

import ipdb
import torch
import torch.nn as nn


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

    def forward(
        self,
        y_pred:torch.Tensor,
        y_true:torch.Tensor,
        mask:torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Root Mean Squared Error (RMSE) loss
        between the predicted and the true values

        Args:
            y_pred (torch.Tensor): The predicted values
            y_true (torch.Tensor): The true values
            mask (torc.Tensor): mask to not consider padded values in the loss computation

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

class WindowMSELoss(nn.Module):
    def __init__(self):
        super(WindowMSELoss, self).__init__()

    def forward(
        self,
        y_pred:torch.Tensor,
        y_true:torch.Tensor,
        mask:torch.Tensor,
        n_const_wins: int,
        n_decreasing_wins: int,
    ) -> torch.Tensor:
        """
        Compute the weighted MSE loss for the windowed approach.
        A different weight is used depending on weather we have
        a constant or non constant window. If we have a constant
        window the weight should be 1/n_const_wins, otherwise it should
        be 1/n_decreasing_wins

        Args:
            y_pred (torch.Tensor): predicted RUL
            y_true (torch.Tensor): true RUL
            mask (torch.Tensor): mask to take into account for padded values
            n_const_wins (int): total number of constant windows in the dataset
            n_decreasing_wins (int): total number of decreasing windows in the dataset

        Returns:
            loss (torch.Tensor): MSE loss value
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()

        is_constant = torch.tensor([
            torch.unique(y_true[i][mask[i]]).numel() == 1
            for i in range(y_true.shape[0])
        ], device=y_true.device)

        weights = torch.where(
            is_constant,
            1.0 / n_const_wins,
            1.0 / n_decreasing_wins
        )

        mse_per_window = torch.tensor([
            torch.mean((y_pred[i][mask[i]] - y_true[i][mask[i]]) ** 2)
            for i in range(y_true.shape[0])
        ], device=y_true.device)

        return torch.sum(weights * mse_per_window)

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

class WindowedQuantileLoss(nn.Module):
    def __init__(self):
        super(WindowedQuantileLoss, self).__init__()

    def forward(
        self,
        y_pred:torch.Tensor,
        y_true:torch.Tensor,
        mask:torch.Tensor,
        n_const_wins: int,
        n_decreasing_wins: int,
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute the Pinball loss between the predicted and the true values

        Args:
            y_pred (torch.Tensor): The predicted values
            y_true (torch.Tensor): The true values
            mask (torch.Tensor): The mask for the padded values
            tau (float): The quantile level
            n_const_wins (int): total number of constant windows in the dataset
            n_decreasing_wins (int): total number of decreasing windows in the dataset

        Returns:
            torch.Tensor: The Pinball loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()

        is_constant = torch.tensor([
            torch.unique(y_true[i][mask[i]]).numel() == 1
            for i in range(y_true.shape[0])
        ], device=y_true.device)

        weights = torch.where(
            is_constant,
            1.0 / n_const_wins,
            1.0 / n_decreasing_wins
        )

        mae_per_window = torch.tensor([
            torch.mean(torch.abs(y_pred[i][mask[i]] - y_true[i][mask[i]]))
            for i in range(y_true.shape[0])
        ], device = y_true.device)

        loss = torch.where(
            mae_per_window>0,
            (1-tau) * mae_per_window, # Overestimation
            tau *  mae_per_window # Underestimation
        )

        return torch.sum(weights*loss)

class WindowedPinballLoss(nn.Module):
    def __init__(self, tau:float):
        super(WindowedPinballLoss, self).__init__()
        self.tau = tau

    def forward(
        self,
        y_pred:torch.Tensor,
        y_true:torch.Tensor,
        mask:torch.Tensor,
        n_const_wins: int,
        n_decreasing_wins: int,
    ) -> torch.Tensor:
        """
        Compute the Pinball loss between the predicted and the true values
        for a specific quantile level passed in input

        Args:
            y_pred (torch.Tensor): The predicted values
            y_true (torch.Tensor): The true values
            mask (torch.Tensor): The mask for the padded values
            n_const_wins (int): total number of constant windows in the dataset
            n_decreasing_wins (int): total number of decreasing windows in the dataset

        Returns:
            torch.Tensor: The Pinball loss
        """

        mask=mask.squeeze(-1).bool() if mask.ndim>2 else mask.bool()

        is_constant = torch.tensor([
            torch.unique(y_true[i][mask[i]]).numel() == 1
            for i in range(y_true.shape[0])
        ], device=y_true.device)

        weights = torch.where(
            is_constant,
            1.0 / n_const_wins,
            1.0 / n_decreasing_wins
        )

        mae_per_window = torch.tensor([
            torch.mean(torch.abs(y_pred[i][mask[i]] - y_true[i][mask[i]]))
            for i in range(y_true.shape[0])
        ], device = y_true.device)

        loss = torch.where(
            mae_per_window>0,
            (1-self.tau) * mae_per_window, # Overestimation
            self.tau *  mae_per_window # Underestimation
        )

        return torch.sum(weights*loss)


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
        criterion=SSMSELoss()
    elif loss_name=="window_mse":
        criterion=WindowMSELoss()
    elif loss_name=="rmse":
        criterion=SSMRMSELoss()
    elif loss_name=="pinball":
        criterion=SSMPinballLoss(tau=tau)
    elif loss_name=="quantile_reg":
        criterion=QuantileLoss()
    elif loss_name=="window_quantile_reg":
        criterion=WindowedQuantileLoss()

    if eval_loss_name=="mae":
        eval_loss=MAELoss()
    elif eval_loss_name=="mse":
        eval_loss=SSMSELoss()
    elif eval_loss_name=="window_mse":
        criterion=WindowMSELoss()
    elif eval_loss_name=="rmse":
        eval_loss=SSMRMSELoss()
    elif eval_loss_name=="pinball":
        eval_loss=SSMPinballLoss(tau=tau)
    elif eval_loss_name=="window_pinball":
        eval_loss=WindowedPinballLoss(tau=tau)

    return criterion, eval_loss
