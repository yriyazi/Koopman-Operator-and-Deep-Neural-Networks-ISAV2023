import torch
import torch.nn as nn
import Utils
class custum_loss(nn.Module):
    def __init__(self, alpha=0.00001)-> None:
        super(custum_loss, self).__init__()
        self.base   = torch.nn.MSELoss()
        self.alpha  = alpha

    def _eigen(layer_output):
        # this term in loss function leads nothing
        return torch.real(torch.linalg.eigvals(layer_output).mean())

    def forward(self, 
                predicted   :torch.Tensor,
                ground_truth:torch.Tensor,
                layer_output   :torch.Tensor,
                ) -> torch.Tensor:

        loss = self.base(predicted,ground_truth)

        if Utils.Eigen:
            _temp =  self._eigen(layer_output)
            if  _temp =='nan':
                pass
            else:
                loss += _temp*self.alpha

        return loss
    