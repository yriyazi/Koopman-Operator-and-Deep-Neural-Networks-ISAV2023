import torch
import torch.nn as nn

class custum_loss(nn.Module):
    def __init__(self, alpha=0.00001)-> None:
        super(custum_loss, self).__init__()
        self.base   = torch.nn.MSELoss()
        self.alpha  = alpha
    def forward(self, 
                predicted   :torch.Tensor,
                ground_truth:torch.Tensor,
                layer_output   :torch.Tensor,
                ) -> torch.Tensor:

        loss = self.base(predicted,ground_truth)
        _temp =  torch.real(torch.linalg.eigvals(layer_output).mean())*self.alpha
       
        if  _temp =='nan':
            pass
        else:
            loss -= _temp.item()
        # if utils.loss_DSR:
        #     loss += self.DSR(attention)*utils.loss_Beta
        # if utils.loss_AVR:
        #     loss += self.AVR(attention,Region_count)*utils.loss_Gamma
        return loss
    