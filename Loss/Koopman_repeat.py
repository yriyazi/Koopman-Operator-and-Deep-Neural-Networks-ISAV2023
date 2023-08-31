import torch
import Model
import Utils
import torch.nn             as nn
import torch.optim          as optim

device = 'cuda'


###############################################################################################
criterion_Koopman = torch.nn.MSELoss()

def cacl_loss_Koopman(model,
                      Koopman_prediction_horizon,
                      prediction_input_size,
                      data,
                      batch,
                      _divition_factr,
                      optimizer,
                      criterion = criterion_Koopman,
                      Koopman_hidden_size = 800
                      ):
    #PlaceHolder
    Koopman_y       = torch.zeros(size = [Koopman_prediction_horizon,Koopman_hidden_size,Koopman_hidden_size],device=device)
    Koopman_pred    = torch.zeros(size = [Koopman_prediction_horizon,Koopman_hidden_size,Koopman_hidden_size],device=device)

    # Koopman_Evolution    = torch.zeros(size = [Koopman_prediction_horizon+1,Koopman_hidden_size,Koopman_hidden_size],device=device)
    # Koopman_Evolution[0,:,:] = model.Koopman_operator.weight.detach().clone()

    optimizer.zero_grad()
    for i in range(Koopman_prediction_horizon):
        _data = data[:,(batch*prediction_input_size)+i    :((batch+1)*prediction_input_size)+i]+(
                                                    (2*torch.rand(size=[prediction_input_size],device=device)-1)/(_divition_factr*2))
        with torch.inference_mode():
            # give input at time t_0
            input_for_koopman = model.encoder.forward(_data.cuda())[0].detach()
            # give input at time t_n to calculate the input for RNN
            Koopman_y[i,:,:] = model.encoder_koopman(_data.cuda())[0].detach()
        _Koopman_Evolution = torch.matrix_power(model.Koopman_operator.weight, i)

        # _Koopman_Evolution =torch.matmul(Koopman_Evolution[i,:,:].detach().clone(),model.Koopman_operator.weight)# torch.matrix_power(, i)
        # Koopman_Evolution[i+1,:,:] = _Koopman_Evolution.detach().clone()

        Koopman_pred[i,:,:] = torch.matmul(input_for_koopman.detach().clone(),_Koopman_Evolution.t())


    loss = criterion(Koopman_pred,Koopman_y)
    loss.backward()
    optimizer.step()

    return model , loss