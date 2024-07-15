import torch
import tqdm
import Model
import Utils
import Loss
import os
import Deeplearning
import torch.nn             as nn
import matplotlib.pyplot    as plt
import numpy                as np
import torch.optim          as optim

Utils.set_seed(42)
device = 'cuda'

npz_file_path = os.path.join(os.getcwd(),"Duffing_Solution","datasets","gamma=0.37 t_span=(0, 50000) initial_conditions=[1.5, 1.5] step_frequency=010.npy")
loaded_data = Utils.read_npz_file(npz_file_path)
#%%
prediction_horizion     = 50
prediction_input_size   = 200
epochs                  = 1
_divition_factr         = 7
_alpha                  = 40


directory_path = os.path.join(os.getcwd(),"Saved","Gamma =0.37 step_frequency=010 prediction_input_size=200 cu_loss=False init=(1.5, -1.5) noise_factor=2.pt")
#%%
data_tensor = torch.from_numpy(loaded_data).to(device=device).to(torch.float)
inception = Model.Encoder_Decoder().to(device)

inception.load_state_dict(torch.load(os.path.join(directory_path)))
inception.eval()
#%%
with torch.inference_mode():
    Batch = 2
    pred_hor = 6000
    x = data_tensor[Batch*prediction_input_size:(Batch+1)*prediction_input_size]#+(torch.rand(size=[prediction_input_size],device=device)/_divition_factr)
    y = data_tensor[(Batch+1)*prediction_input_size:(Batch+1)*prediction_input_size+pred_hor]

    prediction_list = torch.zeros(size=[pred_hor]).to(device)

    decoder_hidden, decoder_cell = torch.zeros(size=[2,prediction_input_size],device=device), torch.zeros(size=[2,prediction_input_size],device=device)
    for i in range(pred_hor):
        # prediction = inception.forward(x)
        prediction,(decoder_hidden, decoder_cell) = inception.forward(x.unsqueeze(0),decoder_hidden, decoder_cell)#
        x =  torch.cat([x[1:],prediction],dim=0)
        prediction_list[i] = prediction
#%%
signal = torch.concat([x[-50:],prediction_list[:]]).detach().cpu().numpy()
prediction = torch.concat([x[-50:],y[:]+torch.rand(size=[pred_hor],device=device)/_divition_factr]  ).detach().cpu().numpy()

Utils.plot_signal_and_prediction(signal, prediction,time = (0,prediction.shape[0]//10),
                            title="Acceleration vs. Prediction", signal_label="Acceleration", prediction_label="Acceleration Predicted")
#%% Example usage
vector = torch.concat([x[-50:],prediction_list[:]])[:].detach().cpu().numpy()
Utils.integrate_and_plot(vector,f"alpha = 0000 gamma=0.37 orgi",0.1,(0,vector.shape[0]//10))
