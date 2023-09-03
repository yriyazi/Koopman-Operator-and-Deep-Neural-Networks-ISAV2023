import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
num_epochs          = config['num_epochs']
seed                = config['seed']

#datasets
Noise_division_factor       = config['datasets']['Noise_division_factor']

#loss
alpha                       = config['loss']['alpha']
Eigen                       = config['loss']['Eigen']

# Access model architecture parameters
model_name              = config['model']['name']
prediction_horizon      = config['model']['prediction_horizon']
prediction_input_size   = config['model']['prediction_input_size']
# Encoder
Inception_NumLayers     = config['model']['Inception_NumLayers']
#Koopman
Koopman_prediction_horizon= config['model']['Koopman_prediction_horizon']
# Decoder
RNN_NumLayer            = config['model']['RNN_NumLayer']

# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
learning_rate   = config['scheduler']['learning_rate']
step_size       = config['scheduler']['step_size']
gamma           = config['scheduler']['gamma']