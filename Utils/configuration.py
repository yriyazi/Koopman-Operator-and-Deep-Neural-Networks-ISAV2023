import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
learning_rate       = config['learning_rate']
batch_size          = config['batch_size']
num_epochs          = config['num_epochs']
seed                = config['seed']
task                = config['task']
# Access dataset parameters
dataset_path        = config['dataset']['path']
pre_pro             = config['dataset']['pre_pro']  
num_classes         = config['dataset']['num_classes']
img_height          = config['dataset']['img_height']
img_width           = config['dataset']['img_width']
img_channels        = config['dataset']['img_channels']
train_split         = config['dataset']['train_split']

# Access model architecture parameters
model_name          = config['model']['name']
pretrained          = config['model']['pretrained']
num_classes         = config['model']['num_classes']

mu                  = config['model']['mu']
sigma               = config['model']['sigma']
bias                = config['model']['bias']
Xavier              = config['model']['Xavier']
L1                  = config['model']['L1']
L2                  = config['model']['L2']
model_Architecture  = config['model']['model']




# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name = config['scheduler']['name']
step_size = config['scheduler']['step_size']
gamma = config['scheduler']['gamma']


# print("configuration hass been loaded!!! \n successfully")
# print(learning_rate)