import torch
import torch.nn as nn
from .encoder import *
from .decoder import *

class Encoder_Decoder(nn.Module):
    """
    A class representing an Encoder-Decoder architecture for prediction tasks.

    This class combines an encoder, decoder, Koopman operator, and reverse map
    to perform prediction tasks on input data sequences.

    Args:
        prediction_input_size (int): The size of the input data for prediction.

    Attributes:
        prediction_input_size (int): The size of the input data for prediction.
        hidden_dim (int): Hidden dimensionality for internal computations.
        encoder (nn.Module): An instance of the InceptionBlock encoder.
        decoder (nn.Module): An instance of the Decoder with hidden_dim and prediction_input_size.
        Koopman_operator (nn.Linear): Linear transformation representing the Koopman operator.
        reverse_map (nn.Sequential): Sequential layers for reverse mapping predictions.

    Methods:
        forward(x, decoder_hidden, decoder_cell):
            Forward pass through the Encoder-Decoder model.
    """

    def __init__(self, prediction_input_size: int)->None:
        """
        Initialize the Encoder_Decoder instance.

        Args:
            prediction_input_size (int): The size of the input data for prediction.
        """
        super(Encoder_Decoder, self).__init__()
        self.prediction_input_size = prediction_input_size
        self.hidden_dim = prediction_input_size * 4

        # Initialize encoder and decoder
        self.encoder = InceptionBlock()
        self.decoder = Decoder(self.hidden_dim, self.prediction_input_size)

        # Initialize Koopman operator
        self.Koopman_operator = nn.Linear(self.hidden_dim, self.hidden_dim,bias =False)

        # Initialize reverse map for prediction transformation
        self.reverse_map = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, 1),
        )

    def encoder_koopman(self,
                        x                   :torch.Tensor,):
        # Pass input through encoder
        out, mean, std = self.encoder(x)

        # Apply Koopman operator
        out = self.Koopman_operator(out)
        return out, mean, std

    def forward(self,
                x                   :torch.Tensor,
                decoder_hidden      :torch.Tensor,
                decoder_cell        :torch.Tensor)->tuple:
        """
        Perform a forward pass through the Encoder_Decoder model.

        Args:
            x (torch.Tensor): Input data tensor.
            decoder_hidden (torch.Tensor): Hidden state of the decoder.
            decoder_cell (torch.Tensor): Cell state of the decoder.

        Returns:
            out (torch.Tensor): Output prediction tensor.
            (decoder_hidden, decoder_cell): Updated decoder hidden and cell states.
        """
        out, mean, std = self.encoder_koopman(x)

        # Pass through decoder
        out, (decoder_hidden, decoder_cell) = self.decoder(out, decoder_hidden, decoder_cell)

        # Concatenate prediction with mean and std, and apply reverse mapping
        out = torch.cat((out[0], mean.view(1), std.view(1)))
        out = self.reverse_map(out).view(1)

        return out, (decoder_hidden, decoder_cell)

class Encoder_Decoder_MLP(nn.Module):
    """
    A class representing an Encoder-Decoder architecture with Multi-Layer Perceptrons (MLPs) for prediction tasks.

    This class combines an encoder, decoder with MLP layers, Koopman operator,
    and reverse map to perform prediction tasks on input data sequences.

    Args:
        prediction_input_size (int): The size of the input data for prediction.

    Attributes:
        prediction_input_size (int): The size of the input data for prediction.
        hidden_dim (int): Hidden dimensionality for internal computations.
        encoder (nn.Module): An instance of the InceptionBlock encoder.
        decoder (nn.Sequential): Sequential layers for the decoder with MLP architecture.
        Koopman_operator (nn.Linear): Linear transformation representing the Koopman operator.
        reverse_map (nn.Sequential): Sequential layers for reverse mapping predictions.

    Methods:
        forward(x):
            Forward pass through the Encoder_Decoder_MLP model.
    """

    def __init__(self, prediction_input_size: int)->None:
        """
        Initialize the Encoder_Decoder_MLP instance.

        Args:
            prediction_input_size (int): The size of the input data for prediction.
        """
        super(Encoder_Decoder_MLP, self).__init__()
        self.prediction_input_size = prediction_input_size
        self.hidden_dim = prediction_input_size * 4

        # Initialize encoder and decoder with MLP architecture
        self.encoder = InceptionBlock()
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 100),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        )

        # Initialize Koopman operator
        self.Koopman_operator = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Initialize reverse map for prediction transformation
        self.reverse_map = nn.Sequential(nn.Linear(3, 1))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Perform a forward pass through the Encoder_Decoder_MLP model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            out (torch.Tensor): Output prediction tensor.
        """
        # Pass input through encoder
        out, mean, std = self.encoder(x)

        # Apply Koopman operator
        out = self.Koopman_operator(out)

        # Pass through decoder
        out = self.decoder(out)

        # Rescale the prediction and add mean to it
        out = (out * std) + mean

        # Apply reverse mapping
        out = self.reverse_map(torch.cat([out, mean, std], dim=1))

        return out
