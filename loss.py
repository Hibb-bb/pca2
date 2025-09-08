import torch
import torch.nn as nn

class MeanRelativeSquaredError(nn.Module):
  
    def __init__(self, epsilon=1e-8):
        super(MeanRelativeSquaredError, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        # Compute the relative error element-wise
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)

        # Compute the squared relative error
        squared_relative_error = relative_error ** 2


        # Compute the mean over all elements
        loss = torch.mean(squared_relative_error)

        return loss
