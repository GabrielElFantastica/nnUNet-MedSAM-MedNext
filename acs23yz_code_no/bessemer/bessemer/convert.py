import math
import torch
import torch.nn as nn

def convert_model_modalities(model, num_channels=3):

    original_conv_weight = model.image_encoder.patch_embed.proj.weight.data
    original_conv_bias = model.image_encoder.patch_embed.proj.bias.data

    # Create a new convolutional layer with 4 input channels
    new_conv = nn.Conv2d(in_channels=num_channels, out_channels=768, kernel_size=(16, 16), stride=(16, 16))

    # Initialize the new convolutional layer weights
    # Use uniform distribution to initialize the new weights
    nn.init.kaiming_uniform_(new_conv.weight, a=math.sqrt(5))
    new_conv.bias.data = original_conv_bias  # Keep the bias unchanged

    # Copy the original weights to the new weights (for the first 3 channels)
    new_conv.weight.data[:, :3, :, :] = original_conv_weight

    # Initialize the weights of the new fourth channel to zero (or other suitable values)
    new_conv.weight.data[:, 3, :, :] = 0.0

    # Replace the convolutional layer in the model
    model.image_encoder.patch_embed.proj = new_conv

    return model