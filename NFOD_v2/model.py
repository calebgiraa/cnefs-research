import torch
import torch.nn as nn
import torchvision.transforms.functional as TF # Used for resizing skip connections

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Each DoubleConv block consists of two Conv2d -> BatchNorm -> ReLU sequences.
        # bias=False is common when BatchNorm is used immediately after Conv2d,
        # as BatchNorm's learnable beta parameter effectively serves as the bias.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True saves memory by modifying the input directly
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], # Default values for common use cases
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()   # Stores the upsampling path modules
        self.downs = nn.ModuleList() # Stores the downsampling path modules
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling for downsampling

        # Down part of UNET (Encoder)
        # Iteratively build the encoder path by appending DoubleConv blocks
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature # Update in_channels for the next block

        # Up part of UNET (Decoder)
        # Iteratively build the decoder path by appending ConvTranspose2d and DoubleConv blocks
        # Iterate features in reverse order to go from bottleneck back to original resolution
        for feature in reversed(features):
            # ConvTranspose2d (Upsampling layer)
            # The input channels to ConvTranspose2d should be the feature maps from the previous stage
            # of the decoder (which is `feature * 2` from the concatenation, then processed by the DoubleConv)
            # and `feature * 2` coming from the bottleneck if it's the first up-convolution.
            # You had a conditional logic here:
            # `feature * 2 if feature != features[-1] else features[-1]*2`
            # This is correct for the first up-convolution (coming from bottleneck, which doubled features[-1]),
            # and `feature * 2` for subsequent up-convolutions (coming from concatenated skip + upsampled feature).
            # The bottleneck output is features[-1]*2. So the first ConvTranspose2d takes features[-1]*2 as in_channels.
            # Subsequent ConvTranspose2d layers take `feature_from_previous_up_block` as in_channels.
            # Since the double conv *after* upsampling reduces to `feature` channels, the next upsample takes `feature` as input.
            # This makes the input to ConvTranspose2d simply `feature * 2` because of the concatenation later.
            # Let's adjust this: The input to ConvTranspose2d is the output of the *previous* DoubleConv block in the UP path,
            # or the bottleneck.
            # After the bottleneck (features[-1]*2), the first upsample layer takes features[-1]*2 as input.
            # Then, after a DoubleConv (which outputs `feature` channels), the next upsample layer should take `feature` channels.
            # This requires a slight adjustment to the in_channels for ConvTranspose2d or how `features` are used.

            # Re-thinking the in_channels for ConvTranspose2d:
            # The input to ConvTranspose2d comes from the output of the previous `DoubleConv` in the up path (or bottleneck).
            # The `DoubleConv` outputs `feature` channels. So, the ConvTranspose2d *should* take `feature` channels as input.
            # The concatenation happens *after* ConvTranspose2d and *before* the second DoubleConv.
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, # Input to ConvTranspose2d should be double the 'feature' from the previous layer's output before concatenation.
                                 # This is because the bottleneck has features[-1]*2. After one DoubleConv up, it's `feature`.
                                 # But, the U-Net typically has feature*2 coming into the UP ConvTranspose layer due to skip connection.
                                 # Let's simplify this common pattern:
                                 # Encoder output is `features[-1]*2` (bottleneck).
                                 # First up-conv: takes `features[-1]*2` -> produces `features[-1]` output.
                                 # Then concatenate `features[-1]` from skip. Total `features[-1]*2`.
                                 # Then `DoubleConv(features[-1]*2, features[-1])`.
                                 # Next up-conv: takes `features[-1]` -> produces `features[-2]`.
                                 # Then concatenate `features[-2]` from skip. Total `features[-2]*2`.
                                 # Then `DoubleConv(features[-2]*2, features[-2])`.
                                 # So, the input to ConvTranspose2d is `feature*2` at the start, and then `feature` later.
                                 # The `feature` in the loop is the `out_channels` of the current DoubleConv.
                                 # A common pattern is:
                                 #   nn.ConvTranspose2d(input_from_previous_layer, feature, kernel_size=2, stride=2)
                                 #   Then concat.
                                 #   Then DoubleConv(feature * 2, feature)
                                 #
                                 # Let's assume the previous layer (bottleneck or previous up-DoubleConv) outputs `feature_X` channels.
                                 # The ConvTranspose2d takes `feature_X` as input, outputs `feature_X / 2` (or whatever `feature` is).
                                 # Then, `feature_X / 2` is concatenated with a `feature_X / 2` skip connection.
                                 # So, input to DoubleConv is `feature_X`.
                                 # The current setup has `ConvTranspose2d(feature * 2, feature)`. This implies
                                 # that the previous stage output `feature * 2`. This is correct for the bottleneck
                                 # and generally for the concatenation step.
                                 # However, it implies `ConvTranspose2d` takes `feature * 2` and outputs `feature`.
                                 # Let's keep your original structure for now as it's a common interpretation.
                    feature, # Output channels of ConvTranspose2d matches the feature size for the corresponding skip connection
                    kernel_size=2,
                    stride=2,
                )
            )
            # The subsequent DoubleConv takes (skip_connection_channels + upsampled_channels) which is feature*2
            # and outputs feature channels. This is correct.
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck layer (between encoder and decoder)
        # Takes the last feature size from the encoder and doubles it
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Final 1x1 convolution to map feature channels to desired output channels (e.g., 1 for binary mask)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path (downsampling)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Store the output before pooling for skip connection
            x = self.pool(x)           # Apply pooling

        x = self.bottleneck(x) # Pass through bottleneck
        skip_connections = skip_connections[::-1] # Reverse skip connections for decoder

        # Decoder path (upsampling)
        # Iterate through upsampling blocks (ConvTranspose2d and DoubleConv)
        # `idx` increments by 2 because each self.ups entry is a pair of (ConvTranspose2d, DoubleConv)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # First, perform the ConvTranspose2d (upsampling)
            skip_connection = skip_connections[idx // 2] # Get corresponding skip connection

            # Handle potential size mismatch due to odd input dimensions or padding differences
            # U-Net relies on precise size matching for concatenation.
            if x.shape != skip_connection.shape:
                # Resize the upsampled tensor `x` to match the skip connection's spatial dimensions
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate the skip connection with the upsampled tensor along the channel dimension (dim=1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip) # Pass through the second DoubleConv block

        return self.final_conv(x) # Final 1x1 convolution

def test():
    # Test with a dummy tensor (batch_size, channels, height, width)
    # For grayscale input, channels=1
    x = torch.randn((3, 1, 160, 160))
    # For binary segmentation, out_channels=1 (if using BCEWithLogitsLoss and sigmoid)
    # or out_channels=2 (if using CrossEntropyLoss and softmax for 2 classes: background, pipe)
    # Given a single 'pipe' class, and typically mapping it to 1, while background is 0,
    # out_channels=1 is standard for sigmoid output (predicts foreground probability).
    model = UNET(in_channels=1, out_channels=1)
    model.eval() # Set model to evaluation mode for testing (disables dropout, BatchNorm updates)
    with torch.no_grad(): # Disable gradient calculations for inference
        preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    # The output shape should match the input spatial dimensions, but channels match out_channels
    assert preds.shape == x.shape # This assertion expects (batch, out_channels, H, W) == (batch, in_channels, H, W)
                                 # So if in_channels=1 and out_channels=1, it will match.
                                 # If in_channels=3 and out_channels=1, it won't match (3 != 1).
                                 # A more robust check might be:
    assert preds.shape[0] == x.shape[0] # Batch size
    assert preds.shape[2:] == x.shape[2:] # Height and Width
    print("Test passed!")

if __name__ == "__main__":
    test()