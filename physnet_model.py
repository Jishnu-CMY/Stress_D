# physnet_model.py

"""
PhysNet model definition and loader.

Remote Photoplethysmograph Signal Measurement from Facial Videos Using
Spatio-Temporal Networks (BMVC 2019, Zitong Yu)
Only for research purpose; commercial use not allowed.
"""

import torch
import torch.nn as nn


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=64,
                kernel_size=[4, 1, 1],
                stride=[2, 1, 1],
                padding=[1, 0, 0],
            ),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=64,
                kernel_size=[4, 1, 1],
                stride=[2, 1, 1],
                padding=[1, 0, 0],
            ),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # pool only spatial space, keep 'frames' along temporal dimension
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # x: [B, 3, T, H, W]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)        # [B, 16, T, H, W]
        x = self.MaxpoolSpa(x)        # [B, 16, T, H/2, W/2]

        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x_visual6464)  # [B, 64, T/2, H/4, W/4]

        x = self.ConvBlock4(x)
        x_visual3232 = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x_visual3232)  # [B, 64, T/4, H/8, W/8]

        x = self.ConvBlock6(x)
        x_visual1616 = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x_visual1616)     # [B, 64, T/4, H/16, W/16]

        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)                  # [B, 64, T/2, H/16, W/16]
        x = self.upsample2(x)                 # [B, 64, T,   H/16, W/16]

        x = self.poolspa(x)                   # [B, 64, frames, 1, 1]
        x = self.ConvBlock10(x)               # [B, 1,  frames, 1, 1]

        rPPG = x.view(batch, -1)              # [B, frames]

        return rPPG, x_visual, x_visual3232, x_visual1616


def load_physnet_model(weights_path="./weights/PhysNet_pretrained.pth", frames=128, device="cpu"):
    """
    Instantiate PhysNet and (optionally) load pretrained weights.

    weights_path: path to .pth/.pt checkpoint from the PhysNet repo.
    frames:      temporal length PhysNet is configured for (usually 128).
    """
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=frames)

    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location=device)

        # Depending on the checkpoint format, this may be:
        #   ckpt['state_dict'], ckpt['model'], or ckpt directly.
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip "module." if it exists (for DataParallel checkpoints)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()
    return model
