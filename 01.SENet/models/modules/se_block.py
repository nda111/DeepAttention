import torch
from torch import nn


class SEBlock(nn.Module):
    """
    INPUT: (B, C, H, W) or (C, H, W)
    OUTPUT: (B, C, H, W) or (C, H, W)
    """
    def __init__(self, num_channels: int, hidden_rate: float=6.25E-2, use_bias: bool=False):
        super(SEBlock, self).__init__()

        self.num_channels = num_channels
        self.hidden_rate = hidden_rate
        self.use_bias = use_bias
        self.hidden_channels = int(num_channels * hidden_rate)

        self.layer1 = nn.Linear(num_channels, self.hidden_channels, bias=use_bias)
        self.layer2 = nn.Linear(self.hidden_channels, num_channels, bias=use_bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_later = True
        elif x.ndim == 4:
            squeeze_later = False
        else:
            raise RuntimeError(f'Expected dimension: [batch_size, channels, height, width] or [channels, height, width] but got {x.shape}')

        squeezed = torch.mean(x, dim=(2, 3))
        excited = torch.sigmoid(self.layer2(torch.relu(self.layer1(squeezed)))) # sigmoid(w2 relu(w1 x))
        excited = excited.view(*excited.shape, 1, 1)
        scaled = torch.multiply(x, excited)

        if squeeze_later:
            scaled = scaled.squeeze(0)
        return scaled


def test():
    in_channels, hidden_channels = 4, 2
    block = SEBlock(num_channels=in_channels, hidden_rate=hidden_channels)

    sample = torch.randn(8, in_channels, 32, 32)
    out = block(sample)

    assert out.shape == sample.shape
    print('PASSED: /models/modules/se_block.py')


if __name__ == '__main__':
    test()
