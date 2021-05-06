import torch.nn
import torch

class CSP(nn.Module):
    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., exp_ratio=1.,
                 groups=1, first_dilation=None, down_growth=False, cross_linear=False, block_dpr=None,
                 block_fn, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
            