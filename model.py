from pdb import set_trace as TT

from einops import rearrange
import torch as th
from torch import nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override


class CustomConvModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=16,
                 fc_size=64,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-1] - 2) * conv_filters
        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        self.conv_1 = nn.Conv2d(obs_space.shape[-3], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)
        # self.conv_2 = nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=3, stride=1, padding=1)

        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
        self.action_branch = SlimFC(self.fc_size, num_outputs)
        self.value_branch = SlimFC(self.fc_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"]  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        # x = nn.functional.relu(self.conv_2(x.float()))
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []

