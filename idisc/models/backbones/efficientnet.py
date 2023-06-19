import copy

from torch import nn


class EfficientNet(nn.Module):
    def __init__(self, backend, outputs):
        super(EfficientNet, self).__init__()
        for k, v in backend._modules.items():
            self._modules[k] = copy.deepcopy(v)
        self.outputs = outputs
        self.embed_dims = [40, 64, 176, 2048]

    def forward(self, x):
        features = [x]
        for k, v in self._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return [feature for i, feature in enumerate(features) if i in self.outputs]
