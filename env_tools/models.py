import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ActorModel(nn.Module):
    
    def __init__(self, num_actions, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_conv_actor = nn.Sequential(
            nn.Conv2d(2, 4, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(4, 8, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(576, 64),

        )
        self.apply(init_params)

    def forward(self, obs):
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)
        # conv_in = obs.unsqueeze(1)

        x = self.image_conv_actor(obs)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)

        return x
    

class DQNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 4, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(4, 8, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.ReLU()
        )
        self.fcs = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, ob):
        x = ob.to(torch.float32)
        x = self.conv_net(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x



if __name__ == "__main__":
    # Define the input tensor
    inputs = torch.randn(1, 3, 224, 224)

    model = ActorModel(output_size=6, device='cpu')
    
    print(model(inputs))