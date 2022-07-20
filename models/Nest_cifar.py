import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import models.modified_linear as modified_linear
from T.timm import models

class Nest(nn.Module):

    def __init__(self, num_classes=10):
        super(Nest, self).__init__()
        self.model=models.factory.create_model("nest_tiny_cifar100",num_classes=32,img_size=32,drop_path_rate=0.1)
        self.model.head=nn.Identity()
        self.fc = modified_linear.CosineLinear(192, num_classes)

    def forward(self, x):
        x1 = self.model(x)
        x = self.fc(x1)

        return x , x1

#     return model
def nest(**kwargs):
    model=Nest(**kwargs)
    return model