from torch_grad_cam.base_cam import BaseCAM
from torch_grad_cam.utils.svd_on_activations import get_2d_projection

class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layers, use_cuda,
                                       reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
