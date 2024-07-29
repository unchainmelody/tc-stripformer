import numpy as np
import torch.nn as nn
# from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5)
        return image_numpy

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)

        real = self.tensor2im(target.data)
        # print(real)
        psnr = PSNR(fake, real)
        ssim, _ = SSIM(fake, real, full=True,win_size=5, channel_axis=2,data_range=1.0)
        # print(fake.astype('uint8'))
        # print(real.astype('uint8'))
        # ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), multichannel=True, win_size=5, channel_axis=2)
        # ssim = SSIM(fake, real, win_size=5, gradient=False, data_range=None, channel_axis=None,
        #                              multichannel=False, gaussian_weights=False, full=False)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
