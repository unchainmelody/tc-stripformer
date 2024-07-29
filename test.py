import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# 读取 TIFF 图像
image1 = Image.open(r'E:\_learn\DeepLearning\Transformer\Stripformer\Stripformer-ECCV-2022--main-change\10401.tiff')

image2 = Image.open(r'E:\_learn\DeepLearning\Transformer\Stripformer\Stripformer-ECCV-2022--main-change\10401_d.tiff')

# 转换为 NumPy 数组
array1 = np.array(image1)
# array1 = np.mean(array1, axis=2)
print(array1.shape)
array2 = np.array(image2)
# array2 = np.mean(array2, axis=2)
print(array2.shape)
# print(array1)
# 计算结构相似性指数
similarity_index, _ = ssim(array1, array2, full=True)

print(f"Structural Similarity Index: {similarity_index}")


# from skimage.metrics import structural_similarity as ssim
# from skimage import io, color
#
# # 读取两张图片
# image1 = io.imread('./10401.tiff')
# image2 = io.imread('./10401_d.tiff')
#
# # 如果图片是彩色的，将其转换为灰度图像
# if image1.ndim == 3:
#     image1 = color.rgb2gray(image1)
# if image2.ndim == 3:
#     image2 = color.rgb2gray(image2)
#
# # 计算SSIM
# ssim_index, _ = ssim(image1, image2, full=True)
#
# print(f"SSIM Index: {ssim_index}")