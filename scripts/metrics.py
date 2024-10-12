import os
import pathlib

import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
# from scipy.misc import imread
# import imageio
import imageio.v2 as imageio
import torch
from scipy import linalg
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2gray
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
import os
import numpy as np
import lpips
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data', help='Path to ground truth data', type=str)
    parser.add_argument('--output', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

class FID():
    """docstring for FID
    Calculates the Frechet Inception Distance (FID) to evalulate GANs
    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one
    of these distributions, while the 2nd distribution is given by a GAN.
    When run as a stand-alone program, it compares the distribution of
    images that are stored as PNG/JPEG at a specified location with a
    distribution given by summary statistics (in pickle format).
    The FID is calculated by assuming that X_1 and X_2 are the activations of
    the pool_3 layer of the inception net for generated samples and real world
    samples respectivly.
    See --help to see further details.
    Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
    of Tensorflow
    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def __init__(self):
        self.dims = 2048
        self.batch_size = 64
        self.cuda = True
        self.verbose = False

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx])
        if self.cuda:
            # TODO: put model into specific GPU
            self.model.cuda()

    def __call__(self, images, gt_path):
        """ images:  list of the generated image. The values must lie between 0 and 1.
            gt_path: the path of the ground truth images.  The values must lie between 0 and 1.
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)

        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_images statistics...')
        m2, s2 = self.calculate_activation_statistics(images, self.verbose)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def calculate_from_disk(self, generated_path, gt_path):
        """
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)
        if not os.path.exists(generated_path):
            raise RuntimeError('Invalid path: %s' % generated_path)

        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_path statistics...')
        m2, s2 = self.compute_statistics_of_path(generated_path, self.verbose)
        print('calculate frechet distance...')
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        print('fid_distance %f' % (fid_value))
        return fid_value

    def compute_statistics_of_path(self, path, verbose):
        npz_file = os.path.join(path, 'statistics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
            imgs = np.array([imageio.imread(str(fn)).astype(np.float32) for fn in files])

            # Bring images to shape (B, 3, H, W)
            imgs = imgs.transpose((0, 3, 1, 2))

            # Rescale images to be between 0 and 1
            imgs /= 255

            m, s = self.calculate_activation_statistics(imgs, verbose)
            np.savez(npz_file, mu=m, sigma=s)

        return m, s

    def calculate_activation_statistics(self, images, verbose):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(images, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(self, images, verbose=False):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        self.model.eval()

        d0 = images.shape[0]
        if self.batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            self.batch_size = d0

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size

        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                # end='', flush=True)
            start = i * self.batch_size
            end = start + self.batch_size

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            # batch = Variable(batch, volatile=True)
            if self.cuda:
                batch = batch.cuda()

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

def calculate_lpips(folder1, folder2):
    # 创建LPIPS模型
    # model = lpips.LPIPS(net='alex')
    model = lpips.LPIPS(net='vgg')

    # 获取两个文件夹中的文件列表
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # 确保文件夹中的文件名相同
    common_files = set(files1).intersection(files2)
    i=0
    # 计算每对相同文件的LPIPS值
    lpips_values = []
    for file in common_files:
        if file.endswith('.png'):
            i=i+1
            # 加载图像
            img1 = Image.open(os.path.join(folder1, file)).convert('RGB')
            img2 = Image.open(os.path.join(folder2, file)).convert('RGB')

            # 调整图像尺寸以匹配模型的输入尺寸
            img1 = img1.resize((256, 256))
            img2 = img2.resize((256, 256))

            # 转换图像为numpy数组
            img1_np = np.array(img1)
            img2_np = np.array(img2)

            # 归一化图像像素值到[0, 1]
            img1_np = img1_np / 255.0
            img2_np = img2_np / 255.0

            # 将图像转换为PyTorch张量并添加批处理维度
            img1_tensor = lpips.im2tensor(img1_np)
            img2_tensor = lpips.im2tensor(img2_np)

            # 计算LPIPS值
            lpips_value = model(img1_tensor, img2_tensor).item()
            lpips_values.append(lpips_value)

        # print(f"LPIPS value for {file}: {lpips_value}")

    # 计算平均LPIPS值
    print(i)
    mean_lpips = np.mean(lpips_values)
    return mean_lpips


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    path_true = args.data
    path_pred = args.output
    LPIPS_value = calculate_lpips(path_pred, path_true)
    psnr = []
    ssim = []
    mae = []
    names = []
    index = 1

    files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)

        img_gt = (imageio.imread(str(fn)) / 255.0).astype(np.float32)
        img_pred = (imageio.imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)

        img_gt = rgb2gray(img_gt)
        img_pred = rgb2gray(img_pred)

        if args.debug != 0:
            plt.subplot('121')
            plt.imshow(img_gt)
            plt.title('Groud truth')
            plt.subplot('122')
            plt.imshow(img_pred)
            plt.title('Output')
            plt.show()

        psnr.append(peak_signal_noise_ratio(img_gt, img_pred, data_range=1))
        ssim.append(structural_similarity(img_gt, img_pred, data_range=1, win_size=51))
        mae.append(compare_mae(img_gt, img_pred))
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % round(np.mean(psnr), 4),
                "SSIM: %.4f" % round(np.mean(ssim), 4),
                "MAE: %.4f" % round(np.mean(mae), 4),
            )
        index += 1

    np.savez(args.output + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
    fid = FID()
    fid_value = fid.calculate_from_disk(path_pred, path_true)


    if args.output:
        # 构建文件路径
        file_path = args.output + '/metrics.txt'

        # 打开文件准备写入，'w'模式会覆盖原有内容
        with open(file_path, 'w') as f:
            # 写入指标，可以根据需要格式化输出
            f.write(f"PSNR: {round(np.mean(psnr), 4):.4f}\n")  # 保留四位小数，具体格式可根据需要调整
            f.write(f"SSIM: {round(np.mean(ssim), 4):.4f}\n")
            f.write(f"MAE: {round(np.mean(mae), 4):.4f}\n")
            f.write(f"FID: {fid_value}\n")
            f.write(f"LPIPS: {LPIPS_value}\n")
            # # 如果names是一个列表或其他可迭代对象，可以遍历写入
            # if isinstance(names, (list, tuple)):
            #     f.write("Names:\n")
            #     for name in names:
            #         f.write(f"{name}\n")
            #         # 如果names是单个值或不需要写入，可以省略上述循环
            # # 注意：根据names的具体结构和您希望保存的内容调整这部分代码
    print(
        "PSNR: %.4f" % round(np.mean(psnr), 4),
        # "PSNR Variance: %.4f" % round(np.var(psnr), 4),
        "SSIM: %.4f" % round(np.mean(ssim), 4),
        # "SSIM Variance: %.4f" % round(np.var(ssim), 4),
        # "MAE Variance: %.4f" % round(np.var(mae), 4),
        "FID:%.6f" % fid_value,
        "LPIPS:%.6f" % LPIPS_value,
        "MAE: %.4f" % round(np.mean(mae), 4),

    )

