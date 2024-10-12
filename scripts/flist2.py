import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

# 遍历大文件夹中的所有子文件夹
for root, dirs, files in os.walk(args.path):
    images = []
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))

    images = sorted(images)
    # 输出文件列表到同级位置
    output_path = os.path.join(root, os.path.basename(root) + '.flist')
    np.savetxt(output_path, images, fmt='%s')
