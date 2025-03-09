import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    parser.add_argument('--output_path', default='./output/')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # cv2 method for gray conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    info = []
    with open(args.setting_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n').split(',')
            info += [line]
        rgb_info = info[1:6]
        sigma_s, sigma_r = int(info[6][1]), float(info[6][3])
    cost = {}
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf = JBF.joint_bilateral_filter(img_rgb, img_gray)
    # L1 normalization calculation
    cost['cv2.COLOR_BGR2GRAY'] = np.sum(np.abs(bf.astype('int32') - jbf.astype('int32'))) # cast to int32 to avoid overflow when subtraction
    jbf_file = cv2.cvtColor(jbf, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_path+f'{args.image_path[-5]}_grayimg_cv2_COLOR_BGR2GRAY.png', img_gray)
    cv2.imwrite(args.output_path+f'{args.image_path[-5]}_jbf_cv2_COLOR_BGR2GRAY.png', jbf_file)

    # different setting giving in 1_setting.txt and 2_setting.txt
    for r, g, b in rgb_info:
        img_gray = img_rgb[:, :, 0] * float(r) + img_rgb[:, :, 1] * float(g) + img_rgb[:, :, 2] * float(b)
        jbf = JBF.joint_bilateral_filter(img_rgb, img_gray)
        cost[f'Wr:{r}_Wg:{g}_Wb:{b}'] = np.sum(np.abs(bf.astype('int32') - jbf.astype('int32'))) # cast to int32 to avoid overflow when subtraction
        jbf_file = cv2.cvtColor(jbf, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.output_path+f'{args.image_path[-5]}_grayimg_Wr:{r}_Wg:{g}_Wb:{b}.png', img_gray)
        cv2.imwrite(args.output_path+f'{args.image_path[-5]}_jbf_Wr:{r}_Wg:{g}_Wb:{b}.png', jbf_file)

    print('The costs of different settings are:', cost)


if __name__ == '__main__':
    main()