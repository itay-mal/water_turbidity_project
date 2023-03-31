#!/usr/bin/python
import os
from itertools import product
copies = 1  # num of pictures in same setup
output_dir = './dataset_for_segmentation'  # './variable_dists_and_sigma_s'
config_file_path = './create_dataset.xml'  # 'underwater_dist_05_turbdity_values.xml'
config_file_path_GT = './create_ground_truth.xml'


def main():
    dists = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    rotate_angle = [-20, -14, -7, 0, 7, 14, 20]  # in degrees
    sigma_s = [(0.2, 0.1, 0.1), (0.4, 0.4, 0.4), (0.7, 0.7, 0.7)]
    sigma_a = [(0.3, 0.04, 0.2), (0.45, 0.06, 0.05), (0.35, 0.12, 0.5)]
    # d = 0.6
    # while d < 6:
    #     dists.append(d)
    #     d += 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    idx = 0
    with open('{}/log.txt'.format(output_dir), 'wt') as file:
        for copy, d1, d2, sig_s, sig_a, target_ang in product(list(range(copies)), dists, dists, sigma_s, sigma_a, rotate_angle):
            for f in [config_file_path, config_file_path_GT]:
                file_suffix = '_gt' if f.endswith('_ground_truth.xml') else ''
                cmd = 'mitsuba {} ' \
                      '-o {}/{}{}.png ' \
                      '-Dtarget_1_dist={} -Dtarget_2_dist={} ' \
                      '-Dsigma_sr={} -Dsigma_sg={} -Dsigma_sb={} ' \
                      '-Dsigma_ar={} -Dsigma_ag={} -Dsigma_ab={} ' \
                      '-Dtarget_rotate_angle={}'.format(f, output_dir, idx, file_suffix, d1, d2, sig_s[0], sig_s[1], sig_s[2], sig_a[0], sig_a[1], sig_a[2], target_ang)
                print(cmd)
                os.system(cmd)
            file.write('{},distance_1={},distance_2={},sigma_s={},sigma_a={},angle={}\n'.format(idx, d1, d2, sig_s, sig_a, target_ang))
            file.flush()
            idx += 1


if __name__ == '__main__':
    main()
