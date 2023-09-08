#!/usr/bin/python
import os
from itertools import product
copies = 1  # num of pictures in same setup
output_dir = './renders/with_ior_scatter_sweep'
config_file_path = './create_dataset_with_ior.xml'  # 'underwater_dist_05_turbdity_values.xml'
# config_file_path_GT = './create_ground_truth.xml'


def main():
    dists = [float(i)/5 for i in range(11)]
    # rotate_angle = [-20, -14, -7, 0, 7, 14, 20]  # in degrees
    sigma_a = (0.2, 0.2, 0.2) # [0.00001 + float(i)/10 for i in range(11)]
    sigma_s = (0.2, 0.2, 0.2) # [0.00001 + float(i)/10 for i in range(11)]
    # light = [0.5 + float(i)/5 for i in range(11)]
    # floor_ref = [float(i)/10 for i in range(11)]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    idx = 0
    d1_base = 0.6
    d2_base = 1.2
    # sig_a = (0.4, 0.15, 0.05)
    floor = 0.2
    light = 1.5
    # sig_s = (0.2, 0.2, 0.2)
    target_ang = 0
    with open('{}/log.txt'.format(output_dir), 'wt') as file:
        for copy, d, in product(list(range(copies)), dists):
            d1 = d1_base + d 
            d2 = d2_base + d 
            light_r = light_b = light_g = light
            # for f in [config_file_path, config_file_path_GT]:
            for f in [config_file_path]:
                file_suffix = '_gt' if f.endswith('_ground_truth.xml') else ''
                cmd = 'mitsuba {} ' \
                        '-o {}/{:04}{}.png ' \
                      '-Dtarget_1_dist={} -Dtarget_2_dist={} ' \
                      '-Dsigma_sr={} -Dsigma_sg={} -Dsigma_sb={} ' \
                      '-Dsigma_ar={} -Dsigma_ag={} -Dsigma_ab={} ' \
                      '-Dlight_r={} -Dlight_g={} -Dlight_b={} ' \
                      '-Dtarget_rotate_angle={} -Dfloor_ref={}'.format(f, output_dir, idx, file_suffix, d1, d2, sigma_s[0], sigma_s[1], sigma_s[2], sigma_a[0], sigma_a[1], sigma_a[2], light_r, light_g, light_b, target_ang, floor)
                print(cmd)
                os.system(cmd)
            file.write('{:04}:distance_1={}:distance_2={}:sigma_s={}:sigma_a={}:angle={}:light={}:floor_ref={}\n'.format(idx, d1, d2, sigma_s, sigma_a, target_ang, (light_r, light_g, light_b), floor))
            file.flush()
            idx += 1


if __name__ == '__main__':
    main()
