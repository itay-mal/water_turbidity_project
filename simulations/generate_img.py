#!/usr/bin/python
import os
from itertools import product
copies = 1  # num of pictures in same setup
output_dir = './variable_dists_and_sigma_s'
config_file_path = 'underwater_dist_05_turbdity_values.xml'


def main():
    dists = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    sigma_s = [(0.2, 0.1, 0.1), (0.4, 0.4, 0.4), (0.7, 0.7, 0.7)]
    # d = 0.6
    # while d < 6:
    #     dists.append(d)
    #     d += 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    idx = 0
    with open('{}/log.txt'.format(output_dir), 'wt') as file:
        for copy, d1, d2, sig in product(list(range(copies)), dists, dists, sigma_s):
            cmd = 'mitsuba {} ' \
                  '-o {}/{}.png -Dtarget_1_dist={} ' \
                  '-Dtarget_2_dist={} -Dsigma_sr={} -Dsigma_sg={} ' \
                  '-Dsigma_sb={}'.format(config_file_path, output_dir, idx, d1, d2, sig[0], sig[1], sig[2])
            print(cmd)
            os.system(cmd)
            file.write('{},distance_1={},distance_2={},sigma_s={}\n'.format(idx, d1, d2, sig))
            file.flush()
            idx += 1


if __name__ == '__main__':
    main()
