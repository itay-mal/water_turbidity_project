import sys

import mitsuba as mi
import numpy as np
import drjit as dr

import matplotlib.pyplot as plt


def main():
    mi.set_variant('llvm_ad_rgb')
    from mitsuba import ScalarTransform4f as T
    scene = mi.load_file('./underwater_dist.xml')
    params = mi.traverse(scene)
    images = []
    num_images = 1
    # print(params['floor.vertex_positions'])
    # print(dr.unravel(mi.Point3f, params['aquarium.vertex_positions']))
    # return

    for i in (range(num_images) - np.floor(num_images/2)):
        params['sensor.to_world'] = T.look_at(
            origin=[3 * i, 0, 7],
            target=[0, 0, 1.5],
            up=[0, 1, 0]
        )
        params.update()
        image = mi.render(scene, spp=512)
        images.append(image)
        print(f'image {i} rendered')

    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, im in enumerate(images):
        ax = fig.add_subplot(int(np.ceil(len(images)/3)), min(3, len(images)), i + 1).imshow(im ** (1.0 / 2.2))
        plt.axis("off")

    plt.show()


if __name__ == '__main__':
    main()
