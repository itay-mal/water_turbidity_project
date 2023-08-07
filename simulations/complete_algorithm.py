from image_processing_AC import AC_detction
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mplPath


path = "D:/Desktop/000.png"

def main():
    target_1, target_2, shape = AC_detction(path)
    poly_path1 = mplPath.Path(np.array(target_1))
    poly_path2 = mplPath.Path(np.array(target_2))
    mask_1 = np.ndarray(shape, bool)
    mask_2 = np.ndarray(shape, bool)
    t1 = []
    t2 = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask_1[i, j] = poly_path1.contains_point((i, j))
            t1.append()
            mask_2[i, j] = poly_path1.contains_point((i, j))

    plt.hist(mask_1*cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
    plt.show()



if __name__ == "__main__":
    main()
