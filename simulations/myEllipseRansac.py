import numpy as np
import random
from skimage.measure import EllipseModel

class myEllipseRansac:
    """
    Ransac based model to estimate ellipse from given set of points
    """
    def __init__(self, xy_data, n=400):
        """
        xy_data - coordinates of the points to evaluate np.array(N,2)
        n       - number of iterations
        
        returns:
        ellipse params: (x0 [pix], y0 [pix], width [pix], 
                         height [pix], theta [rad])
        """
        self.xy_data = xy_data
        self.num_points = xy_data.shape[0]
        self.n = n  # TODO: can we determine dynamically? i.e by len(x_data)
        self.min_score = 1e20 # initialize to arbitrary high value
        self.best_model = None
        self.execute_ransac()
        assert self.best_model is not None, "couldn't find any model ☹"

    def get_params(self):
        return self.best_model.params

    def random_sampling(self):
        # 5 unique idxes in [0..len(self.x_data)]
        idxs = random.sample(range(self.num_points), 5)
        return self.xy_data[idxs]

    @staticmethod
    def make_model(sample):
        # use skimage EllipseModel to predict ellipse given the 5 points
        ellipse = EllipseModel()
        ellipse.estimate(sample)
        return ellipse

    def eval_model(self, model): 
        # get model score
        my_score = np.sum(np.abs(model.residuals(self.xy_data))) # rediculously unreadable
        x ,y, width, height, angle = model.params
        my_score *= (width/height if width > height else height/width) ** 20  # penalize for non-equal axes
        if abs(angle)>0.1: # penalize for angle
            my_score *= 1000
        return my_score

    def execute_ransac(self):
        # find best model
        for _ in range(self.n):
            model = self.make_model(self.random_sampling())
            temp_score = self.eval_model(model)

            if temp_score < self.min_score:
                self.best_model = model
                self.min_score = temp_score