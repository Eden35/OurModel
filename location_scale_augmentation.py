import numpy as np
import random
from scipy.special import comb


class FIESTA(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000): #100000
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])  # 최종적으로는 0~1로 normalization 해줌

    def Local_Location_Scale_Augmentation(self, image, mask):

        N, C, H, W = image.shape

        output_images = []

        for i in range(N):
            img_single = image[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            mask_single = mask[i].squeeze(0)  # (1, H, W) -> (H, W)
            output_image_single = np.zeros_like(img_single)
            mask_single = mask_single.astype(np.int32)

            output_image_single[mask_single == 0] = self.location_scale_transformation(
                self.non_linear_transformation(img_single[mask_single == 0], inverse=True, inverse_prop=0.5))

            if (mask_single == 1).sum() > 0:
                output_image_single[mask_single == 1] = self.location_scale_transformation(
                    self.non_linear_transformation(img_single[mask_single == 1], inverse=True, inverse_prop=0.8))

            if self.background_threshold >= self.vrange[0]:
                output_image_single[img_single <= self.background_threshold] = img_single[
                    img_single <= self.background_threshold]
            output_images.append(output_image_single.transpose(2, 0, 1))  # (H, W, C) -> (C, H, W)

        output_images = np.stack(output_images, axis=0)



        return output_images
