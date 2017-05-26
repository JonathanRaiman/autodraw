import numpy as np
from skimage import draw

def draw_picture(circle_pos, figsize):
    data = np.zeros(figsize, dtype=np.uint8)
    data.fill(255)
    x, _, _ = figsize
    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx, yy, s, c, t in circle_pos:
        if t == "circle":
            rr, cc = draw.circle(xx * x, yy * x, radius=x * s, shape=data.shape)
        elif t == "rectangle":
            rr, cc = draw.polygon([x * xx - x * s, x* xx + x* s,
                                   x * xx + x * s, x* xx - x* s],
                                  [x * yy - x * s, x* yy - x* s,
                                   x * yy + x * s, x* yy + x* s],
                                  shape=data.shape)
        data[rr, cc, :] = c
    return data


def dataset(figsize, max_circles, min_circles, size, scale=None, color=None):
    examples = []
    x, y, _ = figsize
    for i in range(size):
        random_data = []
        num_circles = min_circles if min_circles == max_circles else np.random.randint(min_circles, max_circles)
        for i in range(num_circles):
            circle_scale = scale if scale is not None else np.random.uniform(0, 0.5)
            circle_color = color if color is not None else (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_data.append(
                (np.random.uniform(0, 1.0),
                 np.random.uniform(0, 1.0),
                 circle_scale, circle_color, "circle")
            )
        examples.append(draw_picture(random_data, figsize) / 255.0)
    return examples
