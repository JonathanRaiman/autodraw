import numpy as np
from skimage import draw, transform
from tensorflow.examples.tutorials.mnist import input_data

def draw_picture(circle_pos, figsize, digits):
    data = np.zeros(figsize, dtype=np.uint8)
    data.fill(255)
    x, _, _ = figsize
    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx, yy, s, c, t in circle_pos:
        if t == "circle":
            rr, cc = draw.circle(xx * x, yy * x, radius=x * s, shape=data.shape)
            data[rr, cc, :] = c
        elif t == "rectangle":
            rr, cc = draw.polygon([x * xx - x * s, x* xx + x* s,
                                   x * xx + x * s, x* xx - x* s],
                                  [x * yy - x * s, x* yy - x* s,
                                   x * yy + x * s, x* yy + x* s],
                                  shape=data.shape)
            data[rr, cc, :] = c
        elif t == "digit":
            digit = digits[np.random.randint(0, len(digits))]
            original_shape = digit.shape
            angle = np.random.uniform(0.0, 360.0)
            digit = transform.rotate(digit, angle, preserve_range=True,
                                     mode="constant", cval=255.0)
            digit = transform.resize(digit, output_shape=(
                max(1, int(original_shape[0] * s)), max(1, int(original_shape[1] * s))),
                preserve_range=True, mode="constant", cval=255.0)
            mask = digit < 255.0
            digit = np.tile(digit[:, :, None], (1, 1, 3))
            left_corner = int(np.floor(xx * x - digit.shape[0] / 2))
            right_corner = left_corner + digit.shape[0]
            top_corner = int(np.floor(yy * x - digit.shape[1] / 2))
            bottom_corner = top_corner + digit.shape[1]

            if bottom_corner == 0:
                top_corner += 1
                bottom_corner = 1

            if right_corner == 0:
                left_corner += 1
                right_corner = 1

            if left_corner < 0:
                digit = digit[-left_corner:, :]
                mask = mask[-left_corner:, :]
                left_corner = 0
            if top_corner < 0:
                digit = digit[:, -top_corner:]
                mask = mask[:, -top_corner:]
                top_corner = 0
            if right_corner > data.shape[0]:
                digit = digit[:data.shape[0]-right_corner]
                mask = mask[:data.shape[0]-right_corner]
                right_corner = data.shape[0]

            if bottom_corner > data.shape[1]:
                digit = digit[:, :data.shape[1]-bottom_corner]
                mask = mask[:, :data.shape[1]-bottom_corner]
                bottom_corner = data.shape[1]

            data[left_corner:right_corner, top_corner:bottom_corner, :][mask] = 255
            data[left_corner:right_corner, top_corner:bottom_corner, :] += digit.astype(np.uint8)
    return data


def dataset(figsize, max_circles, min_circles, size, scale=None, color=None):
    digits = (1.0 - input_data.read_data_sets(
        'MNIST_data', one_hot=True).train.images.reshape(-1, 28, 28)) * 255.0
    digits = digits.astype(np.uint8)

    examples = []
    x, y, _ = figsize
    objects = ["digit"] * 10 + ["circle", "rectangle"]

    num_circles = np.random.randint(min_circles, max_circles, size=size)
    total = num_circles.sum()
    circle_colors = color if color is not None else np.random.randint(0, 255, size=(total, 3))
    positions = np.random.uniform(0.0, 1.0, size=(total, 2))
    object_names = np.random.choice(objects, size=total)
    scales = scale if scale is not None else np.random.uniform(0, 0.5, size=(total))
    if scale is not None:
        mask = object_names == "digit"
        scales[mask] = np.random.uniform(0.7, 1.1, size=mask.sum())
    idx = 0
    for i in range(size):
        random_data = []
        num_circles = min_circles if min_circles == max_circles else np.random.randint(min_circles, max_circles)
        for j in range(num_circles):
            circle_color = color if color is not None else circle_colors[idx]
            object_name =
            circle_scale = scale if scale is not None else scales[idx]
            random_data.append(
                (positions[idx, 0], positions[idx, 1],
                 circle_scale, circle_color, object_names[idx])
            )
            idx += 1
        examples.append(draw_picture(random_data, figsize, digits) / 255.0)
    return examples
