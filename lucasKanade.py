import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_img(img, name):
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.title(name)


def get_derivatives(img1, img2):
    """Getting the derivatives fx, fy ,ft."""

    kernel = 0.25 * np.array(([-1, 1], [-1, 1]))
    kernel2 = 0.25 * np.array(([-1, -1], [1, 1]))
    kernel = np.fliplr(kernel)

    fx = cv2.filter2D(img1, -1, kernel) + cv2.filter2D(img2, -1, kernel)
    fy = cv2.filter2D(img1, -1, kernel2) + cv2.filter2D(img2, -1, kernel2)
    ft = cv2.filter2D(img1, -1, 0.25 * np.ones((2, 2))) + \
        cv2.filter2D(img2, -1, -0.25 * np.ones((2, 2)))

    return (fx, fy, ft)


def lucas_kanade(img1, img2,window):
    """Lucase Kanade algorithm without pyramids.

    Implemented with convolution"""

    fx, fy, ft = get_derivatives(img1, img2)

    denom = cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy**2, -1, window) - \
        cv2.filter2D((fx*fy), -1, window)**2
    denom[denom == 0] = np.inf

    u = (-cv2.filter2D(fy**2, -1, window)*cv2.filter2D(fx*ft, -1, window) +
         cv2.filter2D(fx*fy, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
        denom
    v = (cv2.filter2D(fx*ft, -1, window)*cv2.filter2D(fx*fy, -1, window) -
         cv2.filter2D(fx**2, -1, window)*cv2.filter2D(fy*ft, -1, window)) / \
        denom

    return (u, v)

# Reading the image
img1 = cv2.imread('oflow/taxi1.pgm', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('oflow/taxi2.pgm', cv2.IMREAD_GRAYSCALE)

# Convert the image to floating point for calculations
img1 = np.float32(img1)
img2 = np.float32(img2)

# Calculate optical flow using two different algorithms
u, v = lucas_kanade(img1, img2, np.ones((3, 3)))
flow = cv2.calcOpticalFlowFarneback(img1, img2, 0.5, 3, 15, 3, 5, 1.2, 0)


# Create grid for display
x = np.arange(0, img1.shape[1], 1)
y = np.arange(0, img1.shape[0], 1)
x, y = np.meshgrid(x, y)

# Display
display_img(img1, 'My lucas_kanade')
step = 3
plt.quiver(x[::step, ::step], y[::step, ::step],
           u[::step, ::step], v[::step, ::step],
           color='g', pivot='middle', headwidth=2, headlength=3)

display_img(img1, 'Farneback')
plt.quiver(x[::step, ::step], y[::step, ::step],
           flow[::step, ::step, 0], flow[::step, ::step, 1],
           color='g', pivot='middle', headwidth=4, headlength=6)
plt.show()
