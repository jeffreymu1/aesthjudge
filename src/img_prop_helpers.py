import numpy as np
import cv2

### IMAGE PROPERTY HELPERS ###

# light/dark of image
def brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.mean(gray)


# how spread out grayscale is
def contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.std(gray)


# mean of saturation
def saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return np.mean(hsv[:, :, 1])


# left right symmetry
def symmetry_lr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    half = w // 2
    
    left = gray[:, :half]
    right = cv2.flip(gray[:, -half:], 1)

    min_w = min(left.shape[1], right.shape[1])
    left = left[:, :min_w]
    right = right[:, :min_w]
    
    return float(np.corrcoef(left.flatten(), right.flatten())[0, 1])


# top down symmetry
def symmetry_td(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape
    half = h // 2
    
    top = gray[:half, :]
    bottom = cv2.flip(gray[-half:, :], 0)

    min_h = min(top.shape[0], bottom.shape[0])
    top = top[:min_h, :]
    bottom = bottom[:min_h, :]
    
    return float(np.corrcoef(top.flatten(), bottom.flatten())[0, 1])


# light balance between left and right
def light_balance_lr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    left_mean = np.mean(gray[:, :w//2])
    right_mean = np.mean(gray[:, w//2:])

    return abs(left_mean - right_mean)


# light balance between top and down
def light_balance_td(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape
    top_mean = np.mean(gray[:h//2, :])
    bottom_mean = np.mean(gray[h//2:, :])
    
    return abs(top_mean - bottom_mean)


# measures distinct hues
def hue_diversity(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [36], [0,180])
    hist = hist / np.sum(hist)

    return -np.sum(hist * np.log2(hist + 1e-7))


# dynamic range (constrast between darkest and lightest)
def dynamic_range(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.percentile(gray, 95) - np.percentile(gray, 5)


# measures strong edging
def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var


# rule of thirds (extreme vals at rule of thirds)
def thirds_balance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    h, w = gray.shape
    thirds_y = [h/3, 2*h/3]
    thirds_x = [w/3, 2*w/3]

    cy, cx = np.unravel_index(np.argmax(gray), gray.shape)
    grid_pts = [(y, x) for y in thirds_y for x in thirds_x]
    dist = min(np.hypot(cy - y, cx - x) for y, x in grid_pts)

    return dist / np.sqrt(h**2 + w**2)


# constract between finer grain vs blurry
def texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    return np.std(grad_mag)