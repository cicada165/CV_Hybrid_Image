import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # # # TODO-BLOCK-BEGIN
    m, n = kernel.shape
    new_img = np.empty(img.shape)

    if len(img.shape)==2:#Grey Scale, 2 channels
        
        h, w = img.shape
        enlarged = np.zeros((h+m-1,w+n-1))
        enlarged[(m - 1)//2:(m - 1)//2 + h,(n - 1)//2:(n - 1)//2 + w] = img

        for i in range(w):
            for j in range(h):
                temp=kernel*enlarged[j:j + m,i:i + n]
                new_img[j,i] = temp.sum()

    else:#RGB, 3 channels
        
        h,w,rgb=img.shape
        enlarged=np.zeros((h+m-1,w+n-1,rgb))
        enlarged[(m - 1)//2:(m - 1)//2 + h,(n - 1)//2:(n - 1)//2 + w] = img

        for i in range(w):
            for j in range(h):
                for k in range(rgb):
                    
                    temp = kernel * enlarged[:,:,k][j:j + m,i:i + n]
                    new_img[:,:,k][j,i]=temp.sum()

    return(new_img)

    
    # raise Exception("TODO in hybrid.py not implemented")
    # # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, np.flip(kernel, (0, 1)))
    # # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    # Create a blank numpy array to represent the kernel
    kernel = np.zeros([height, width])

    # Get the center index of the kernel
    x_center, y_center = height // 2, width // 2

    # Fill in the kernel array
    for i in range(height):
        for j in range(width):
            # Calculating squared distance from the current index to center
            dist_sqr = (i - x_center)**2 + (j - y_center)**2
            kernel[i,j] = np.exp(-dist_sqr/(2*sigma**2))/(2*np.pi*sigma**2)

    sum = kernel.sum()

    # Kernel Normalization
    for i in range(height):
        for j in range(width):
            kernel[i,j] = kernel[i,j]/sum

    return(kernel)
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))
    # # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)
    # # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


