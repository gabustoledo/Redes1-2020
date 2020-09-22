import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy


# Function that reads an image.
# Input: Image name, with its extension included.
# Output: Image in the form of a pixel matrix with values ​​from 0 to 255.
def readImg(name):
    # Image name, 0 -> To read the grayscale image.
    img = cv2.imread(name, 0)
    return img


# Function that saves an image.
# Input: Image in the form of a pixel matrix with values ​​from 0 to 255,
#        Image name, with its extension included.
def saveImg(img, name):
    cv2.imwrite(name, img)


# Function that shows an image.
# Input: Image in the form of a pixel matrix with values ​​from 0 to 255,
#        Image name.
def showImg(img, name):
    plt.imshow(img, cmap='gray')
    plt.title(name), plt.xticks([]), plt.yticks([])
    plt.show()


# Function that performs the convolution between the kernel and the image to apply the filter.
# Input: Image and Kernel.
# Output: Filtered image.
def convolution(img, kernel):
    edge = int(len(kernel) / 2)
    filtered = copy.deepcopy(img)
    for i in range(len(img) - edge):
        for j in range(len(img[i]) - edge):
            if (i >= edge):
                if (j >= edge):
                    filtered[i][j] = operation(img, kernel, i - edge, j - edge)
    return filtered


# Function that executes the convolution operation.
# Input: Matrix, kernel, top right index of the image portion.
# Output: Number between 0 and 255.
def operation(img, kernel, i, j):
    acum = 0
    for k in range(len(kernel)):
        for l in range(len(kernel[k])):
            acum += (kernel[k][l] * img[i + k][j + l])
    if (acum < 0):
        return 0
    elif (acum > 255):
        return 255
    else:
        return acum


# Function that normalize a matrix..
# Input: Matrix to normalize.
# Output: Normalized matrix.
def normalization(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= 1 / 256
    return matrix


# Function that calculates the Fourier transform
# Input: Matrix.
# Output: Frequency matrix.
def fourier(img):
    f = np.fft.fft2(img)
    fShift = np.fft.fftshift(f)
    magnitudeSpectrum = 20 * np.log(np.abs(fShift))
    return magnitudeSpectrum


if __name__ == '__main__':
    # """"
    # Matrix for applying Gaussian blur.
    kernelBlur = [[1, 4, 6, 4, 1],
                  [4, 16, 24, 16, 4],
                  [6, 24, 36, 24, 6],
                  [4, 16, 24, 16, 4],
                  [1, 4, 6, 4, 1]]
    kernelBlur = normalization(kernelBlur)

    # Edge detection matrix.
    kernelEdge = [[1, 2, 0, -2, -1],
                  [1, 2, 0, -2, -1],
                  [1, 2, 0, -2, -1],
                  [1, 2, 0, -2, -1],
                  [1, 2, 0, -2, -1]]

    # The image opens.
    img = readImg('lena512.bmp')

    # Applying the convolution with the respective kernel.
    imgEdge = convolution(img, kernelEdge)
    imgBlur = convolution(img, kernelBlur)

    # Show filtered images.
    showImg(img, 'Original')
    showImg(imgEdge, 'Edge detection')
    showImg(imgBlur, 'Gaussian blur')

    #  Fourier Transform.

    # Fourier transform to original image.
    spectrumOriginal = fourier(img)
    showImg(spectrumOriginal, 'Fourier transform to\noriginal image')

    # Fourier transform to Gaussian blur image.
    spectrumBlur = fourier(imgBlur)
    showImg(spectrumBlur, 'Fourier transform to\nGaussian blur image')

    # Fourier transform to image with edge detection.
    spectrumEdge = fourier(imgEdge)
    showImg(spectrumEdge, 'Fourier transform to\nimage with edge detection')

    # Save images.
    saveImg(imgBlur, 'gaussianBlur.jpg')
    saveImg(imgEdge, 'edge.jpg')

    """

    ##### Test
    test = [[255, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30]]

    kernelTest = [[0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0]]

    filterTest = convolution(test, kernelTest)

    plt.imshow(filterTest, cmap='gray')
    plt.show()

    print(filterTest[0])
    print(filterTest[1])
    print(filterTest[2])
    print(filterTest[3])
    print(filterTest[4])

    """
