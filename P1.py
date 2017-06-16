#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

os.chdir(os.path.dirname(os.path.realpath(__file__)))   # let working dir same as this file.
#print(os.getcwd()) # the working dir

def colorselection1(img):
    # Grab the x and y size and make a copy of the image
    ysize = img.shape[0]
    xsize = img.shape[1]
    color_select = np.copy(img)

    # Define color selection criteria
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = 0
    green_threshold = 229
    blue_threshold = 0
    ######

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    thresholds = (img[:,:,0] < rgb_threshold[0]) \
                | (img[:,:,1] < rgb_threshold[1]) \
                | (img[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0,0,0]

    return color_select

def colorselection2(img):
    # https://discussions.udacity.com/t/converting-color-thresholded-image-to-binary/257543 
    # https://discussions.udacity.com/t/methods-in-jupyter-notebook/212556/3 
    # https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv 
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

    # Yellow RGB = [255, 255, 0]
    # Yellow BGR = [0, 255, 255]
    yellow = np.uint8([[[255, 255, 0]]])
    hsv_yellow = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # yellow hsv: [[[ 30 255 255]]]

    yellow_min = np.array([20, 100, 100], np.uint8)
    yellow_max = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsv_yellow, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(hsv_yellow, white_min, white_max)

    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(yellow_mask, white_mask))
    return img

def colorselection3(img):
    yellow = np.uint8([[[255, 255, 0]]])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([20,100,100])
    upper_blue = np.array([40,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    return res

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # https://discussions.udacity.com/t/improving-the-draw-line-method/253295/3

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, color=[0,255,0])
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

#reading in an image
#image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#plt.show()

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def pipeline(image):

    x = image.shape[1] / 2
    y = image.shape[0] / 2
    vertices = np.array([[ (0,image.shape[0]), (x-20, y+50), (x+20, y+50), (image.shape[1],image.shape[0]) ]], dtype=np.int32)
    res = region_of_interest(image, vertices)

    res = colorselection2(res)
    res = grayscale(res)
    res = canny(res, 50, 150)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 #np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10 # minimum number of pixels making up a line
    max_line_gap = 3    # maximum gap in pixels between connectable line segments
    res = hough_lines(res, rho, theta, threshold, min_line_len, max_line_gap)

    res = weighted_img(res, image)

    return res

for image in os.listdir("test_images/"):
    img = cv2.imread("test_images/"+image)  # default colorspace = BGR
    print(image)
    res = pipeline(img)
    cv2.imwrite("test_images_output/"+image, res)
