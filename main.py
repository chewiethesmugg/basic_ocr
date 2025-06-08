import cv2
from PIL import Image
import pytesseract


# opening and showing inage
image_name="IMG_1047.jpg"
file_name = "test_images/"+image_name

print(file_name)

def show_image(file):
    im = Image.open(file)
    print(im.show())

#inverting image

def invert_image(image_file):
    img=cv2.imread(file_name)
    inverted_image = cv2.bitwise_not(img)
    inv_name= "temp/inverted_"+image_name
    cv2.imwrite(inv_name,inverted_image)
    return inv_name

invertedImage = invert_image(file_name)
#show_image(invertedImage)

def greyscale(image_file):
    img = cv2.imread(image_file)
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_image = greyscale(file_name)
gray_name= "temp/gray_"+image_name
cv2.imwrite(gray_name, gray_image)


#binarizing the image
#we need to greyscale it first

thresh, im_bw = cv2.threshold(gray_image, 86, 500, cv2.THRESH_BINARY)
bin_name="temp/bin_"+image_name
print(bin_name)
cv2.imwrite(bin_name,im_bw)
#show_image(bin_name)

#NOISE REMOVAL

def noise_rem(image_file):
    import numpy as np
    #the kernal for erosion
    kernal = np.ones((1,1), np.uint8)
    image = cv2.dilate(image_file, kernal, iterations = 1)
    kernal = np.ones((1,1), np.uint8)

    image = cv2.erode(image, kernal, iterations = 1)
    image = cv2.morphologyEx(image_file, cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image, 3)

    return (image)

print(bin_name)
no_noise = noise_rem(bin_name)
cv2.imwrite("temp/no_noise_"+image_name,no_noise)
show_image("temp/no_noise_"+image_name)