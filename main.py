import cv2
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv

#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
import numpy as np

# image functions

# display function
def show_image(file):
    im = Image.open(file)
    print(im.show())

#inverting image
def invert_image(image_file):
    img=cv2.imread(image_file)
    inverted_image = cv2.bitwise_not(img)
    inv_name= "temp/inverted_"+image_file
    print("inverted file name: "+ inv_name)
    cv2.imwrite(inv_name,inverted_image)
    return inverted_image

#greyscaling image
def greyscale(image_file):
    img = cv2.imread(image_file)
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#binarizing the image
def binarization(image_file):
    image_file = cv2.imread(image_file)
    thresh, im_bw = cv2.threshold(image_file, 100, 300, cv2.THRESH_BINARY)
    return  im_bw
    

#noise removal
#greyscale image first before input
def noise_rem(image_file):
    image_file=cv2.imread(image_file)
    #the kernal for erosion
    kernal = np.ones((1,1), np.uint8)
    image = cv2.dilate(image_file, kernal, iterations = 1)
    kernal = np.ones((1,1), np.uint8)

    image = cv2.erode(image, kernal, iterations = 1)
    image = cv2.morphologyEx(image_file, cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image, 3)
    return (image)

# NOT MY CODE, CHECK HYPERLINK IN IMPORT
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(image_file):
    angle = getSkewAngle(image_file)
    return  rotateImage(image_file,-1.0 * angle)

#REMOVE BORDERS BEFORE DESKEWING
def fixingSkew(image_file):
    image_file = cv2.imread(image_file)
    fixed = deskew(image_file)
    outputpath = "C:\\Users\\user1\\Desktop\\PROJECTS\\text_puller\\temp\\"+"deskewed_"+imageFileName
    cv2.imwrite(outputpath, fixed)

#REQUIRES BINARY IMAGE
def remove_borders(image_file):
    image_file = cv2.imread(image_file)
    image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    contours, heiarchy = cv2.findContours(image_file, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key= lambda x:cv2.contourArea(x))
    cnt = cntSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image_file[y:y+h, x:x+w]
    return(crop)

#MAIN

load_dotenv()

#clear temp folder before running code again


temp_path = os.environ['temp_path']
for file in os.listdir(temp_path):
    file_path = os.path.join(temp_path,file)
    os.remove(file_path)
print("Deleted previous temp files")

# loop through input images
pwd_path = os.environ['test_path']
os.chdir(pwd_path)

print(pwd_path)
pwd = os.fsencode(pwd_path)

for imageFile in os.listdir(pwd_path):
    if imageFile.endswith(".jpg"):
        imageFileName = os.fsdecode(imageFile)
        print("input file name: "+imageFileName)
        

        invertedImage = invert_image(imageFileName)
        #outputpath = temp_path+"inverted_"+imageFileName
        outputpath = temp_path+imageFileName
        print("output: "+outputpath)
        cv2.imwrite(outputpath,invertedImage)

        greyscaleImage = greyscale(imageFileName)
        #greyOutputName = temp_path+"greyscale_"+imageFileName
        greyOutputName = temp_path+imageFileName
        cv2.imwrite(greyOutputName,greyscaleImage)
        
        bwImage = binarization(greyOutputName)
        #bwOutputName = temp_path+"bw_"+imageFileName
        bwOutputName = temp_path+imageFileName
        cv2.imwrite(bwOutputName,bwImage)
        print("BW: "+bwOutputName)

        #noiRemImage = noise_rem(bwOutputName)
        #noiseOutputName = temp_path+"noise_"+imageFileName
        #noiseOutputName = temp_path+imageFileName
        #cv2.imwrite(noiseOutputName,noiRemImage)

        #no_borders = remove_borders(noiseOutputName)
        #borderOutputName =temp_path+"border_"+imageFileName
        #cv2.imwrite(borderOutputName,no_borders)
        #deskewing is messed up
        #fixingSkew(borderOutputName)



