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

img=cv2.imread(file_name)
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted_"+image_name,inverted_image)
print("done")
show_image("temp/inverted_"+image_name)