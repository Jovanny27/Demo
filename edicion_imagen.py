#Prueba de cambios en imagen
import cv2
import GaussianBlur


initialImage = cv2.imread("image1.jpg")
blurredImage = gaussianBlur(copy.deepcopy(initialImage))
cv2.imwrite("blurred.jpg", blurredImage)

#Borroso
def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

sharpen(initialImage)

#Sepia
def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.filter2D(image, -1, kernel)

sepia(initialImage)

#Relieve
def gaussianBlur(image):
    return cv2.GaussianBlur(image, (35, 35), 0)

gaussianBlur(initialImage)

#Afilado
def emboss(image):
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    return cv2.filter2D(image, -1, kernel)

emboss(initialImage)

#Brillosa
def brightnessControl(image, level):
    return cv2.convertScaleAbs(image, beta=level)

brightnessControl(initialImage, 2)

#Cambio de temperatura
def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))
spreadLookupTable(5,6)
def warmImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

warmImage(initialImage)

def coldImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

coldImage(initialImage)