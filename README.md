1st:
```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
image = cv2.imread('grayscale.jpg',cv2.IMREAD_GRAYSCALE)

cv2_imshow(image)
img = image.astype(np.float64)


mean_value = np.mean(img)
std_value = np.std(img)

shifted_image = np.roll(img,shift=20,axis=1)
cv2_imshow(shifted_image)
img_flat =img.flatten()
shifted_flat = shifted_image.flatten()

correlattion_coeff = np.corrcoef(img_flat,shifted_flat)[0,1]

print("mean of image is :", mean_value)
print("std deviation of image is :", std_value)
print("correlation coefficient of images are :", correlattion_coeff)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

2nd:
```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image = cv2.imread('grayscale.jpg')

if image is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2_imshow(gray)

tx,ty = 100 , 50
T = np.float32([[1,0,tx],[0,1,ty]])
h,w = image.shape[:2]

translator = cv2.warpAffine(image,T,(w,h))
cv2_imshow(translator)

scaled = cv2.resize(image,None,fx=0.5,fy=0.5)
cv2_imshow(scaled)

angle = 45
center = (w//2,h//2)
R = cv2.getRotationMatrix2D(center,angle,1)

rotated = cv2.warpAffine(image,R,(w,h))
cv2_imshow(rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
3rd:
```
#write a program of contrast streching of a low contrast image,histogram,and histogramm equalization and display of a bit planes of an image

import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

img = cv2.imread(r"low.jpg" , cv2.IMREAD_GRAYSCALE)

min_val = np.min(img)
max_val = np.max(img)

constrast_streched = ((img-min_val) / (max_val -min_val)) * 255
constrast_streched = constrast_streched.astype(np.uint8)

hist = cv2.calcHist([img],[0],None,[256],[0,256])

hist_eq = cv2.equalizeHist(img)

bit_planes = []
for i in range(8):
    bit_plane = (img >> i) & 1
    bit_planes.append(bit_plane * 255)
print('img')
cv2_imshow(img)
print('contrast streched')
cv2_imshow(constrast_streched)
print('histogram_eq')
cv2_imshow(hist_eq)

for i in range (8):
    print(f"Bit plane {i}")
    cv2_imshow(bit_planes[i])

plt.figure()
plt.title("histogram of original image")
plt.xlabel("gray level")
plt.ylabel("frequency")
plt.plot(hist)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
```
4th:
```
#implement image smoothening filters mean and median filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r'noisy.jpg')

mean_filtered = cv2.blur(img,(3,3))
median_filtered = cv2.medianBlur(img,3)

plt.figure(figsize =(10,4))

plt.subplot(1,3,1)
plt.imshow(img , cmap='gray')
plt.title("original image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mean_filtered , cmap='gray')
plt.title("mean filtered image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(median_filtered , cmap='gray')
plt.title("median filter image")
plt.axis("off")

plt.show()
```
5th:
```
#display of FFt of an image and apply 2d foriour transform to represent the content of an image using the discrete foriour transform and masking with dft
#2nd image
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(r'image.jpg' , cv2.IMREAD_GRAYSCALE)
rows,cols = image.shape

#1d
fft_1d = np.fft.fft(image[0])
fft_1d_shift = np.fft.fftshift(fft_1d)
magnitude_1d = np.abs(fft_1d_shift)

#2d
fft_2d = np.fft.fft2(image)
fft_2d_shift = np.fft.fftshift(fft_2d)
magnitude_spectrum = 10*np.log(np.abs(fft_2d_shift) * 1)


crow ,ccol = rows//2 , cols//2
mask = np.zeros((rows,cols), np.uint8)

mask[crow-30:crow+30,ccol-30:ccol+30] = 1

filtered_fft = fft_2d_shift * mask

ifft_shift = np.fft.ifftshift(filtered_fft)
reconstructed_image = np.fft.ifft2(ifft_shift)
reconstructed_image = np.abs(reconstructed_image)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2,3,2)
plt.plot(magnitude_1d)
plt.title('1D FFT (row spectrum)')

plt.subplot(2,3,3)
plt.imshow(magnitude_spectrum , cmap='gray')
plt.title('2D FFT magnitute spectrum')

plt.subplot(2,3,4)
plt.imshow(mask , cmap='gray')
plt.title('low-pass mask')

plt.subplot(2,3,5)
plt.imshow(20*np.log(np.abs(filtered_fft)+1), cmap='gray')
plt.title('Masked spectrum')

plt.subplot(2,3,6)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')


plt.tight_layout()
plt.show()

```
6th:
```
#6a 6b  image sharpening filter and edge detection filter using gradient filters
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"cat.jpg", cv2.IMREAD_GRAYSCALE)

kernel1 = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

kernel2 = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])

kernel3 = np.array([
     [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

sharpened1 = cv2.filter2D(img,-1,kernel1)
sharpened2= cv2.filter2D(img,-1,kernel2)
sharpened3= cv2.filter2D(img,-1,kernel3)

plt.figure(figsize=(12,12))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("original image")

plt.subplot(2,2,2)
plt.imshow(sharpened1, cmap='gray')
plt.title("sharpened 8 n")

plt.subplot(2,2,3)
plt.imshow(sharpened2, cmap='gray')
plt.title("sharpened 4 n")

plt.subplot(2,2,4)
plt.imshow(sharpened3, cmap='gray')
plt.title("sharpened mild")

plt.show()
```
7th:
```
import cv2
import numpy as np
from scipy.signal import wiener
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


img = cv2.imread('cat.jfif', cv2.IMREAD_GRAYSCALE)


img = img.astype(np.float32) / 255.0

# 1. ADD NOISE
# Gaussian Noise
gaussian_noise = np.random.normal(0, 0.05, img.shape)
noisy_gaussian = np.clip(img + gaussian_noise, 0, 1)

# Salt & Pepper Noise
noisy_sp = img.copy()
prob = 0.02
rand = np.random.rand(*img.shape)
noisy_sp[rand < prob/2] = 0
noisy_sp[rand > 1 - prob/2] = 1

# 2. MEDIAN FILTER (NON-LINEAR) for Salt & Pepper Noise
median_restored = cv2.medianBlur((noisy_sp * 255).astype(np.uint8), 5)

# 3. WIENER FILTER (SPATIAL) FOR Gaussian Noise
wiener_restored = wiener(noisy_gaussian, (5, 5))


# DISPLAY RESULTS
plt.figure(figsize=(4,10))
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.show()

plt.figure(figsize=(4,10))
plt.imshow(noisy_gaussian, cmap='gray')
plt.title("Gaussian Noise")
plt.axis('off')
plt.show()

plt.figure(figsize=(4,10))
plt.imshow(noisy_sp, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis('off')
plt.show()

plt.figure(figsize=(4,10))
plt.imshow(median_restored, cmap='gray')
plt.title("Median Restored")
plt.axis('off')
plt.show()

plt.figure(figsize=(4,10))
plt.imshow(wiener_restored, cmap='gray')
plt.title("Wiener (Spatial)")
plt.axis('off')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
```
8th
```
#8 implementation of image intensity slicing techniques for image enhancement.
import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('tiger.jpg',cv2.IMREAD_GRAYSCALE)

# Define Intensity Range
lower = 90
upper = 180

# 1. Binary Intensity Slicing
binary_slice = np.zeros_like(gray)
binary_slice[(gray >= lower) & (gray <= upper)] = 255

# 2. Gray-Level Slicing (Without Background)
gray_slice_no_bg = np.zeros_like(gray)
gray_slice_no_bg[(gray >= lower) & (gray <= upper)] = gray[(gray >= lower) & (gray <= upper)]

# 3. Gray-Level Slicing (With Background)
gray_slice_with_bg = gray.copy()
gray_slice_with_bg[(gray >= lower) & (gray <= upper)] = 255

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_slice, cmap='gray')
plt.title("Binary Intensity Slicing")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gray_slice_no_bg, cmap='gray')
plt.title("Gray-Level Slicing (no Background)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(gray_slice_with_bg, cmap='gray')
plt.title("Gray-Level Slicing (With Background)")
plt.axis('off')

plt.tight_layout()
plt.show()
```
9th:
```
#9 canny edge detction algo to images and compare it with the existing edge detection algorithm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

gray = cv2.imread('white_rose.jpg',cv2.IMREAD_GRAYSCALE)

#canny edge detction
canny = cv2.Canny(gray ,100 , 200)

#sobel edge detection
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)
sobel = np.uint8(sobel)

#prewitt Edge detection
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittx = cv2.filter2D(gray ,-1 , kernelx)
prewitty = cv2.filter2D(gray ,-1 , kernely)
prewitt =  np.sqrt(prewittx**2 + prewitty**2)
prewitt = np.uint8(prewitt)

#laplacian edge detection
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

#metrics calculation
methods = {
    "canny": canny,
    "sobel" : sobel,
    "prewitt" : prewitt,
    "laplacian" : laplacian
}

print("\n--- Performance Comparison ---")
print("\n methods\tedge density\tPSNR\tSSIM")
for name ,edge_img in methods.items():
  #normalize
  edge_norm = cv2.normalize(edge_img,None,0,255,cv2.NORM_MINMAX)
  #edge detection
  edge_density = np.sum(edge_img > 0) / edge_img.size
  #psnr
  mse = np.mean((gray - edge_img)**2)
  psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')
  #ssim
  ssim_val = ssim(gray,edge_norm)
  print(f"{name:10s}\t{edge_density:.4f}\t\t{psnr:.2f}\t\t{ssim_val:.4f}")

#visualisation
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(canny, cmap='gray')
plt.title("canny filter")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(sobel, cmap='gray')
plt.title("sobel filter")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(prewitt, cmap='gray')
plt.title("prewitt filter")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(laplacian, cmap='gray')
plt.title("laplacian filter")
plt.axis('off')

plt.tight_layout()
plt.show()
```
10th:
```
#10  edge detction algo to images
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

gray = cv2.imread('butterfly.png',cv2.IMREAD_GRAYSCALE)

#sobel edge detection
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)
sobel = np.uint8(sobel)

#prewitt Edge detection
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittx = cv2.filter2D(gray ,-1 , kernelx)
prewitty = cv2.filter2D(gray ,-1 , kernely)
prewitt =  np.sqrt(prewittx**2 + prewitty**2)
prewitt = np.uint8(prewitt)

#laplacian edge detection
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

#visualisation
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(sobel, cmap='gray')
plt.title("sobel filter")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(prewitt, cmap='gray')
plt.title("prewitt filter")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(laplacian, cmap='gray')
plt.title("laplacian filter")
plt.axis('off')

plt.tight_layout()
plt.show()
```
