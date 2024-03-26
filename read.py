import cv2 as cv 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#to read image 
img = cv.imread('image/b.jpg')
cv.imshow('Bright Image',img)
cv.waitKey(0); 

#to convert image to gray scale image
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv.imshow('Bright Image to Gray Level Image',imgGray)
cv.waitKey(0); 


#hist=cv.calcHist([imgGray],[0],None,[256],[0,256])
#plt.plot(hist)
#plt.show()

#Show the histogram of the image
hist,bins = np.histogram(imgGray.flatten(),256,[0,256]) 
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.title('Histogram Of Image')
plt.xlim([0,256])
plt.show()


#Print the necessary features/values that can be useful to identify the shape of the histogram
# Calculate the mean value
mean = imgGray.mean()
print("Mean Value for grayscale image : " + str(mean))

# Calculate the standard deviation value
std= np.std(imgGray)
print("Standard Deviation Value for grayscale image :" , std)  





# Apply Gamma on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
power_low_gamma = np.array(255*(imgGray/255)**3.9,dtype='uint8')
# Display the images in subplots
img3 = cv2.hconcat([power_low_gamma,imgGray])
cv2.imshow('Power Low Gamma Image',img3)
cv2.waitKey(0)
plt.hist(power_low_gamma.flatten(),256,[0,256], color = 'g')
plt.title('Histogram of power low gamma Image')
plt.xlim([0,256])
plt.show()


# creating a Histograms Equalization of a image using cv2.equalizeHist()
equalization = cv2.equalizeHist(imgGray)
# stacking images side-by-side
res = np.hstack((equalization,imgGray))
cv2.imshow('Histograms Equalization Image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(res.flatten(),256,[0,256], color = 'b')
plt.title('Histogram Equalization Image')
plt.xlim([0,256])
plt.show()


#Enhance the contrast of the image using the following techniques and compare between the resulting images
img4 = cv2.hconcat([equalization,power_low_gamma])
cv2.imshow('Histogram Equalization Image Vs Power Low Gamma Image ',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()




    






