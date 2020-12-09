#!/usr/bin/env python
# coding: utf-8

# # Deep Convolutional Neural Networks For Accurate Coronavirus Detection                                                 Through Chest CT Screening

# # Tasks Performed
# 1] Visualize the CT
# 2] Filtering Algorithm
# 3] Convert to Grayscale
# 4] Convert to PNG
# 5] Resize the CT
# 6] Morphological Transformation

# # 1] Visualize the CT Scans

# In[5]:


# To converts a 16bit tif CT slices to a Visualizable format(on my monitor)
#Here I import the essential libraries
import os
import PIL.Image as pil_image
import io
import numpy as np
path='C:/Users/Abhishek Nagrecha/Desktop/SR_2/IM00008.tif'
save_path='C:/Users/Abhishek Nagrecha/Desktop/saved_img/visualize/IM00008v.tif'
with open(path, 'rb') as f:
    tif = pil_image.open(io.BytesIO(f.read()))
array=np.array(tif)
max_val=np.amax(array)
normalized=(array/max_val)
im = pil_image.fromarray(normalized)
im.save(save_path) 


# # 2] Filtering Algorithm to remove the scans having closed lungs.
# 

# In[2]:


#Here I import the essential libraries
import os
import numpy as np
import cv2
import shutil
from  PIL import Image
#Here i have given the path of 1 covid patient containing 3 different types of CT slices(thickness wise)
#That is 262 slices
data_path='C:/Users/Abhishek Nagrecha/Desktop/SEM4/Final_project/Covid-CT-iran/COVID-CTset/COVID-CTset/covid1/patient1/'
data_files=[]
for r,d,f in os.walk(data_path): #add the path of the CT scan images of the patient
  for file in f:
    if '.tif' in file:
      data_files.append(os.path.join(r,file)) #get the images path from the data_folder


# In[3]:


selected=[]          
zero=[]
names=[]
for img_path in data_files:
    names.append(img_path)
    pixel=cv2.imread(img_path,cv2.IMREAD_UNCHANGED ) #read the TIFF file
    sp=pixel[240:340,120:370] #Crop the region
    counted_zero=0
    for i in np.reshape(sp,(sp.shape[0]*sp.shape[1],1)):
        if i<300: #count the number of pixel values in the region less than 300
            counted_zero+=1
    zero.append(counted_zero) #add the number of dark pixels of the image to the list
min_zero=min(zero)
max_zero=max(zero)
threshold=(max_zero-min_zero)/1.5 #Set the threshold
indices=np.where(np.array(zero)>threshold) #Find the images that have more dark pixels in the region than the calculated threshold
selected_names=np.array(names)[indices] #Selected images

#All the selected CT slices are sored in the 'Filtered_CT' folder
#it sores 141 slices in this case i.e. 54% of the input given. 
for selected_img in selected_names:
    shutil.copy(selected_img,'C:/Users/Abhishek Nagrecha/Desktop/Filtered_CT')


# # 3]Since we had around 12000 CT scans, we splitted it into two folders, and then applied the pre-processing

# In[15]:


# Now since the CT scans are filtered, I will convert then to grayscale mode 'L' for further processing
#So this is for first sub-folder
import cv2
from os import listdir, makedirs
from os.path import isfile,join
from PIL import Image


path = r'C:/Users/Abhishek Nagrecha/Desktop/SEM4/Final_project/Covid-CT-iran/Train&Validation1/' # Source Folder
dstpath = r'C:/Users/Abhishek Nagrecha/Desktop/gray/' # Destination Folder



# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:
#     try:
        img = cv2.imread(os.path.join(path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,gray)
        


# In[16]:


# Now since the CT scans are filtered, I will convert then to grayscale mode 'L' for further processing
#And this is for Second sub-folder
import cv2
from os import listdir, makedirs
from os.path import isfile,join
from PIL import Image


path = r'C:/Users/Abhishek Nagrecha/Desktop/SEM4/Final_project/Covid-CT-iran/Train&Validation2/' # Source Folder
dstpath = r'C:/Users/Abhishek Nagrecha/Desktop/gray/' # Destination Folder



# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:
#     try:
        img = cv2.imread(os.path.join(path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,gray)
        


# # 4]Once the images were filtered and converted into Grayscale, I convert them to png format which reduces the size of CT from 323kb to 54kb in an average

# In[18]:


import cv2, os
base_path = "C:/Users/Abhishek Nagrecha/Desktop/gray/"
new_path = "C:/Users/Abhishek Nagrecha/Desktop/gray_png/"
for infile in os.listdir(base_path):
    print ("file : " + infile)
    read = cv2.imread(base_path + infile)
    outfile = infile.split('.')[0] + '.png'
    cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])


# In[11]:


get_ipython().system('pip install imagecodecs ')


# # 5]Now using the folder 'gray_png'  to resize the image to 256*256

# # Once the image is resized its rize reduces to 12kb from the original raw 323kb

# # Dataset size reduces from 4.11 GB to 154 MB

# In[10]:


import os
import cv2
from os import listdir, makedirs
from os.path import isfile,join
from PIL import Image


path = r'C:/Users/Abhishek Nagrecha/Desktop/gray_png/' # Source Folder
dstpath = r'C:/Users/Abhishek Nagrecha/Desktop/resized_png/' # Destination Folder
width = 256
height = 256
dim = (width, height)


files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:
        img = cv2.imread(os.path.join(path,image))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,resized)
        


# # 6]Morphological Transformations [Dilation]
# 

# # Dataset size reduces from 154 MB to 132 MB
# 

# In[14]:


import os
import cv2
from os import listdir, makedirs
from os.path import isfile,join
from PIL import Image
import numpy as np



path = r'C:/Users/Abhishek Nagrecha/Desktop/resized_png/' # Source Folder
dstpath = r'C:/Users/Abhishek Nagrecha/Desktop/dilation_png/' # Destination Folder



files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:
        img = cv2.imread(os.path.join(path,image))
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,dilation)
        

