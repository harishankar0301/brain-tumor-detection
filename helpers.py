
import numpy as np

import cv2



from skimage.feature import *


#from google.colab.patches import cv2_imshow
kernel = np.ones((7,7),np.uint8)

class Segmentation(object):

    def __init__(self, Image):
        self.Image = Image
        
    def binarization(self):
        ret, thresh1 = cv2.threshold(self.Image,30,255,cv2.THRESH_BINARY)
        # convert to white
        closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

        # save to /tmp folder
        # cv2.imwrite("imgs/thresh1.jpg", thresh1)
        # cv2.imwrite("imgs/closing1.jpg", closing1)
        
        return closing1, thresh1

    # removing skul
    def removing_skul(self, closing1):
        erosion = cv2.erode(closing1,kernel,iterations = 6)
        NO_skull = self.Image * (-erosion)

        # save to /tmp folder
        # cv2.imwrite("imgs/NO_skull.jpg", NO_skull)
        # cv2.imwrite("imgs/erosion.jpg", erosion)
        
        return NO_skull, erosion
    
    # enhance image to segmentation
    def enhance_image_t_seg(self, NO_skull):
        median = cv2.medianBlur(NO_skull,5)
        blur = cv2.GaussianBlur(median,(5,5),0)

        # save to /tmp folder
        #cv2.imwrite("imgs/blur.jpg", blur)
        
        return blur
    
    def to_Segmentation(self, blur):
        ret, thresh2 = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
        img = thresh2
        # Remove noise
        no_noise = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # save to /tmp folder
        # cv2.imwrite("imgs/no_noise.jpg", no_noise)
        # cv2.imwrite("imgs/thresh2.jpg", thresh2)
       
        return no_noise, thresh2
    
    # edge detection
    def Edge_Detection(self,thresh2):
        edges = cv2.Canny(thresh2,100,200)

        # save to /tmp folder
        #cv2.imwrite("imgs/edges.jpg", edges)
        
        return edges

class Preprocessing(object):
    grayImage   = ''
    image       = ''
    closing1    = ''
    thresh1     = ''
    NO_skull    = ''
    erosion     = ''
    blur        = ''
    no_noise    = ''
    edge        = ''
    tumourImage = ''
    
    def preproces(self, originalImageUrl):
        self.url = originalImageUrl
        image = cv2.imread(originalImageUrl)
        self.grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.image = Segmentation(self.grayImage)

    # binarization
    def binarization(self):
        self.closing1, self.thresh1 = self.image.binarization()
        #plt.imshow(closing1,'gray')

    # removing_skul
    def removingSkul(self):
        self.NO_skull, self.erosion = self.image.removing_skul(self.closing1)
        #plt.imshow(NO_skull,'gray')

    # enhance image to segmentation
    def enhanceImage(self):
        self.blur = self.image.enhance_image_t_seg(self.NO_skull)
        #plt.imshow(blur,'gray')

    # Segmentation
    def segmentation(self):
        self.no_noise, self.thresh2 = self.image.to_Segmentation(self.blur)
        #plt.imshow(no_noise,'gray')

        self.edge = self.image.Edge_Detection(self.thresh2)

        # Plot
        #segm.Show_plots(img,thresh1,closing1,erosion,blur,NO_skull,no_noise)

    def getInfectedRegion(self):
        col1 = 0
        col2 = 0
        row1 = 0
        row2 = 0
        
        for i in range(self.no_noise.shape[1]):
            for j in range(self.no_noise.shape[0]):
                if self.no_noise.item(j, i) > 0:
                    if col1 == 0 & col2 == 0:
                        col1 = j
                        row1 = i
                    else:
                        if col1 > j:
                            col1 = j
                        else:
                            if col2 < j:
                                col2 = j
                            
                        if row1 > i:
                            row1 = i
                        else:
                            if row2 < i:
                                row2 = i

        # draw rectangle to select tumour region                    
        cv2.rectangle(self.no_noise, (row1, col1), (row2, col2), (255,0,0), 2)

        # save again for good preview 
#         cv2.imwrite("./tmp/no_noise.jpg", self.no_noise)

        # tumourImage
        self.tumourImage = self.no_noise[col1:col2, row1:row2]
    
        
        #plt.imshow(self.tumourImage, "gray")
        if col1==0 or col2 ==0 or row1==0 or row2==0:
            pass
        
        #bprint(col1, col2, row1, row2)
        if self.tumourImage is not None and (row2-row1)>0 and (col2-col1)>0:
            cv2.imwrite(str("tmp/" + self.url.split('/')[-1]), self.tumourImage)

        
        
        return col1, col2, row1, row2
