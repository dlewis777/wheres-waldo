import cv2
import os
import sys
# only argument is directory of data

wkdir = sys.argv[1]
imsize = sys.argv[2] # 256, 128, 64

for filename in  os.listdir(wkdir + '/' + imsize + '/waldo'):
    
    im = cv2.imread(wkdir + '/' + imsize + '/waldo/' + filename)
    #up/down
    for i in range(imsize):
        #left/right
        for j in range(imsize):

            temp = cv2.copyMakeBorder(im,0,i,0,j,cv2.BORDER_WRAP)
            temp2 = temp[j:imsize+j+1,i:imsize+i+1]

            cv2.imwrite(filename + '-' + i + '-' + j, temp2)

