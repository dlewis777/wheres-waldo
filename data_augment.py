import cv2
import os
import sys
# only argument is directory of data

wkdir = sys.argv[1]
imsize = int(sys.argv[2]) # 256, 128, 64

for filename in  os.listdir(wkdir + '/' + str(imsize) + '/waldo'):
    
    im = cv2.imread(wkdir + '/' + str(imsize) + '/waldo/' + filename)
    #up/down
    for i in range(imsize):
        #left/right
        for j in range(imsize):
            print('OG SHAPE', im.shape)
            temp = cv2.copyMakeBorder(im,0,i,0,j,cv2.BORDER_WRAP)
            print(temp.shape)
            temp2 = temp[i:imsize+i,j:imsize+j]
            print(temp2.shape)
            cv2.imwrite(wkdir + '/augment/waldo/' + filename + '-' + str(i) + '-' + str(j) + '.jpg', temp2)	
            temp3 = cv2.flip(temp2,1)
            cv2.imwrite(wkdir + '/augment/waldo/' + filename + '-' + str(i) + '-' + str(j) + 'r.jpg', temp3)	
