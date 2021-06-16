'''
import cv2
import numpy as np

def diffimage(lastframe, nextframe):
    diff_frame = nextframe - lastframe
    ABS = abs(diff_frame)
    diff_value = (ABS.sum(axis = 0)).sum(axis = 0)
    return diff_frame, diff_value

if __name__ == '__main__':
    cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    ret, lastframe = cap.read()
    lastgray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
    lastgray = cv2.blur(lastgray, (5,5))
    ret, nextframe = cap.read()
    nextgray = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
    nextgray = cv2.blur(nextgray, (5,5))
    index = 1
    last_diff_value = 0
    while(ret):
        index += 1
        diff_frame, next_diff_value = diffimage(lastgray, nextgray)
        cv2.imshow('difffame', diff_frame)
        if index == 251:
        	print(next_diff_value/nextgray.size)
        if (next_diff_value/nextgray.size > 150):
        	cv2.imwrite('Frame_%04d.png'%index,nextframe)
        #if (last_diff_value*3 < next_diff_value):
        #    cv2.imwrite('Frame_%04d.png'%index,nextframe)

        k = cv2.waitKey(15)
        if k == 27:
            break
        lastgray = nextgray
        last_diff_value = next_diff_value
        ret, nextframe = cap.read()
        if (ret):
            nextgray = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
            nextgray = cv2.blur(nextgray, (5,5))
    cap.realease()
    cv2.destroyAllWindows()
'''

import cv2
import numpy as np

# la difference entre frame derniere et frame actuelle
def diffimage(lastframe, nextframe):
    diff_frame = nextframe - lastframe 
    ABS = abs(diff_frame)
    diff_value = np.sum(ABS) # 
    return diff_frame, diff_value

if __name__ == '__main__':
    cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    ret, lastframe = cap.read()
    lasthsv = cv2.cvtColor(lastframe, cv2.COLOR_BGR2HSV)
    index = 1

    while(ret):
        index += 1
        ret, nextframe = cap.read()
        nexthsv = cv2.cvtColor(nextframe, cv2.COLOR_BGR2HSV)
        diff_frame, diff_value = diffimage(lasthsv, nexthsv)
        cv2.imshow('frame', nexthsv)

        if (diff_value/nextframe.size > 120):
            cv2.imwrite('Frame_%04d.png'%index,nextframe)
            cv2.imwrite('Frame_nexthsv%04d.png'%index,nexthsv)

        lastframe = nextframe
        lasthsv = nexthsv
        k = cv2.waitKey(5)
        if k == 27:
            break
        
    cap.realease()
    cv2.destroyAllWindows()



 

