import cv2
import numpy as np
'''
This function can cut the shots of the video.
adresse : path of video
index : frame index
fps : frequency of frame display. In default fps = 30
'''
def PlanCapture(adresse, index, fps=30):
    cap = cv2.VideoCapture(adresse)
    lens = len(index)
    i = 0
    for l in range(lens):
        if l >= (lens - 1):
            break
        ret,frame = cap.read()
        fps = fps
        size = (frame.shape[1], frame.shape[0]) # video size : (width, height)
        videoWriter =cv2.VideoWriter('video%04d.m4v'%(l+1),cv2.VideoWriter_fourcc('M','J','P','G'),fps,size) # define the video type
        j = index[l]
        k = index[l+1]
        while ret:
            ret,frame = cap.read()
            i += 1
            if(i >= j and i<k):
                videoWriter.write(frame) 
            else: 
                print("next plan")
                break
    cap.realease()

if __name__ == '__main__':
    index = np.array([0, 50,100]) # for example, here we take the planes between the frames (0, 50), (50, 100)
    PlanCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v", index)


