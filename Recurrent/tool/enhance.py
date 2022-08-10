import cv2
import numpy as np
 
INPUT_FILE1 = '/mnt/hdd1/chenbeitao/data/datasets/UrbanPipe-Track/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video/4707.mp4'
INPUT_FILE2 = '/mnt/hdd1/chenbeitao/data/datasets/UrbanPipe-Track/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video/4709.mp4'
OUTPUT_FILE = './video/merge_four.mp4'
 
reader1 = cv2.VideoCapture(INPUT_FILE1)
reader2 = cv2.VideoCapture(INPUT_FILE2)
width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
length1 = int(reader1.get(cv2.CAP_PROP_FRAME_COUNT))
length2 = int(reader2.get(cv2.CAP_PROP_FRAME_COUNT))
print(length1, length2)
writer = cv2.VideoWriter(OUTPUT_FILE,
            #   cv2.VideoWriter_fourcc('I', '4', '2', '0'), # (*"mp4v") for mp4 output
              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), # (*"mp4v") for mp4 output
              30, # fps
              (width, height)) # resolution
 
print(reader1.isOpened())
print(reader2.isOpened())
have_more_frame = True
two_flag = True
c = 0

while have_more_frame and two_flag:
    have_more_frame, frame1 = reader1.read()
    two_flag, frame2 = reader2.read()
    if frame1 is None or frame2 is None:
        break
    frame1 = cv2.resize(frame1, (width//2, height//2))
    frame2 = cv2.resize(frame2, (width//2, height//2))
    row1 = np.hstack((frame1, frame2))
    row2 = np.hstack((frame2, frame1))
    img = np.vstack((row1, row2))
    # print(img.shape)
    cv2.waitKey(1)
    writer.write(img)
    c += 1
    print(str(c) + ' is ok', end='\r')
 
 
writer.release()
reader1.release()
reader2.release()
cv2.destroyAllWindows()