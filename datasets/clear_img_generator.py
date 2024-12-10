import os
import cv2

if __name__ == '__main__':
    path = '/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/20211021/数据'
    i = 0
    j = 0
    scenes = os.listdir(path)
    for scene in scenes:
        video_path = os.path.join(path, scene, '工业相机')
        file_name = os.listdir(video_path)
        if len(file_name) > 0:
            cap = cv2.VideoCapture(os.path.join(video_path, file_name[0]))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('', frame)
                key = cv2.waitKey(0)
                if key == ord('5'):
                    continue
                elif key == ord('c'):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/sampled/clear_rgb/'+str(i)+'.jpg', frame)
                    i += 1
                elif key == ord('b'):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('/media/ywqqqqqq/YWQ/Dataset/VidarCity/Real/VidarCity/dataset/sampled/blurred_rgb/'+str(j)+'.jpg', frame)
                    j += 1
            cap.release()