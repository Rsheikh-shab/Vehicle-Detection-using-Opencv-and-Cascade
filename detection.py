# Run python detection.py

import cv2
import datetime

cascade = cv2.CascadeClassifier('vehicle.xml')

start_time = datetime.datetime.now()
print(start_time)

def main():
    video = cv2.VideoCapture('cars.mp4')
    
    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            break       
            
        else:
            vehicle = cascade.detectMultiScale(frame, 1.15, 4)
            for (x, y, w, h) in vehicle:
                cv2.rectangle(frame, (x, y), (x+w,y+h), color=(255, 255, 0), thickness=1)
            
            cv2.imshow('Vehicle Detection', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    end_time = datetime.datetime.now()
    print(end_time)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()