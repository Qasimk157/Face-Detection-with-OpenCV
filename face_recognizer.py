
import cv2, numpy as np


face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer/trainer.yml');

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color=img[y:y+h,x:x+w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf > 20):

         if(id==12):
            id = 'M-Qasim'

         elif(id==48):
            id = 'Saif ul Rahman'

         elif(id==52):
            id = 'Shahzaib Shair'
         elif(id==92):
            id = 'Allah Wasaya'
         elif(id==84):
            id = 'Ahmad Raza'

         elif(id==95):
            id = 'Shujahat haider'

         elif(id==101):
            id = 'Trump'

         elif(id==102):
            id = 'Imran Khan'

         elif(id==103):
            id = 'General Bawa'

         elif(id==104):
            id = 'Doct Qadeer Khan'

         elif(id==105):
            id = 'Mian Shabaz Sahab'

         elif(id==106):
            id = 'Dr.M.Haroon Yousaf'


         else:
             id = 'Unknown, can not recognize'

             break

        cv2.putText(img,str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));

        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles=smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    cv2.imshow('frame',img);
    #cv2.imshow('gray',gray);

    #if time.time()>start+period:
     #   break;

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
