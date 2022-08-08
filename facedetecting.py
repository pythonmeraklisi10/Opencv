import cv2,os
cas = cv2.CascadeClassifier("./yapay zeka/face.xml")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = cas.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.line(img,(x+w//2,y),(x+w//2,y-50),(0,0,255),3)
        cv2.line(img,(x,y+h//2),(x-50,y+h//2),(0,0,255),3)
        cv2.line(img,(x+w//2,y+h),(x+w//2,y+h+50),(0,0,255),3)
        cv2.line(img,(x+w,y+h//2),(x+w+50,y+h//2),(0,0,255),3)
        os.system("cls")
        print(f"""
{(x,y)}-----{(x+w,y)}
|                    |
|     Human Face     |
|                    |
{(x,y+h)}-----{(x+w,y+h)}""")
    cv2.imshow("Kamera",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
