import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 35)
    smile = smile_cascade_db.detectMultiScale(img_gray, 1.1, 50)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
    if smile.any():
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Face', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
# img = cv2.imread("gwl1S-Be6tU.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade_db.detectMultiScale(img, 1.1, 19)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
#
# cv2.imshow('rez', img)
# cv2.waitKey()
