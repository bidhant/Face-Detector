import cv2

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("Robert Downey.png")

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#print(face_coordinates)

for (x, y, w, h) in face_coordinates:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 256, 0), 5)

cv2.imshow("Rober Downey Face Detector", img)

cv2.waitKey()





