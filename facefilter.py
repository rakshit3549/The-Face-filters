import cv2
import math
import dlib
from datetime import datetime

record_flag = False
new_path = __file__.rsplit("/", 1)[0] # Getting file path

cap = cv2.VideoCapture(0)

if_record = input("Do you want to video record your experience (Y,N)").upper()
print("Press 'ESC' to exit. \nPress 'SPACE' to save an image.")

if if_record == "Y":
    print("Video has been started recording")
    # Gitting the location to save the image
    record_flag = True
    time_stamp = datetime.now().strftime("%d-%M-%Y_%I'%M'%S")
    video_name = "piggy-nose-video_{}.avi".format(time_stamp)
    video_path = new_path + "/video_records/" + video_name

    # Setting Video Encoding
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    rec = cv2.VideoWriter(video_path, fourcc, 12, (640, 480))

else:
    print("Recording had NOT been started")

# Getting the nose filter
nose_image = cv2.imread("nose_image/pig_nose.png")
(h, w) = nose_image.shape[:2]
(cX, cY) = (w // 2, h // 2) # Finding the center of the image for rotation.

# for face deduction 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks/shape_predictor_68_face_landmarks.dat")

while True:
    _, frameunflip = cap.read()
    frame = cv2.flip(frameunflip, 1) # Fliping to get mirror image

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_frame)

    for face in faces:
        # Provides 68 points on face for every face in the image.
        # Point image in landmarks_image/landmarks_68_points.png
        landmarks = predictor(gray_frame, face)

        # Head center line
        top_center = (landmarks.part(27).x, landmarks.part(27).y)
        bottom_center = (landmarks.part(8).x, landmarks.part(8).y)

        angle = -(math.degrees(math.atan2(bottom_center[1] - top_center[1],
                                        bottom_center[0] - top_center[0]))-90)

        # Head landmark
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(math.hypot(left_nose[0] - right_nose[0],
                               left_nose[1] - right_nose[1]) * 2.5)

        nose_height = int(nose_width * 0.72)

        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                    int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                        int(center_nose[1] + nose_height / 2))

        # Adding the new nose
        M = cv2.getRotationMatrix2D((cX, cY), angle, 0.7)
        img_rotated = cv2.warpAffine(nose_image, M, (w, h))
        nose_pig = cv2.resize(img_rotated, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]:top_left[1] + nose_height,
                          top_left[0]:top_left[0] + nose_width]
        try:
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]:top_left[1] + nose_height,
            top_left[0]:top_left[0] + nose_width] = final_nose

        except:
            print("Nose area not in frame")
            pass

    key = cv2.waitKey(1)
    if key == 32:
        # Enters when pressed space bar for clicking image.
        time_stamp = datetime.now().strftime("%d-%M-%Y_%I'%M'%S")
        image_name = "piggy-nose-image_{}.png".format(time_stamp)
        image_path = new_path + "/image_records/" + image_name
        cv2.imwrite(image_path,frame)
        print("Image {} has been saved".format(image_name))
    elif key == 27:
        break

    if record_flag:
        rec.write(frame)
    cv2.imshow("Frame",frame)

if record_flag:
    rec.release()
cap.release()
cv2.destroyAllWindows()
