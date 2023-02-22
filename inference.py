from retinaface import RetinaFace
from deepface import DeepFace
import os
import cv2
import glob

def image_is_unknown(facial_img_path):
    for faceFold in os.listdir("/Users/niloofar/Documents/Projects/FaceDetection/dataset/"):
        for imagePath in glob.glob("/Users/niloofar/Documents/Projects/FaceDetection/dataset/"+faceFold+"/*.jpg"):    
            obj = DeepFace.verify(facial_img_path, imagePath, model_name = 'ArcFace', detector_backend = 'retinaface',enforce_detection = False)
            if(obj['distance'] < 0.5):
                img = cv2.imread(facial_img_path)
                count = len(os.listdir("/Users/niloofar/Documents/Projects/FaceDetection/dataset/"+faceFold))
                cv2.imwrite("/Users/niloofar/Documents/Projects/FaceDetection/dataset/"+faceFold+"/"+faceFold+str(count+1)+".jpg",img)
            print(obj['distance'])
    return True

img_path = "/Users/niloofar/Documents/Projects/FaceDetection/img2.jpg"
unknown_path = "/Users/niloofar/Documents/Projects/FaceDetection/dataset/unknown"

faces = RetinaFace.detect_faces(img_path)
img = cv2.imread(img_path)
for face in faces: 
    identity = faces[face]
    facial_area = identity["facial_area"]
    landmarks = identity["landmarks"] 
    cv2.rectangle(img, (facial_area[2], facial_area[3])
    , (facial_area[0], facial_area[1]), (255, 255, 255), 1)
    
    facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
    cv2.imwrite('/Users/niloofar/Documents/Projects/tmp.jpg',facial_img)
    if(image_is_unknown('/Users/niloofar/Documents/Projects/tmp.jpg')):
        count = len(os.listdir(unknown_path))
        cv2.imwrite(unknown_path+'/unknown'+str(count+1)+'.jpg',facial_img)

    cv2.circle(img, (int(landmarks["left_eye"][0]),int(landmarks["left_eye"][1])), 1, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks["right_eye"][0]),int(landmarks["right_eye"][1])), 1, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks["nose"][0]),int(landmarks["nose"][1])), 1, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks["mouth_left"][0]),int(landmarks["mouth_left"][1])), 1, (0, 0, 255), -1)
    cv2.circle(img, (int(landmarks["mouth_right"][0]),int(landmarks["mouth_right"][1])), 1, (0, 0, 255), -1)
    cv2.imshow('window_name', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# import matplotlib.pyplot as plt
# faces = RetinaFace.extract_faces(img_path = "/Users/niloofar/Documents/Projects/FaceDetection/img1.jpg", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()