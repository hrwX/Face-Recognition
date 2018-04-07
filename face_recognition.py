import numpy as np
import os
import math
from matplotlib import pyplot as plt
import cv2
from IPython.display import clear_output
import sys
from tkinter import *

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VideoCamera(object):
    def __init__(self, index = 0):
        self.video = cv2.VideoCapture(index)

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale = False):
        _, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        return frame
    
    def show_frame(self, seconds, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('SnapShot', frame)
        key_pressed = cv2.waitKey(seconds * 1000)
        return key_pressed & 0xFF

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FaceDetector(object):
    def __init__(self, xml_path1, xml_path2):
        self.classifier_face = cv2.CascadeClassifier(xml_path1)
        self.classifier_eye = cv2.CascadeClassifier(xml_path2)

    def detect(self, frame, biggest_only = True):
        scale_factor = 1.2
        min_neighbors = 3
        min_size = (30, 30)
        biggest_only = True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier_face.detectMultiScale(frame, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size, flags = flags)
        return faces_coord
        
    def rotate(self, faces_coord, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Theta = 0
        rows, cols = frame.shape
        eye = self.classifier_eye.detectMultiScale(frame)
        if faces_coord == ():
            print("none")
            return frame
        else:
            for (fx, fy, fw, fh) in faces_coord:
                for (sx, sy, sw, sh) in eye:
                    print("in for")
                    if eye.shape[0] == 2:                                                             
                        if eye[1][0] > eye[0][0]:
                            DY = ((eye[1][1] + eye[1][3] / 2) - (eye[0][1] + eye[0][3] / 2))    # Height diffrence between the eye
                            DX = ((eye[1][0] + eye[1][2] / 2) - eye[0][0] + (eye[0][2] / 2))    # Width difference between the eye
                        else:
                            DY = (-(eye[1][1] + eye[1][3] / 2) + (eye[0][1] + eye[0][3] / 2))
                            DX = (-(eye[1][0] + eye[1][2] / 2) + eye[0][0] + (eye[0][2] / 2))
                        if (DX != 0.0) and (DY != 0.0):                                         # Make sure the the change happens only if there is an angle
                            Theta = math.degrees(math.atan(round(float(DY) / float(DX), 2)))    # Find the Angle
                            print("Theta  " + str(Theta))
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, 1)         # Find the Rotation Matrix
                            frame = cv2.warpAffine(frame, M, (cols, rows)) 
                            print(frame)
                            return frame
                        else:
                            print(frame)
                            return frame
                    else:
                        print(frame)
                        return frame
            
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def normalize_intensity(images):
    images_norm = []
    images_norm.append(cv2.equalizeHist(images))
    return images_norm

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def resize(images, size=(300, 300)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def draw_rectangle(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
    return image

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_images(frame, faces_coord):
    faces_img = normalize_intensity(frame)
    faces_img = resize(faces_img)
    return (frame, faces_img)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_person():
    people_folder = "people/"
    person_name = Entry1.get().lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        os.mkdir(folder)
        video = VideoCamera()
        detector = FaceDetector('haarcascade_frontalface_alt2.xml', 'haarcascade_eye.xml')
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        while counter < 21:
            frame = video.get_frame()
            faces_coord = detector.detect(frame, False)
            frame = detector.rotate(faces_coord, frame)
            if len(faces_coord):
                draw_rectangle(frame, faces_coord)
                frame, face_img = get_images(frame, faces_coord)
                if timer % 100 == 5:
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg', face_img[0])
                    face_img[0] = cv2.flip(face_img[0], 1 )
                    cv2.imwrite(folder + '/' + str(counter) + '-flip.jpg', face_img[0])
                    counter += 1
            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)
            timer += 5
        update_algo(person_name)
    else:
        print ("This name already exists.")
        sys.exit()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def update_algo(person_name):
    people_folder = "people/"

    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()

    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        if person == person_name:
            labels_people[i] = person
            for image in os.listdir(people_folder + person):
                images.append(cv2.imread(people_folder + person + '/' + image, 0))
                labels.append(i)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    try:
        recognizer.update(images, np.array(labels))
        recognizer.save('trainer/trainer.yml')
    except:
        print ("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_algo():
    people_folder = "people/"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()
    
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
            
    try:
        recognizer.train(images, np.array(labels))
        recognizer.save('trainer/trainer.yml')
    except:
        print ("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()        
				
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def recognize_people():
    people_folder = "people/"
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()
    detector = FaceDetector('haarcascade_frontalface_alt2.xml', 'haarcascade_eye.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    threshold = 4000
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.read('trainer/trainer.yml')
    except:
        print ("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()
    
    video = VideoCamera()
    
    while True:
        frame = video.get_frame()       
        faces_coord = detector.detect(frame, False)
        print(faces_coord)
        frame = detector.rotate(faces_coord, frame)
        if len(faces_coord):
            frame, faces_img = get_images(frame, faces_coord)
            for i, face_img in enumerate(faces_img):
                collector = cv2.face.StandardCollector_create()
                recognizer.predict_collect(face_img, collector)
                conf = collector.getMinDist()
                pred = collector.getMinLabel()
                print ("Prediction: " + labels_people[pred].capitalize())
                print ('Confidence: ' + str(round(conf)))
                print ('Threshold: ' + str(threshold))
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(), (faces_coord[i][0], faces_coord[i][1] - 2), cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Unknown", (faces_coord[i][0], faces_coord[i][1]), cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2, cv2.LINE_AA)
            draw_rectangle(frame, faces_coord)
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            sys.exit()

    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

root = Tk()

Label1 = Label(root, text="Name")
Entry1 = Entry(root)
Button1 = Button(root, text ="Add Person", command = add_person)
Button2 = Button(root, text ="Train algo", command = train_algo)
Button3 = Button(root, text ="Recognize Person", command = recognize_people)

Label1.grid(row=0,column=0)
Entry1.grid(row=0, column=2)
Button1.grid(row=1, column=0)
Button2.grid(row=1, column=1)
Button3.grid(row=1, column=2)

root.mainloop()
