import numpy as np
import os
import math
from matplotlib import pyplot as plt
import cv2

from IPython.display import clear_output
import sys

class VideoCamera(object):
    def __init__(self, index = 0):
        self.video = cv2.VideoCapture(index)

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale = False):
        _, frame = self.video.read()
        #print (frame)
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

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, frame, biggest_only = True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(frame, scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size, flags = flags)
        return faces_coord

def normalize_intensity(images):
    images_norm = []
    is_color = len(images.shape) == 3
    if is_color:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    images_norm.append(cv2.equalizeHist(images))
    return images_norm

def resize(images, size=(100, 100)):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

def draw_rectangle(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
    return image

def get_images(frame, faces_coord):
    faces_img = normalize_intensity(frame)
    faces_img = resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder):
    print ("add_person")
    person_name = input('What is the name of the new person: ').lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        input("20 pictures will be taken. Press ENTER when ready.")
        os.mkdir(folder)
        video = VideoCamera()
        detector = FaceDetector('haarcascade_frontalface_default.xml')
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        while counter < 21:
            frame = video.get_frame()
            face_coord = detector.detect(frame)
            if len(face_coord):
                draw_rectangle(frame, face_coord)
                frame, face_img = get_images(frame, face_coord)
                if timer % 100 == 5:
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg', face_img[0])
                    print ('Images Saved:' + str(counter))
                    counter += 1
                    cv2.imshow('Saved Face', face_img[0])
            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)
            timer += 5
	
	people = [person for person in os.listdir(people_folder)]
	images = []
	labels = []
	labels_people = {}
	for i, person in enumerate(people):
		labels_people[i] = person
		for image in os.listdir(people_folder + person):
			images.append(cv2.imread(people_folder + person + '/' + image, 0))
			labels.append(i)
	recognizer.train(images, np.array(labels))
    	recognizer.save('trainner/trainner.yml')
	
    else:
        print ("This name already exists.")
        sys.exit()
				
def recognize_people(people_folder):
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()
    print ("This are the people in the Recognition System:")
    for person in people:
        print ("-" + person)
    print (30 * '-')
    print ("EigenFaces Algorithm")
    print (30 * '-')
    detector = FaceDetector('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.EigenFaceRecognizer_create()
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
        recognizer.read('trainner/trainner.yml')#images, np.array(labels))
        #recognizer.save('trainner/trainner.yml')
    except:
        print ("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()
    video = VideoCamera()
    
    while True:
        frame = video.get_frame()
        faces_coord = detector.detect(frame, False)
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

if __name__ == '__main__':
	print (30 * '-')
	print ("1. Add person to the recognizer system")
    	print ("2. Start recognizer")
    	print ("3. Exit")
    	print (30 * '-')
    	PEOPLE_FOLDER = "people/"
    	CHOICE = int(input('Enter your choice [1-3] : '))
    	if CHOICE == 1:
        	add_person(PEOPLE_FOLDER)
    	if CHOICE == 2:
        	recognize_people(PEOPLE_FOLDER)
    	if CHOICE == 3:
        	sys.exit()
