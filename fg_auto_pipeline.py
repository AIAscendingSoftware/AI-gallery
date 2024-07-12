'''

import numpy as np
import os,cv2
from face_detedtion import event_detect_faces,user_detect_faces
from convert_Base64ToImage import convert_Base64ToImgae

def user_encode_faces(one_user_data):
    user_encoded_face = []

    for image_base64 in one_user_data["image"]:
        oneimage = convert_Base64ToImgae(image_base64)
        
        encodings = user_detect_faces(oneimage)
        # print('event encoding',len(encodings))
        if encodings:
            for encoding in encodings:
                encoded_face = {
                    "encoding": np.array(encoding),
                    "UserId" : one_user_data["UserId"]
                }
                user_encoded_face.append(encoded_face)
                # print(encoded_face)
    return user_encoded_face

def both_data(event_data, user_data):
    # print('no. of user_data', len(user_data))

    similar_faces = []
    for one_user_data in user_data:   
        user_face_encoded = user_encode_faces(one_user_data)

        for one_event_data in event_data:
            image_base64=one_event_data["image"]
            event_image = convert_Base64ToImgae(image_base64)
            # print(event_image, len(user_face_encoded))
            new_entry=event_detect_faces(event_image, user_face_encoded, one_event_data)
            if new_entry is not None:
                similar_faces.append(new_entry)
    return similar_faces

def separate_unique_data(similar_tour_faces):
    unique_data = {}
    for item in similar_tour_faces:
      # Create a dictionary key using a unique combination of values from the data
      key = (item['imagePath'], item['imageName'], item['orgId'], item['userId'], item['eventId'])
      if key not in unique_data:
        unique_data[key] = item
    unique_data_list = list(unique_data.values())
    return unique_data_list




'''
import numpy as np
import cv2
from face_detection import FaceDetection
from convert_Base64ToImage import ImageConverter

class FaceRecognitionPipeline:
    def __init__(self):
        self.image_converter = ImageConverter()
        self.face_detection = FaceDetection()

    def user_encode_faces(self, one_user_data):
        user_encoded_face = []

        for image_base64 in one_user_data["image"]:
            one_image = self.image_converter.convert(image_base64)
            encodings = self.face_detection.user_detect_faces(one_image)

            '''Here we have to add a filter to go when there are only 3 faces, 
            if there are more than three faces, we should terminate'''
            
            if encodings:
                for encoding in encodings:
                    encoded_face = {
                        "encoding": np.array(encoding),
                        "UserId": one_user_data["UserId"]
                    }
                    user_encoded_face.append(encoded_face)
        return user_encoded_face

    def both_data(self, event_data, user_data):
        similar_faces = []
        for one_user_data in user_data:
            user_face_encoded = self.user_encode_faces(one_user_data)
            for one_event_data in event_data:
                image_base64 = one_event_data["image"]
                event_image = self.image_converter.convert(image_base64)
                new_entry = self.face_detection.event_detect_faces(event_image, user_face_encoded, one_event_data)
                if new_entry is not None:
                    similar_faces.append(new_entry)
        return similar_faces

    def separate_unique_data(self, similar_tour_faces):
        unique_data = {}
        for item in similar_tour_faces:
            key = (item['imagePath'], item['imageName'], item['orgId'], item['userId'], item['eventId'])
            if key not in unique_data:
                unique_data[key] = item
        return list(unique_data.values())
