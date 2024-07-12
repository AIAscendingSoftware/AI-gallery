'''# Use the encoding for face recognition tasks...
from mtcnn.mtcnn import MTCNN
import cv2,torch
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize MTCNN and InceptionResnetV1 models
mtcnn = MTCNN()  # Initialize without 'keep_all' argument
from facenet_pytorch import InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) #argface, insightface
     
def embedding(face, image_rgb):
    box = face['box']
    box = [int(v) for v in box]
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height
    face_image = image_rgb[y1:y2, x1:x2]
    # Use FaceNet for embedding
    face_image = cv2.resize(face_image, (160, 160))
    face_image = face_image.astype(np.float32) / 255.0
    face_tensor = torch.Tensor(face_image).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        face_embedding = resnet(face_tensor).cpu().numpy().flatten()
    # plt.imshow(face_image)
    # plt.axis('off')
    # plt.show()
    return face_embedding, face_image
  
def user_detect_faces(image):

      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      faces = mtcnn.detect_faces(image_rgb)  # Use detect_faces method
      # If faces are detected, encode them
      if faces:          
          encoding_faces_from_one_image=[]
          for face in faces:
  
              face_embedding, face_image=embedding(face, image_rgb)
              # Print or store the face embedding
              encoding_faces_from_one_image.append(face_embedding)
              # # Display the face image
              plt.imshow(face_image)
              plt.axis('off')
              plt.show()
          return encoding_faces_from_one_image

def similarity_score(user_encoding , event_encoding, metric = "cosine"):
  
    # For Euclidean distance, lower thresholds indicate more similar faces.
    # A common starting point is 0.5 or 1.0, but you might need to adjust based on your data.
    if metric == "euclidean":
        similarity_score = np.linalg.norm(user_encoding - event_encoding, axis=0)

    # For cosine similarity,  aces.
    # A common starting point is 0.7 or 0.8, but you might need to adjust based on your data.
    elif metric == "cosine":
        similarity_score = np.dot(user_encoding, event_encoding) / (
            np.linalg.norm(user_encoding) * np.linalg.norm(event_encoding)
        )

    else:
        raise ValueError(f"Invalid metric: {metric}")

    return similarity_score

def event_detect_faces(image, user_face_encoded, one_event_data):
      
      
      # print(user_face_encoded)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      faces = mtcnn.detect_faces(image_rgb)  # Use detect_faces method
      
      # If faces are detected, encode them
      if faces:
          
          for face in faces:
              event_face_embedding, face_image=embedding(face, image_rgb)
            #   print(face_embedding)
              
              total_score=[]
              
              for user_face in user_face_encoded:
                  # print(user_face)
                  user_encoding = user_face["encoding"]
                  # print(user_encoding)0000
                  # # Print or store the face embedding
                  similare_score=similarity_score(user_encoding, np.array(event_face_embedding))
                #   print(similare_score)
                  total_score.append(similare_score)

              data=np.array(total_score) #[0.88676, 0.467657, 0.446565]
              cumulative_score=sum(data)

              #conditions based on the cosine similarity 
              secnario_1=((np.any(data >= 0.7))  and (cumulative_score >= 1.7))
              secnario_2=((np.any(data >= 0.6)) and (cumulative_score >= 2))

              if secnario_1 or secnario_2:
                # print((f'total score:{total_score},cumulative score:{cumulative_score}'))
                plt.imshow(face_image)
                plt.axis('off')
                plt.title(f'total score:{total_score},cumulative score:{cumulative_score}')
                plt.show()
                new_entry = {"imagePath": one_event_data['eventFolderPath'],
                             'imageName': one_event_data['ImageName'],
                             'orgId': one_event_data['OrgId'],
                             'userId': user_face['UserId'],
                             'eventId': one_event_data['EventId']} 
                return new_entry
      
              '''

from mtcnn.mtcnn import MTCNN
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1

class FaceDetection:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embedding(self, face, image_rgb):
        box = face['box']
        box = [int(v) for v in box]
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height
        face_image = image_rgb[y1:y2, x1:x2]
        face_image = cv2.resize(face_image, (160, 160))
        face_image = face_image.astype(np.float32) / 255.0
        face_tensor = torch.Tensor(face_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            face_embedding = self.resnet(face_tensor).cpu().numpy().flatten()
        return face_embedding, face_image

    def user_detect_faces(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.mtcnn.detect_faces(image_rgb)
        if faces:
            encoding_faces_from_one_image = []
            for face in faces:
                face_embedding, face_image = self.embedding(face, image_rgb)
                encoding_faces_from_one_image.append(face_embedding)
            return encoding_faces_from_one_image

    def similarity_score(self, user_encoding, event_encoding, metric="cosine"):
        if metric == "euclidean":
            return np.linalg.norm(user_encoding - event_encoding, axis=0)
        elif metric == "cosine":
            return np.dot(user_encoding, event_encoding) / (np.linalg.norm(user_encoding) * np.linalg.norm(event_encoding))
        else:
            raise ValueError(f"Invalid metric: {metric}")

    def event_detect_faces(self, event_image, user_face_encoded, one_event_data):
        image_rgb = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)
        faces = self.mtcnn.detect_faces(image_rgb)
        if faces:
            for face in faces:
                event_face_embedding, face_image = self.embedding(face, image_rgb)
                total_score = [self.similarity_score(user_face["encoding"], np.array(event_face_embedding)) for user_face in user_face_encoded]
                data = np.array(total_score)
                cumulative_score = sum(data)
                secnario_1 = (np.any(data >= 0.7) and cumulative_score >= 1.7)
                secnario_2 = (np.any(data >= 0.6) and cumulative_score >= 2)
                if secnario_1 or secnario_2:
                    new_entry = {
                        "imagePath": one_event_data['eventFolderPath'],
                        'imageName': one_event_data['ImageName'],
                        'orgId': one_event_data['OrgId'],
                        'userId': user_face_encoded[0]['UserId'],
                        'eventId': one_event_data['EventId']
                    }
                    return new_entry
