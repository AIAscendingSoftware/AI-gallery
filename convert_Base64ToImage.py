'''import base64, cv2
import numpy as np

def convert_Base64ToImgae(base64_string):

    # # Decode the base64 string into bytes
    image_byte_data = base64.b64decode(base64_string)
  
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(image_byte_data, np.uint8)
    # print(len(nparr))
    if len(nparr) != 0 :
        # Decode the numpy array as an image
        try:
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)  # Wait for a key press before closing the window
            # cv2.destroyAllWindows()
            return image
        except cv2.error as e:
            # print(f"Error decoding image: {e}")
            return None

'''

import base64
import cv2
import numpy as np

class ImageConverter:
    @staticmethod
    def convert(base64_string):
        image_byte_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_byte_data, np.uint8)
        if len(nparr) != 0:
            try:
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except cv2.error as e:
                return None
