import numpy as np
import face_recognition 
import os
from typing import List, Union


class Face:
    def __init__(self, encodings:Union[List[int], np.array], name:str):
        self.encodings = encodings
        self.name = str(name.split('.')[0]).title()

    def __repr__(self):
        return f"<Face {self.name}>"


class KnownFaceEncodings:
    """
    KnownFaceEncondings class will help in managing the encodings of the Known faces
    """
    # Constants / static members
    DATA_FOLDER = './data'
    DATA_FILE = 'known-faces.npy'


    def __init__(self, new_image_added:bool = False, face_folder:str = 'people'):
        images = []  # for temporary storage
        KNOWN_FACE_FOLDER = self.DATA_FOLDER + '/' + face_folder + '/'
        
        if (self.DATA_FILE in os.listdir(self.DATA_FOLDER)) and not new_image_added:  
            # read the data if no new image added or removed
            self.data = np.load(self.DATA_FOLDER + self.DATA_FILE, allow_pickle=True) 

        else:
            face_images = os.listdir(KNOWN_FACE_FOLDER)  # Get every face image names
            
            for image_name in face_images:  

                encodings = self.get_face_encoding(KNOWN_FACE_FOLDER + image_name)
                images.append(Face(encodings, image_name))
                
            self.data = np.array(images)

            np.save(self.DATA_FOLDER + self.DATA_FILE, self.data)

    def get_face_encoding(path:str) -> np.array:
        '''Get face encoding of the given path'''
        image = face_recognition.load_image_file(path)    
        return face_recognition.face_encodings(image)[0]  # return encoded face

    @property
    def encodings(self):
        """The numpy.array of all the encodings"""
        return np.array(list(map(lambda face: face.encodings, self.data)))

    @property
    def names(self):
        """The numpy.array of all the names"""
        return np.array(list(map(lambda face: face.name, self.data)))
