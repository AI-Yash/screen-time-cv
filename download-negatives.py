import os
import urllib.request 
import numpy as np
import cv2
import random

ENDPOINT1 = 'https://source.unsplash.com/featured'  # unsplash api  
# if we are getting many duplicates, the number of queries is to be increased (should be ok then)
# queries are the search query by which the image is identified
queries = [
    'people', 
    'furniture', 
    'nature', 
    'park', 
    r'group%20of%20people', 
    'garden', 
    'carpet', 
    'lawn', 
    'street', 
    'road', 
    'home', 
    'bed', 
    r'corner%20inside'
]  # queries which help grab the image


def preprocess_image(image_path:str, image_size=100):
    '''preprocesses the given image'''
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (image_size, image_size))
    cv2.imwrite(image_path, resized_image)


def download_random_image(dest:str):
    '''Downloads a random image using ENDPOINT1 and a random query'''
    url = f"{ENDPOINT1}?{random.choice(queries)}"  # generate the api url

    urllib.request.urlretrieve(url, dest)  # download the images


def store_raw_images(images_num:int = 3000, data_folder:str = 'data/negatives'):
    '''Downloads `images_num` images from the web and stores it in `data_folder`'''
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)  # check if the forlder exists, if not create it
        
    for i in range(images_num):
        try:
            print(f"Downloading {str(i).rjust(4, ' ')}/{images_num} image...", end='\r')  # be verbose
            
            download_random_image(f"{data_folder}/{i}.jpg")
        
            # preprocess the images
            preprocess_image(f"{data_folder}/{i}.jpg")
            
        except Exception as e:
            print(str(e))  # if some exception occured

    print(f'Completed downloading {images_num} images check if the numbers are downloaded in {data_folder}')  

if __name__ == '__main__':
    store_raw_images(2000, 'data/neg')
