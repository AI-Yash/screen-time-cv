import os
import pathlib
import cv2


def get_images_path(neg_dir:str = 'data'):
    '''
    yields the path of files in the given directory
    '''
    for image in os.listdir(neg_dir):
        image_path = f"{neg_dir}/{image}\n"
        yield image_path
        

def create_neg_txt(neg_dir:str = 'data/negatives'):
    '''Makes bg.txt'''
    with open('bg.txt','w') as file:
        file.writelines(get_images_path(neg_dir))


def rename_by_index(new_start_index:int = 1, data_dir:str='data', token:str='old'):
    '''renames the files which have token in their file names'''
    os.chdir(data_dir)
    print(os.getcwd())

    for file in os.listdir():
        if token in file:
            os.rename(file, f"{new_start_index}{pathlib.Path(file).suffix}")
            new_start_index += 1


def preprocess_image(image_path:str, image_size=50):
    '''preprocesses the given image'''
    # print(pathlib.Path(image_path).exists())
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (image_size, image_size))
    cv2.imwrite(image_path, resized_image)


def batch_process_images(data_dir:int='data'):
    '''batch processes the images in the `data_dir`'''
    for image_path in get_images_path(data_dir):
        preprocess_image(image_path.replace('\n', ''))


def create_info_file(data_dir:str='data', append_val:str='1 0 0 50 50'):
    '''creates a info.lst file'''
    for file_name in get_images_path(data_dir):
        info = file_name.replace('\n', ' ') + append_val + '\n'

        with open('info.lst', 'a') as info_file:
            info_file.write(info)


if __name__ == '__main__':
    # create_neg_txt()
    # rename_by_index(2000, 'data/negatives')   # rename files
    # rename_by_index(1, 'data/positives', 'cat')   # rename files
    # batch_process_images('data/positives')
    create_info_file('data/positives')

# for opencv-dev
# git clone https://github.com/Itseez/opencv.git
# sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
# sudo apt-get install libopencv-dev
