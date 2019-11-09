import numpy as np
import os
import h5py
from PIL import Image


def im2double(image):
    return image.astype(np.float32) / np.iinfo(image.dtype).max

def image_mod_crop(image, scale=3):
    size = image.shape[0:2] #extract height and width size only
    size -= np.mod(size, scale) #subtract the size by the remainder between the division of size and scale (to make sure image are divisible by the scale)
    return image[0:size[0], 0:size[1]] #extract the image with size divisable by the scale


def create_dataset(image_dir, output_dir, input_size=33, label_size=21, stride=14, scale=3):
    data = []
    label = []
    counter = 0

    #loop through the image folder
    for _, _, files in os.walk(image_dir):
        #loop thourgh the image files
        for img_file in files:
            #read image and convert it to ndarray
            img = Image.open(image_dir+img_file)
            img = np.array(img)
    
            #create groundtruth image that divisable by scale
            label_image = image_mod_crop(img)

            #store height and width of cropped groundtruth image
            height = label_image.shape[1]
            width = label_image.shape[0]
       
            #to make sure sub image extraction doesn't go out of image border
            height_lim = height - input_size
            width_lim = width - input_size

            #Generate subimages by extracting sub matrix
            for x in range(0, height_lim, stride):
                for y in range(0, width_lim, stride):
                    
                    #crop image by (label_size x label_size)
                    sub_label_image = label_image[y:y+label_size, x:x+label_size] 
                    #downsampling ground truth image by (input_size x input_size) using bicubic interpolation
                    sub_input_image = np.array(Image.fromarray(sub_label_image).resize([input_size, input_size] , resample=Image.BICUBIC)) 

                    #rescale image intensities from [0-255] to [0-1]
                    sub_label_image = im2double(sub_label_image)
                    sub_input_image = im2double(sub_input_image)

                    # resize (without this line the code goes to error)
                    sub_input_image = np.resize(sub_input_image, [input_size, input_size, 3])
                    sub_label_image = np.resize(sub_label_image ,[label_size, label_size, 3])

                    #append to data and label list
                    data.append(sub_input_image)
                    label.append(sub_label_image)
                    counter += 1
            
    
    #Shuffle pairs of data
    order = np.random.choice(counter, counter, replace=False)
    data = np.array([data[i] for i in order])
    label = np.array([label[i] for i in order])

    
    #save to HDF5 file
    with h5py.File(output_dir, 'w') as hf:
        hf.create_dataset('data', data=data, dtype='f4')
        hf.create_dataset('label', data=label, dtype='f4')



if __name__ == "__main__":
    create_dataset('/home/kudryavka/programming/python/gitclone/srcnn/data/91-image/',
    "/mnt/disk1/big_dataset/h5/train-3x.h5", input_size=10, label_size=30, scale=0, stride=15
    )

    create_dataset('/mnt/disk1/big_dataset/DVIK/',
    "/mnt/disk1/big_dataset/h5/40-divk.h5", input_size=10, label_size=30, scale=3, stride=15
    )

    create_dataset('/home/kudryavka/programming/python/gitclone/srcnn/data/Set5/',
    "/mnt/disk1/big_dataset/h5/valid-3x.h5", input_size=10, label_size=30, scale=3, stride=15
    )



