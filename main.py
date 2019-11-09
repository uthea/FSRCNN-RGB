from fsrcnn import FSRCNN
import os
from data_processing import image_mod_crop
from PIL import Image

#run export OMP_NUM_THREADS=1 in terminal first berofre run this
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    #small setting  = 32, 5 and m = 1
    #original setting = 56, 16 and m = 4

    #image 91 1000 epoch
    #divk-40 dataset for 100 epoch
    #stride 3 with pad [1,0,1,0,0]
    #stride 2 with pad [1,0,1,0,1] 50 epoch

    #48,12 and m = 4
    net = FSRCNN(
        d = 48, 
        s = 12, 
        c = 3,
        m = 4, 
        upscale_factor = 3, 
        num_epochs = 100,
        batch_size = 128,
        layers_lr=1e-3,
        deconv_lr=1e-4,
        ckpt="./model/model-3x.ckpt",
        ckpt_mode='resume',
        padding=[1,0,1,0,0]
    )

    # start training
    net.train(
        train_deconv_only=True,
        train_path='/mnt/disk1/big_dataset/h5/train-3x.h5',
        validation_path='/mnt/disk1/big_dataset/h5/valid-3x.h5',
        summary_path='./runs/deconv_train-3x'
    )

    img_path = '/home/kudryavka/programming/python/gitclone/srcnn/data/Set5/butterfly_GT.bmp'

    #upscalling
    net.upscale(
    input_image=img_path
    )

    net.bicubic(
        input_image=img_path
    )





