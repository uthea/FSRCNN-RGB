import torch
import torchvision
from custom_loader import CustomDataset
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from math import log10

class FSRCNN(torch.nn.Module):

    def __init__(self, d, s, c, m, upscale_factor, num_epochs, batch_size, layers_lr, deconv_lr, ckpt, ckpt_mode, padding):
        super(FSRCNN, self).__init__()

        #hyper-parameters
        self.upscale_factor = upscale_factor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers_lr =  layers_lr
        self.deconv_lr = deconv_lr

        #model parameters
        self.d = d
        self.s = s
        self.c = c
        self.m = m
        self.model = torch.nn.Sequential()

        #criterion and optimizer function
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.optimizer = None

        #state dict path
        self.ckpt = ckpt

        #load state for initial epoch
        self.initial_epoch = 0
        
        #initialize model
        self.init_model(ckpt_mode)

        #padding configuration
        self.padding = padding


    def save(self, state):
        torch.save(state, self.ckpt)
    
    def resume(self):
        #resume state
        checkpoint = torch.load(self.ckpt, map_location=self.device)
        self.initial_epoch = checkpoint['initial_epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        #manually bring optimizer tensor to cuda if gpu is available
        if torch.cuda.is_available():
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    
    def load(self):
        #load checkpoint and model 
        checkpoint = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        #set learning rate to 1
        self.initial_epoch = 1

        #reset learning rate
        checkpoint['optimizer']['param_groups'][0]['lr'] = self.deconv_lr
        checkpoint['optimizer']['param_groups'][1]['lr'] = self.layers_lr
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        #manually bring optimizer tensor to cuda if gpu is available
        if torch.cuda.is_available():
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    

    def init_weight(self, model):
        #initialize convolutional layers with kaiming normal distribution
        if type(model) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0)
        
        #initialize deconv layer with gaussian distribution with std = 1e-3
        elif type(model) == torch.nn.ConvTranspose2d:
            torch.nn.init.normal_(model.weight, std=1e-3)
            model.bias.data.fill_(0)


    def init_model(self, ckpt_mode):
        #feature extraction and shrinking
        self.model.add_module('feature_extraction', torch.nn.Conv2d(in_channels=self.c, out_channels=self.d, kernel_size=(5,5), padding=self.padding[0]))
        self.model.add_module('prelu_1', torch.nn.PReLU())
        self.model.add_module('shrink', torch.nn.Conv2d(in_channels=self.d, out_channels=self.s, kernel_size=(1,1), padding=self.padding[1]))
        self.model.add_module('prelu_2', torch.nn.PReLU())

        #linear mapping layer
        for i in range(self.m):
            self.model.add_module('lm{}'.format(i+1), torch.nn.Conv2d(in_channels=self.s, out_channels=self.s, kernel_size=(3,3), padding=self.padding[2]))
        self.model.add_module('prelu_3', torch.nn.PReLU())

        #expanding and upscalling
        self.model.add_module('expanding', torch.nn.Conv2d(in_channels=self.s, out_channels=self.d, kernel_size=(1,1), padding=self.padding[3]))
        self.model.add_module('prelu_4', torch.nn.PReLU())
        self.model.add_module('deconvolution', torch.nn.ConvTranspose2d(in_channels=self.d, out_channels=self.c, kernel_size=(9,9), stride=self.upscale_factor, padding=self.padding[4]))

        #divide layers into two groups
        deconvolution = []
        rest_of_layers = []
        for name, param in self.model.named_parameters():
            if name in ['deconvolution.weight', 'deconvolution.bias']:
                deconvolution.append(param)
            else:
                rest_of_layers.append(param)
        
        #set the learning rate for deconvoltuion layer and the rest of layers
        self.optimizer = torch.optim.Adam(
            [
                {'params': deconvolution, 'lr': self.deconv_lr},
                {'params': rest_of_layers, 'lr': self.layers_lr}
            ]
        )

        # initialize weight and bias or resume / load checkpoint
        if ckpt_mode == "resume":
            self.resume()
        elif ckpt_mode == "load":
            self.load()
        elif ckpt_mode == "new":
            self.model.apply(self.init_weight)

        
        #bring model to cpu or gpu
        self.model.to(self.device)


    
    def train(self, train_deconv_only, train_path, validation_path, summary_path):
        #load summary writer
        writer = SummaryWriter(summary_path)

        #load dataset
        custom_dataset_train = CustomDataset(train_path, 
        transform=torchvision.transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(
            dataset=custom_dataset_train,
            batch_size=self.batch_size,
            shuffle=False
        )

        custom_dataset_valid = CustomDataset(validation_path,
        transform=torchvision.transforms.ToTensor())

        valid_loader = torch.utils.data.DataLoader(
            dataset=custom_dataset_valid,
            batch_size=self.batch_size,
            shuffle=True
        )


        total_step = len(train_loader)

        #train deconv only or all layers
        for name, param in self.model.named_parameters():
            if name in ['deconvolution.weight', 'deconvolution.bias']:
                param.requires_grad = True
            else:
                if train_deconv_only == True:
                    param.requires_grad = False
                else :
                    param.requires_grad = True  


        #start training
        for epoch in range(self.initial_epoch, self.num_epochs+1):
            total_loss_train = 0
            total_loss_val = 0
            valid_psnr = 0
            train_psnr = 0

            iterator = iter(valid_loader)

            for step, (train_images, train_labels) in enumerate(train_loader):
                self.optimizer.zero_grad() #reset gradient computation

                #also move tensor directly to cpu or gpu
                train_images = train_images.to(self.device) 
                train_labels =  train_labels.to(self.device)
            
                #forward pass 
                output = self.model(train_images)
                #calulate loss
                loss = self.criterion(output, train_labels).to(self.device)
                total_loss_train+= loss.data.item()

                with torch.no_grad():
                    #validation phase
                    try:
                        valid_images, valid_labels = iterator.next()
                    except StopIteration:
                        iterator = iter(valid_loader)
                        valid_images, valid_labels = iterator.next()

                    valid_images = valid_images.to(self.device)
                    valid_labels = valid_labels.to(self.device)

                    valid_out = self.model(valid_images)
                    valid_loss = self.criterion(valid_out, valid_labels).to(self.device)
                    total_loss_val += valid_loss.item()

                    #calculate psnr
                    valid_psnr += 10 * log10(1/ valid_loss.item())
                    train_psnr += 10 * log10(1/ loss.item())

                #backward pass and optimization
                loss.backward() #compute gradient
                self.optimizer.step() #update weight


            #print total loss
            if (epoch) % 1 == 0:
                print('Epoch [{}/{}], Train Loss: {:.6f}, Valid Loss: {:.6f}, Train PSNR: {:.3f}, Valid PSNR: {:.3f}'.format(
                    epoch, self.num_epochs, total_loss_train/total_step, total_loss_val/total_step, train_psnr/total_step, valid_psnr/total_step
                )) #loss.item() convert single scalar tensor to numbers
            
            #logging
            writer.add_scalar('epoch_loss/train', total_loss_train/(total_step), epoch)
            writer.add_scalar('epoch_loss/validation', total_loss_val/(total_step), epoch)
            writer.add_scalar('metric/train_psnr', train_psnr/total_step, epoch)
            writer.add_scalar('metric/valid_psnr', valid_psnr/total_step, epoch)
        
            #save model, optimizer and epoch
            self.save(
                {
                    'initial_epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
            )

    def rescale(self, img):
        img = img * 255
        img[img > 255] = 255
        img[img < 0] = 0
        return img.astype(np.uint8)
    
    def bicubic(self, input_image):
        #test method for comparing fsrcnn result with the original bicubic interpolation
        img = Image.open(input_image)
        img = img.resize((img.size[0] * self.upscale_factor, img.size[1] * self.upscale_factor), Image.BICUBIC)
        img.save("bicubic.png")
            

    def upscale(self, input_image):
        #read image and transfrom it to tensor 
        img = Image.open(input_image)
        img = torchvision.transforms.ToTensor()(img) #transform automatically rescale the intensities from [0-255] to [0-1]
        img = img.expand(1, img.size()[0], img.size()[1], img.size()[2]).to(self.device) #[batch_size, depth, width, height]

        with torch.no_grad():
            out = self.model(img).cpu() #take output to ram
            out = out.squeeze() #remove the batch dimension (now it becomes [depth, width, height])
            out = out.permute(1, 2, 0) #swap axis from [depth, width, height] to [height, width, depth]

            final = self.rescale(np.array(out)) #rescale from [0-1] to [0-255] (using torchvision.transforms.ToPILImage() cause artifacts)
            final = Image.fromarray(final) #convert back ndarray to pil image
            
            final.save("output.png") #save output to file
    

        
        
        
        
            




        