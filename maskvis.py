import torchio as tio
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
import math
import random
import numpy as np
import torch

import os
import argparse


from helpful_files.networks import PROTO, avgpool, covapool, pL, pCL, fsL, fsCL, fbpredict
from helpful_files.testing import *













parser = argparse.ArgumentParser()
parser.add_argument('--img_num', required=True,
                    help='Path to the image to be found activations on')
parser.add_argument('--model', required=True,
                    help='Path to the pretrained model')
parser.add_argument('--export', required=False,
                    default=False, help='Path to the pretrained model')


opt = parser.parse_args()
print(opt)


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, models, expanders):
        self.models = models
        self.expanders = expanders
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x,dat):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output = None
        x = self.models[0](x,dat)
        x = self.expanders[0](x,None,None)
        x.register_hook(self.save_gradient)
        conv_output = x  # Save the convolution output on that layer
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, models, expanders):
        self.models = models
        self.expanders = expanders
        for model in self.models:
            model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.models, self.expanders)

    def generate_cam(self, subj, centroids, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        k=1
        img = torch.unsqueeze(subj['img'][torchio.DATA].float(),0).cuda()
        dat = torch.unsqueeze(subj['dat'][torchio.DATA].float(),0).cuda()
        targ = torch.tensor(subj['cat'])
        conv_output, model_output = self.extractor.forward_pass(img,dat)
        out = predict(centroids[0].unsqueeze(0), model_output.unsqueeze(1).cpu())
        #out = predict(centroids[0].unsqueeze(0), out.unsqueeze(1))
        #print(out)
        _, pred = out.topk(k, 1, True, True)
        #print(pred)
        pred = pred.t()
        print(pred,targ)
        #print(pred)
        right= pred.eq(targ.view(1, -1).expand_as(pred))[:k].view(-1).sum(0, keepdim=True).float()
        """
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        """
        # Zero grads
        #self.model.fc.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        right.backward(retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2, 3))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]
        """
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        """
        return cam

def load_model(modelpath):    
    # Make Models
    localizing=True
    fewshot_local = False
    covariance_pooling=True
    ensemble = 3
    model=modelpath+'myModel.pth'
    w = 32
    models = [PROTO(w,dat_flag=True).cuda() for i in range(ensemble)]
    expander = avgpool()
    if localizing:
        if fewshot_local:
            expander = fsCL if covariance_pooling else fsL
        else:
            expander = pCL(w) if covariance_pooling else pL()
    elif covariance_pooling:
        expander = covapool
    expanders = [expander for _ in range(ensemble)]
    
    # Load saved parameters
    model_state = torch.load(model)
    for i in range(ensemble):
        models[i].load_state_dict(model_state[i])
        #models[i].eval()
        # Zero out the bias on the final layer, since it doesn't do anything
        #models[i].process[-1].layers[-1].bias.data.zero_()
    
    # Load additional parameters for parametric localizer models
    if localizing and not fewshot_local:
        fbcentroids = torch.load(model[:model.rfind('.')]+'_localizers'+model[model.rfind('.'):])
        for i in range(ensemble):
            expanders[i].centroids.data = fbcentroids[i]
            expanders[i].cuda()
    
    print("Ready to go!")
    
    return models,expanders,fbcentroids

def preprocess_image(idx, data_shape):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # Resize image
    numer_data = pd.read_json('data/kits.json', convert_dates=False).to_numpy()
    
    #this value is null which messes everything up, this is just a hacky fix for that (i should just skip this pt but I didn't do that yet)
    numer_data[151,22] = '0'
    
    #turns the one case that is a 5 into a 4, to lump these classes b/c data limitation
    numer_data[23,22] = '4'
    
    subjects = []
    
    img = nib.load(("data/case_{:05d}/imaging.nii.gz".format(idx)))
    
    case = numer_data[idx]
    datlist = []
    
    # 1-age (normalize by age 100)
    datlist.append(case[1]/100)
    # 2-gender (categorical)
    [datlist.append(genderval) for genderval in onehot(numer_data, 2, idx)]
    # 3-bmi (normalize by max/min)
    datlist.append((case[3]-min(numer_data[:,3]))/(max(numer_data[:,3])-min(numer_data[:,3])))
    # 4-comorbidities dict
    comodict = case[4]
    [datlist.append(boolval) for boolval in [float(comodict[key]) for key in comodict.keys()]]
    # 5-smoking history (categorical)
    [datlist.append(smokeval) for smokeval in onehot(numer_data, 5, idx)]
    # 6-age when quit smoking (0 for n/a or never, 1 for still, age/100 for rest)
    if case[6]==None or (case[6]=='not_applicable' and case[5]=='never_smoked'):
        datlist.append(0.)
    elif case[6]=='not_applicable':
        datlist.append(1.)
    else:
        datlist.append(float(case[6])/100)
    # 7-pack years (normalize by 100 and clip for >100)
    if math.isnan(case[7]):
        datlist.append(0.)
    else:
        val = case[7]/100
        if val>1:
            val = 1.
        datlist.append(val)
    # 8-chewing tobacco use (categorical)
    #all the same except one pt, leave out
    # 9-alcohol use (categorical)
    [datlist.append(alcohol) for alcohol in onehot(numer_data, 9, idx)]
    # 10-26 post-op measures
    # 27-surgery type (categorical-partial or radial neph)
    [datlist.append(surgtype) for surgtype in onehot(numer_data, 27, idx)]
    # 28-surgical approach (categoral- trans- or retroperitoneal)
    [datlist.append(approach) for approach in onehot(numer_data, 28, idx)]
    # 29-operative time
    # 30-cytoreductive (boolean, debulking purpose for procedure)
    datlist.append(float(case[30]))
    # 31-postop was all of the tumor removed
    # 32-last_preop_egfr (measure of kidney fxn, has value and days before procedure)
    # both have a max of 90 so normalize by that
    egfr_val = case[32]['value']
    egfr_days = case[32]['days_before_nephrectomy']
    #val
    if egfr_val == None:
        datlist.append(0.)
    elif egfr_val == '>=90':
        datlist.append(1.0)
    else:
        datlist.append(egfr_val/90)
    #days
    if egfr_days == None:
        datlist.append(0.)
    else:
        datlist.append(egfr_days/90)
    #rest is postop
    #convert all to numpy
    
    #get data in np array, clip, and normalize
    image_out = img.get_fdata()
    image_out = np.clip(image_out,-80,300) #right now is [-1024 3071]
    image_out = (image_out - np.min(image_out))/(np.max(image_out) - np.min(image_out))
    (img.get_fdata()-np.min(img.get_fdata()))/np.max(img.get_fdata()-np.min(img.get_fdata()))
    
    
    
    cd = ['0','1','2','3a','3b','4']
    cd_score_out = cd.index(case[22])
    
    
    #shuffle axes, resize, add channel axis to image
    image_out = np.moveaxis(image_out, [0, 1], [-1, -2])
    zoom_vals = tuple(want/have for want,have in zip(data_shape, image_out.shape))
    image_out = zoom(image_out,zoom_vals)
    image_out = image_out[np.newaxis,:,:,:]
    
    
    a = np.array(datlist)
    a = np.expand_dims(a,0)
    b_mat = a.T*a
    b_mat = np.expand_dims(b_mat,2)
    pt_data_out = a*b_mat
    zoom_vals = tuple(want/have for want,have in zip(data_shape, pt_data_out.shape))
    pt_data_out = zoom(pt_data_out,zoom_vals)
    pt_data_out = pt_data_out[np.newaxis,:,:,:]
    
    #pt_data_out = torch.tensor(datlist)
    
    subj = tio.Subject(
        img=tio.ScalarImage(tensor=torch.tensor(image_out)),
        dat=tio.ScalarImage(tensor=torch.tensor(pt_data_out)),
        cat=cd_score_out,
        )
    return subj
    
def onehot(alldata, index, caseindex):
	#this deals with getting the onehot encoding for fields w/ categorical text vals
	unique = np.unique(alldata[:,index])
	onehot_vec = np.zeros(len(unique))
	onehot_vec[np.where(unique==alldata[caseindex,index])] = 1
	return onehot_vec
        
    
if __name__ == '__main__':    
    image_num = int(opt.img_num)
    # Open CV preporcessing
    image_prep = preprocess_image(image_num,(128,128,128))
    # Load the model
    modelpath=opt.model
    models,expanders,fbcentroids = load_model(modelpath)
    # Grad cam
    grad_cam = GradCam(models, expanders)
    # Generate cam mask
    cam = grad_cam.generate_cam(image_prep, fbcentroids)
    # Save mask
    save_class_activation_on_image(image, cam, file_name_to_export)    
    print('Grad cam completed')
