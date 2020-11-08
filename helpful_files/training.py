import numpy as np
import torch
from copy import deepcopy
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm
import torchio
from scipy.ndimage import zoom

def load_transform(path, boxdict, transform, flipping, masking):
    # Load the image
    flip = np.random.choice([True, False])
    with open(path, 'rb') as f:
        p = Image.open(f)
        p = p.convert('RGB')
    if flip and flipping:
        p = p.transpose(Image.FLIP_LEFT_RIGHT)
    t = transform(p)
    # Load the bounding boxes
    m = t
    
    #I'm removing this section altogether b/c I'm not using annotated bounding boxes right now
    """
    if masking:
        allmasks = np.zeros((10,10))
        boxes = boxdict[path]
        for box in boxes:
            mask = np.zeros((10,10))
            xmin = box[0]
            xmax = box[2]-.0000001 # Prevents overflow when xmax is exactly 10
            ymin = box[1]
            ymax = box[3]-.0000001
            xmin_int = int(xmin)
            xmax_int = int(xmax)+1
            ymin_int = int(ymin)
            ymax_int = int(ymax)+1
            if not flip or not flipping:
                mask[ymin_int:ymax_int, xmin_int:xmax_int] = 1
                # Fade out the left and right edges of the mask
                mask[:, xmin_int] *= 1 - (xmin - xmin_int)
                mask[:, xmax_int-1] *= 1 - (xmax_int - xmax)
            else:
                mask[ymin_int:ymax_int, 10-xmax_int:10-xmin_int] = 1
                # Fade out the left and right edges of the mask
                mask[:, 10-xmin_int-1] *= 1 - (xmin - xmin_int)
                mask[:, 10-xmax_int] *= 1 - (xmax_int - xmax)
            # Fade out the top and bottom edges of the mask
            mask[ymin_int,:] *= 1 - (ymin - ymin_int)
            mask[ymax_int-1,:] *= 1 - (ymax_int - ymax)
            # Take the union of the previous and current masks
            allmasks = 1 - (1-allmasks)*(1-mask) 
        m = torch.FloatTensor(allmasks).unsqueeze(0)
    """
    return [t, m]

def pt_data_expand(dat,data_shape):
    vecs = torch.tensor([])
    for vec in dat:
        #print(vec.size(),data_shape[2:])
        vec = torch.unsqueeze(vec,0)
        b_mat = vec.T*vec
        b_mat = torch.unsqueeze(b_mat,2)
        vec = (vec*b_mat)
        zoom_vals = tuple(want/have for want,have in zip(data_shape, vec.size()))
        vec = torch.tensor(zoom(vec,zoom_vals))
        vec = torch.unsqueeze(vec,0) #channel, next one is batch size
        vecs = torch.cat((vecs,torch.unsqueeze(vec,0)))
    return vecs

class ProtoSampler(Sampler):
    def __init__(self, data_source, way, shots):
        iddict = dict()
        for i,pt in enumerate(tqdm(data_source)):
            cat = pt['cat']
            if cat in iddict:
                iddict[cat].append(i)
            else:
                iddict[cat] = [i]
        self.iddict = iddict
        self.way = way
        self.shots = shots
        
    def __iter__(self):
        # Build new dictionary, shuffle entries
        trackdict = deepcopy(self.iddict)
        for key in trackdict:
            np.random.shuffle(trackdict[key])
        # Choose categories, sample, eliminate small categories
        idlist = []
        while len(trackdict.keys()) >= self.way:
            # Draw categories proportional to current size
            pcount = np.array([len(trackdict[k]) for k in list(trackdict.keys())])
            cats = np.random.choice(list(trackdict.keys()), size=self.way, replace=False, p=pcount/sum(pcount))
            for shot in self.shots:
                for cat in cats:
                    for _ in range(shot):
                        idlist.append(trackdict[cat].pop())
            for cat in cats:
                if len(trackdict[cat]) < sum(self.shots):
                    trackdict.pop(cat)
            yield idlist
            idlist = []
            
            
def train(train_loader, models, optimizer, criterion, way, shots, verbosity):
    for model in models:
        model.train()
    nqueries = shots[-1]
    ensemble = len(models)
    targ = torch.LongTensor([i//nqueries for i in range(nqueries*way)]).cuda()
    allloss = [0]*ensemble
    acctracker = [0]*ensemble
    #val_acc = [0]*ensemble
    print("Training images covered this round:")
    for i, pt in enumerate(train_loader):
        img = pt['img'][torchio.DATA].float().cuda()
        dat = pt['dat'].float()
        dat = torch.clamp((dat + (torch.rand(dat.size())-0.5)/5.), 0., 1.) #add noise
        dat = pt_data_expand(dat,img.size()[2:]).cuda() #expand to 3d
        #masks = masks.cuda()
        for j in range(ensemble):
            models[j].zero_grad()
            # Predict, step
            out = models[j](img, img, dat)
            loss = criterion(out, targ)
            loss.backward()
            optimizer[j].step()
            # Record training statistics
            allloss[j] += loss.item()
            _,bins = torch.max(out,1)
            #print(bins,targ)
            acc = torch.sum(torch.eq(bins,targ)).item()/nqueries/way
            acctracker[j] += acc
            
    return [L/(i+1) for L in allloss], [L/(i+1) for L in acctracker] 


