import torch
import torchio as tio
from torch.nn.functional import interpolate
from torchio.transforms import Lambda,RescaleIntensity
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm





def load_pretrain_datasets(data_shape,batch=3,workers=4,transform=None):
    
    data_path = '/home/mitch/Data/MSD/'
    directories = sorted(glob.glob(data_path + '*/'))
    
    loaders = [] #var to store dataloader for each task
    datasets = [] #store dataset objects before turning into loaders
    
    if transform == None:
        transform = tio.RandomFlip(p=0.)
    #preprocess all
    clippy = Lambda(lambda x: torch.clip(x,-80,300),types_to_apply=[tio.INTENSITY])
    normal = RescaleIntensity((0.,1.))
    resize = Lambda(lambda x:torch.squeeze(interpolate(torch.unsqueeze(x,dim=0),data_shape),dim=0))
    rounding = Lambda(lambda x: torch.round(x),types_to_apply=[tio.LABEL])
    transform = tio.Compose([clippy,normal,resize,rounding,transform])
    
    
    #deal with weird shapes
    braintransform = Lambda(lambda x: torch.unsqueeze(x[:,:,:,2],dim=0),
                                    types_to_apply=[tio.INTENSITY])
    braintransform = tio.Compose([braintransform,transform])
    prostatetransform = Lambda(lambda x: torch.unsqueeze(x[:,:,:,1],dim=0),
                                    types_to_apply=[tio.INTENSITY])
    prostatetransform = tio.Compose([prostatetransform,transform])
    
    for i,directory in enumerate(directories):
        images = sorted(glob.glob(directory+'imagesTr/*'))
        segs = sorted(glob.glob(directory+'labelsTr/*'))
        
        subject_list = []
        
        for image,seg in zip(images,segs):
            
            subject_list.append(tio.Subject(
                                img=tio.ScalarImage(image),
                                label=tio.LabelMap(seg)
                                ))
            
        #handle special cases
        if i==0:
            datasets.append(tio.SubjectsDataset(subject_list,transform=braintransform))
        elif i==4:
            datasets.append(tio.SubjectsDataset(subject_list,transform=prostatetransform))
        else:
            datasets.append(tio.SubjectsDataset(subject_list,transform=transform))
        
        loaders.append(DataLoader(datasets[-1],num_workers=workers,batch_size=batch, pin_memory=True))
    
    return loaders
    
def load_kidney_seg(data_shape,batch=3,workers=4,transform=None):
    
    #take input transform and apply it after clip, normalization, resize
    if transform == None:
        transform = tio.RandomFlip(p=0.)
    #preprocess all
    clippy = Lambda(lambda x: torch.clip(x,-80,300),types_to_apply=[tio.INTENSITY])
    normal = RescaleIntensity((0.,1.))
    resize = Lambda(lambda x:torch.squeeze(interpolate(torch.unsqueeze(x,dim=0),data_shape),dim=0))
    rounding = Lambda(lambda x: torch.round(x),types_to_apply=[tio.LABEL])
    transform = tio.Compose([clippy,normal,resize,rounding,transform])
    
    
    subject_list = []
    for i in range(210):
        pt_image = ("data/case_{:05d}/imaging.nii.gz".format(i))
        pt_label = ("data/case_{:05d}/segmentation.nii.gz".format(i))
        subject_list.append(tio.Subject(
                                img=tio.ScalarImage(pt_image),
                                label=tio.LabelMap(pt_label)
                                ))
    dataset = tio.SubjectsDataset(subject_list,transform=transform)
    return DataLoader(dataset,num_workers=workers,batch_size=batch,pin_memory=True)
    
    
if __name__ == "__main__":
    loaders = load_pretrain_datasets((128,128,128))#,transform=tio.RandomFlip())
    
    for loader in loaders:
        for i,dat in enumerate(loader):
            print(dat['img'][tio.DATA].size())

