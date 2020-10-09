from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import math
import torch
from scipy.ndimage import zoom

class KiTS_Set(Dataset):
	"""
	Each item will contain "scan_data", preoperative data (labels from KiTS dataset json file)
	Right now the label/gt is "clavien_surgical_complications"
    """
		
	def __init__(self,data_shape):
		#read the json file and convert to np array
		self.numer_data = pd.read_json('data/kits.json', convert_dates=False).to_numpy()
		
		#this value is null which messes everything up, this is just a hacky fix for that (i should just skip this pt but I didn't do that yet)
		self.numer_data[151,22] = '0'
		
		#turns the one case that is a 5 into a 4, to lump these classes b/c data limitation
		self.numer_data[23,22] = '4'
		
		#for resizing
		self.data_shape = data_shape
		
		#for binary classification which CD scores are 0
		#self.neglist = neglist
	
	
	def __len__(self):
		return len(self.numer_data)
		
	def class_weights(self):
	    #get total counts per-class for class balancing via lr
		class_counts = pd.value_counts(self.numer_data[:,22])
		return class_counts
	
	def onehot(self, alldata, index, caseindex):
		#this deals with getting the onehot encoding for fields w/ categorical text vals
		unique = np.unique(alldata[:,index])
		onehot_vec = np.zeros(len(unique))
		onehot_vec[np.where(unique==alldata[caseindex,index])] = 1
		return onehot_vec
		
	
	def __getitem__(self,idx):
		
		#start with image data
		img = nib.load(("data/case_{:05d}/imaging.nii.gz".format(idx)))
		
		#deal with patient numerical case data
		case = self.numer_data[idx]
		
		
		
		#hold on to your hats this is going to get messy
		#grab the preop fields and normalize or append onehots or parse through dict
		
		datlist = []
		
		# 1-age (normalize by age 100)
		datlist.append(case[1]/100)
		# 2-gender (categorical)
		[datlist.append(genderval) for genderval in self.onehot(self.numer_data, 2, idx)]
		
		# 3-bmi (normalize by max/min)
		datlist.append( (case[3]-min(self.numer_data[:,3]))/(max(self.numer_data[:,3])-min(self.numer_data[:,3])) )
		
		# 4-comorbidities dict
		comodict = case[4]
		[datlist.append(boolval) for boolval in [float(comodict[key]) for key in comodict.keys()]]
		
		# 5-smoking history (categorical)
		[datlist.append(smokeval) for smokeval in self.onehot(self.numer_data, 5, idx)]
		
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
		[datlist.append(alcohol) for alcohol in self.onehot(self.numer_data, 9, idx)]
		
		# 10-26 post-op measures
		
		# 27-surgery type (categorical-partial or radial neph)
		[datlist.append(surgtype) for surgtype in self.onehot(self.numer_data, 27, idx)]
		
		# 28-surgical approach (categoral- trans- or retroperitoneal)
		[datlist.append(approach) for approach in self.onehot(self.numer_data, 28, idx)]
		
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
		
		"""
		#now get the label, which right now will be the one hot encoding of clavien-dindo
		label = []
		
		[label.append(cdscore) for cdscore in self.onehot(self.numer_data, 22, idx)]
		"""
		
		#binary label for cd - if in neglist or null then is 0
		"""
		if case[22] in self.neglist:
			label = 0.
		else:
			label = 1.
		"""
		
		#convert all to numpy
		
		#get data in np array, clip, and normalize
		image_out = img.get_fdata()
		image_out = np.clip(image_out,-80,300) #right now is [-1024 3071]
		image_out = (image_out - np.min(image_out))/(np.max(image_out) - np.min(image_out))
		#(img.get_fdata()-np.min(img.get_fdata()))/np.max(img.get_fdata()-np.min(img.get_fdata()))
		
		pt_data_out = np.array(datlist)
		
		cd_score_out = case[22]
		
		
		#shuffle axes, resize, add channel axis to image
		image_out = np.moveaxis(image_out, [0, 1], [-1, -2])
		zoom_vals = tuple(want/have for want,have in zip(self.data_shape, image_out.shape))
		image_out = zoom(image_out,zoom_vals)
		image_out = image_out[np.newaxis,:,:,:]
		
		img_tensor = torch.from_numpy(image_out)
		pt_data_tensor = torch.from_numpy(pt_data_out)
		#cd_score_tensor = torch.from_numpy(cd_score_out)
		
		
		return img_tensor, pt_data_tensor, cd_score_out
				
