#
# postprocessing.py - The post processing class, which contains a bunch of static
# methods that we used, including methods for smoothing, thresholding, graphing 
# and output formatting.
#


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from scipy.signal import savgol_filter


class PostProcessing():

	@staticmethod	
	def flipBinaryVec(vec):
		for i in range(vec.shape[-1]):
			if vec[i].item() == 0:
				vec[i] = 1
			else:
				vec[i] = 0	
		return vec

	@staticmethod
	def smoothBinary(vec, window_size, thresh):
		if window_size%2 == 0:
			window_size -= 1
		vec2 = savgol_filter(vec, window_size, 18)	
		return vec2

	@staticmethod
	def smoothBinaryPost(vec,window_size,thresh):	
		for i in range(window_size, vec.shape[-1]):
			before = np.sum(vec[i-window_size : i])
			after  = np.sum(vec[i+1 : i + 1 + window_size])
			if before <= thresh and after <= thresh:
				vec[i] = 0
			elif before > thresh and after > thresh:
				vec[i] = 1
		return vec
	@staticmethod
	def movingAverageFilter(vec, window_size):
		for i in range(window_size, vec.shape[-1]):
			avg = torch.sum(vec[i-window_size:i]) / window_size
			vec[i-1] = avg			
		return vec

	@staticmethod
	def graphRawOutput():
		y_vec = torch.tensor([])
		folder = sys.argv[1]
		for file_name in os.listdir(folder):
			f = torch.load(file_name)
			softmaxed = torch.nn.functional.softmax(f[1:3,:], dim=0)
			y_vec = torch.cat((y_vec, softmaxed[0,:]) )
		
		torch.save(y_vec, "y_vecSoftmaxedRawOutput.pt")
		plt.figure()
		plt.hist(y_vec, bins=50, density=True)
		plt.title("Raw Output Histogram")
		plt.axis([0, 1, 0, 100000000])
		plt.savefig("rawOutputHist.png")	

	@staticmethod
	def graphRawRocCurve():
		threshold = np.linspace(0,1,10)
		tp = np.zeros(len(threshold))
		tn = np.zeros(len(threshold))
		fp = np.zeros(len(threshold))
		fn = np.zeros(len(threshold))
		folder = sys.argv[1]
		for file_name in os.listdir(folder):
			f = torch.load(file_name)	
			softmaxed = torch.nn.functional.softmax(f[1:3,:], dim=0)
			for i, thresh in enumerate(threshold):
				tmp = np.zeros(softmaxed.shape[-1])
				tmp[softmaxed[1,:] > thresh]  = 1
				for j in range(tmp.shape[-1]):
					if tmp[j] == 1 and f[0,j] == 1:
						tp[i] += 1	
					elif tmp[j] == 1 and f[0,j] == 0:
						fp[i] += 1
					elif tmp[j] == 0 and f[0,j] == 1:
						fn[i] += 1
					elif tmp[j] == 0 and f[0,j] == 0:
						tn[i] += 1
		sensitivity  = np.divide(tp, tp+fn)  
		specificity = np.divide(tn, fp+tn)
		print("sensitivity: ", sensitivity)
		print("specificity: ", specificity)
		plt.figure()
		plt.plot(1-specificity, sensitivity)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.savefig("rawRocOutput.png")	

	@staticmethod
	def keeneGraph(target_vec, pred, threshold, name, window_size, vote_count):
		plt.figure()
		plt.subplot(3,1,1)
		plt.plot(target_vec)
		plt.subplot(3,1,2)
		thresholded = PostProcessing.thresholding(pred, threshold)
		plt.plot(thresholded)
		plt.subplot(3,1,3)
		smoothed = PostProcessing.smoothBinary(thresholded, window_size, vote_count) 
		smoothed = PostProcessing.smoothBinary(smoothed, window_size, vote_count) 
		plt.plot(smoothed)
		clean_name = name.split("/")[-1].split(".")[0]
		plt.savefig("keeneGraph_{}_{}_{}_{}.png".format(threshold,clean_name, window_size, vote_count))


	@staticmethod
	def graphOutputs(target_vec, pred):
		plt.figure()
		plt.plot(target_vec, label='Target')
		plt.plot(pred, 'o', label='Prediction')
		plt.legend()
		plt.axis([0, target_vec.shape[0], 0, 1.2])
		plt.savefig("graphOutputPlot.png")

	@staticmethod
	def thresholding(pred_tuple, threshold):
		softmaxed = torch.nn.functional.softmax(pred_tuple, dim=0)
		thresholded = np.zeros(pred_tuple.shape[-1])
		for i in range(softmaxed.shape[1]):
			if softmaxed[1,i] > threshold:
				thresholded[i] = softmaxed[1,i]
		return thresholded

	@staticmethod
	def justThresh(vec, thresh):
		for i in range(vec.shape[-1]):
			if vec[i] < thresh:
				vec[i] = 0
		return vec
			
	@staticmethod
	def softmaxing(vec):
		softmaxed = torch.nn.functional.softmax(vec, dim=0)
		return softmaxed	

	@staticmethod	
	def createTseFile(vec, name, path, freq):
		"""
		Takes a vector, converts it into tse form, and writes it into the file with specified path.
		Returns its full path name.		
		"""	
		pwd = os.path.join(path, name+'.txt')
		f = open(pwd, 'w')

		f.write("version = tse_v1.0.0\n")
		f.write("\n")

		
		seiz_or_bckg = False if (vec[0].item() == 0) else True 	# seiz=True | bckg=False
		start = 0
		end   = 0
	
		for i, ele in enumerate(vec):
			# 4 cases: we can be in seiz/bckg, and it can currently be a 
			# a seiz/bckg
			if not seiz_or_bckg:	    # in bckg
				if ele.item() == 0: # keep bckg
					continue	
				else:		    # start seiz
					seiz_or_bckg = True
					end = (i-1)/freq
					f.write(f"{start} {end} bckg 1.0000\n")
					start = i/freq
			else: # in a seizure	
				if ele.item() == 0:
					seiz_or_bckg = False
					end = (i-1)/freq
					f.write(f"{start} {end} seiz 1.0000\n")
					start = i/freq
				else:
					continue
		vec_len = vec.shape[0]

		if not seiz_or_bckg:	# finished in bckg
			end = (vec_len-1)/freq
			f.write(f"{start} {end} bckg 1.0000\n")
		else: 			# finished in seiz
			end = (vec_len-1)/freq
			f.write(f"{start} {end} seiz 1.0000\n")
		
		return pwd	


	@staticmethod	
	def contest_createTseFile(vec, freq):
		"""
		Takes a vector, store the seizure occurances in tse form and returns
		it into as an array of dictionaries with 'begin', 'end' and 
		'confidence' keys..		
		"""	
		output_arr = []	
		seiz_or_bckg = False if (vec[0].item() == 0) else True 	# seiz=True | bckg=False
		start = 0
		end   = 0

		avg_confidence = [0,0] if (seiz_or_bckg==False) else [vec[0].item(),1]  # [total sum, number of elements]
		for i, ele in enumerate(vec):
			# 4 cases: we can be in seiz/bckg, and it can currently be a 
			# a seiz/bckg
			if not seiz_or_bckg:	    # in bckg
				if ele.item() == 0: # keep bckg
					continue	
				else:		    # start seiz
					seiz_or_bckg = True
					start = i/freq
					avg_confidence[0] = ele.item()
					avg_confidence[1] = 1
			else: # in a seizure	
				if ele.item() == 0:
					seiz_or_bckg = False
					end = (i-1)/freq
					cur_dict = {}
					cur_dict["begin"]      = start
					cur_dict["end"]        = end
					cur_dict["confidence"] = avg_confidence[0]/avg_confidence[1]
					output_arr.append(cur_dict)	
	
					start = i/freq
					avg_confidence[0] = ele.item()
					avg_confidence[1] = 1
				else:
					avg_confidence[0] += ele.item()
					avg_confidence[1] += 1
		vec_len = vec.shape[0]

		if seiz_or_bckg:	# finished in bckg
			end = (vec_len-1)/freq
			cur_dict = {}
			cur_dict["begin"]      = start
			cur_dict["end"]        = end
			cur_dict["confidence"] = avg_confidence[0]/avg_confidence[1]		
			output_arr.append(cur_dict)	
		
		return output_arr

