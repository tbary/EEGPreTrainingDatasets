import pandas as pd
import numpy as np
import pywt
import pickle
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import scipy.stats as stats
import pyedflib as edf
from os import listdir, remove, environ, chdir
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# Data pre-processing ============================================================================================

class Dataset():
	"""This class is used to preprocess a signal into a suitable dataset by :
		- loading the signal and the target, 
		- removing unsuitable segments (shortly before a target change),
		- splitting the whole signal into segments of a given time, and
		- transforming the splits into scalograms using a wavelet.

		It can also load already processed signals in order to directly transform them
		into scalograms.
	"""

	def __init__(self, freq):
		self.data = None
		self.target = None
		self.slice_time = None
		# Sampling frequency of the signal
		self.freq = freq

	def initialize(self, data_file, targ_file):
		"""Reads the raw data from [data_file] and [targ_file], removes nan target values,
			adds a time vector, and saves the resulting numpy arrays in self.data and 
			self.target respectively.
			
			pre : - data_file (string): path to the file containing the raw signal.
					- targ_file (string): path to the file containing the target.
			post: /"""

		self.target = pd.read_csv(targ_file).values[:,0]
		nan_indexes = np.isnan(self.target)
		self.target = self.target[~nan_indexes]

		self.data = (pd.read_csv(data_file).values[~nan_indexes]).T
		# Insert a row with time at position 0.
		self.data = np.insert(self.data,0,np.arange(self.data.shape[1])/self.freq,axis=0)

	def discard_delay(self, delay):
		"""Removes [delay] seconds in self.data after each change of state in self.target. 
				pre : - delay (float): number of second to remove after each change of state.
				post: /"""
			
			# Number of samples to remove after each state change.
		x_samples = int(self.freq*delay)

			# Find where the target changes state.
		changes_idx = np.flatnonzero(np.diff(self.target))

			# Array with indexes to remove.
		if changes_idx != []:
			idx_to_rmv = np.empty((len(changes_idx), x_samples), dtype=int)
			for idx in range(len(changes_idx)-1):
				idx_to_rmv[idx]=np.arange(changes_idx[idx],changes_idx[idx]+x_samples)
			idx_to_rmv[-1]=np.arange(changes_idx[-1],min(changes_idx[-1]+x_samples,len(self.target)))

			idx_to_rmv = idx_to_rmv.flatten()

			# Delete indexes from arrays.
			self.data = np.delete(self.data,idx_to_rmv,axis=1)
			self.target = np.delete(self.target,idx_to_rmv)

	def split_and_clean(self, slice_time):
		"""Splits self.data and self.target into segments of [slice_time]*self.freq samples,
			removes segments including a target change as well as the last one, and removes
			the time vector.
			pre : - slice_time (float): duration in seconds of each segments.
			post: /"""

		self.slice_time = slice_time
		slice_sample_nbr = slice_time*self.freq

		# Split dataset and target into slices of slice_sample_nbr samples.
		prep_data = np.split(self.data,np.arange(slice_sample_nbr,self.data.shape[1],slice_sample_nbr,dtype=int),axis=1)
		prep_targ = np.split(self.target,np.arange(slice_sample_nbr,len(self.target),slice_sample_nbr,dtype=int))

		# Discards sub-datasets that contain a time jump (and may contain a change in target state)
		# by computing time vector differences.
		elem = 0
		while elem < len(prep_data):
				if (prep_data[elem][0,-1]-prep_data[elem][0,0])>slice_time:
						del prep_data[elem]
						del prep_targ[elem]
				else:
						elem+=1

		# Since all splits of target dataset have same value, replace each split by said value.
		prep_targ = [np.mean(elem) for elem in prep_targ]

		# Discard last sub-dataset as it is smaller than the others.
		del prep_data[elem-1]
		del prep_targ[elem-1]
		self.data = np.array(prep_data)
		# Remove time vector.
		self.data = np.delete(self.data,0,axis=1)
		self.target = np.array(prep_targ, dtype=int)

	def initialize_processed(self,data,target=None):
		"""Stores already pre-processed [data] in self.data, saves a vector of ones the same size 
			as data.shape[0] and infers self.slice_time from data.shape[2]
			pre : - data (np.ndarray - 3D): pre-processed data, in our case artificial data for pre-training
			post: /"""

		self.data=data
		if target is None:
			self.target=np.ones(data.shape[0])
		else:
			self.target=target
		self.slice_time=data.shape[2]/self.freq

	def gen_inputs(self, path, n_patches, projection_dim, start=0, stop=-1, shift=0, wavelet='morl', upper_freq=80, show=False, overwrite=False, disable_progress_bar = False):
		"""Takes slices of self.data ranging from [start] to [stop], applies a Continuous Wavelet Transform (CWT) to each 
			of them, splits the resulting scalogram into [n_patches]**2 patches ([n_patches] horizontally and vertically),
			and randomly projects each patch into a [projection_dim] dimensional space with added positional embedding. 
			The result is then saved in [path] under the format "[path]/scal_spl{i}.mat", where {i} is the index 
			corresponding to the slice. The scalogram obtained after CWT is a squared image.

			pre : - path (string): path to the directory where the projected patches are saved.
					- n_patches (int): number of horizontal and vertical divisions of each scalogram.
					- projection_dim (int): output dim of the patch random projection.
					- start (int): start index for the slice in self.data.
					- stop (int): stop index for the slice in self.data. If equal to -1, goes to the last slice.
					- wavelet (string): wavelet to be used in the CWT. Same as the ones available in pywt package
					- upper_freq (float): highest frequency to be considered by the CWT. A lower [upper_freq] means a 
											higher frequency resolution.
					- show (bool): if True, shows the first generated scalogram with the corresponding signal imprinted on it.
					- overwrite (bool): if True, deletes all the files in the [path] directory before adding the results.
			post: /"""

		if stop==-1:
			stop=self.target.shape[0]

		if overwrite:
			for file in listdir(path):
				remove("{}/{}".format(path,file))
		
		# Save the target vector as it is in the [path] directory.
		self.target.dump('{}/target.mat'.format(path))

		# Determine the wavelet scales corresponding to the frequencies to study.
		scales = pywt.frequency2scale(wavelet,np.linspace(1/self.slice_time,upper_freq,self.data.shape[2])/self.freq)
		
		# Used for projection and positionnal encoding.
		pe = _PatchEncoder(n_patches, projection_dim)

		for sample in tqdm(range(start,stop), desc='Generating scalograms', ascii=False, disable = disable_progress_bar):
			channels=[]
			for channel in range(self.data.shape[1]):
				signal = self.data[sample,channel]
				scalogram, _ = pywt.cwt(signal-np.mean(signal), scales, wavelet)

				# For each channel of the sample, patch the scalogram and project it 
				channels.append(pe(_patching(scalogram,n_patches)))
				
				# Visualize first scalogram and first signal.
				if show and (sample,channel)==(0,0):
					plt.imshow(scalogram)
					plt.plot(np.arange(len(signal)),signal-np.mean(signal)+scalogram.shape[1]//2,color="C1")
					plt.show()
			
			# Save the patched and projected sample in [path]
			np.array(channels).dump("{}/scal_spl{}.mat".format(path,sample+shift+1))
		return

def _patching(scalogram, n_patches):
	"""Splits the [scalogram] into [n_patches]**2 ([n_patches] horizontally and vertically)
		pre : - scalogram (np.ndarray - 2D): the squared scalogram to patch.
			  - n_patches (int): number of vertical and horizontal splits to apply to the scalogram.
		post: - patches (np.ndarray - 2D) array of flattened patches with the first being the top left
				and the last the bottom right.""" 

	size = scalogram.shape[0]//n_patches
	patches = np.empty((n_patches**2, size**2))
	for x_pos in range(n_patches):
		for y_pos in range(n_patches):
			patches[x_pos*n_patches+y_pos] = (scalogram[x_pos*size:(x_pos+1)*size,y_pos*size:(y_pos+1)*size]).flatten()
	return patches

class _PatchEncoder():
# Inspired from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb
	"""Class consisting of two layers :
		- The projection layer projects the input into a random (but fixed) lower dimensional space.
		- The positional embedding layer adds 2D positional encoding to the projected input using sine waves.
		The call function applies both layers in sequential order to an input which is an image cut in patches."""

	def __init__(self, num_patches, projection_dim):
		"""num_patches (int): the number of patches dividing the image horizontally and vertically.
			 projection_dim (int): dimension of the projection space."""
		self.projection = SparseRandomProjection(n_components=projection_dim, random_state=0)
		self.position_embedding = self.positional_encoding(num_patches, num_patches, projection_dim)

	def positional_encoding(self, dim_x, dim_y, depth):
	# Adapted from https://www.tensorflow.org/text/tutorials/transformer
		"""Uses a 2D version of the sine waves positional encoding proposed by Vaswani et al. 2017, which adds
			positional information to patches constituting an image.  
			pre : dim_x (int): the number of patches along the x direction.
					dim_y (int): the number of patches along the y direction.
					depth (int): the number of dimensions of the projection space.
			post: pos_encoding (np.ndarray of tf.float32): array containing the positional encodings to be added 
					to the projected data. The encoding for the patch at position (x,y) in the image is at index 
					x*[dim_x]+y of pos_encoding.""" 

		depth = depth/2

		# Map the x and y positions on the image grid.
		pos_x = np.tile(np.arange(dim_x), dim_y) [:, np.newaxis]
		pos_y = dim_x+np.repeat(np.arange(dim_y), dim_x ) [:, np.newaxis]

		depths = np.arange(depth)[np.newaxis, :]/depth

		angle_rates = 1 / (10000**depths)
		angle_rads_x = pos_x * angle_rates
		angle_rads_y = pos_y * angle_rates

		pos_encoding = np.concatenate(
			[np.sin(angle_rads_x)*np.sin(angle_rads_y), np.cos(angle_rads_x)*np.cos(angle_rads_y)],
			axis=-1) 
		return tf.cast(pos_encoding, dtype=tf.float32)

	def __call__(self, patch):
		encoded = self.projection.fit_transform(patch) + self.position_embedding
		return encoded

def unipolar_to_bipolar(uni_signal, length, reference):
	"""Transfers the unipolar montage of [uni_signal] into a bipolar montage compatible with all possible unipolar montages.
	   pre : uni_signal (dict {str:np.array}): maps each signal to its corresponding electrode.
	   		 length (int): length of the EEG signals.
	   		 reference (string): name of the reference electrode, in this case 'LE' or 'REF'.
	   post: bi_signal (np.ndarray - 2D): The bipolar EEG signals with time on axis 0 and channel on axis 1.
	"""
	bi_signal = np.empty((20,length))

	# Bipolar montage follows the recommandation of The Temple University Hospital EEG Corpus: 
	# Electrode Location and Channel Labels - Annex A (https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/)
	bi_signal[0] = uni_signal["EEG FP1-{}".format(reference)] - uni_signal["EEG F7-{}".format(reference)]
	bi_signal[1] = uni_signal["EEG F7-{}".format(reference)] - uni_signal["EEG T3-{}".format(reference)]
	bi_signal[2] = uni_signal["EEG T3-{}".format(reference)] - uni_signal["EEG T5-{}".format(reference)]
	bi_signal[3] = uni_signal["EEG T5-{}".format(reference)] - uni_signal["EEG O1-{}".format(reference)]
	bi_signal[4] = uni_signal["EEG FP2-{}".format(reference)] - uni_signal["EEG F8-{}".format(reference)]
	bi_signal[5] = uni_signal["EEG F8-{}".format(reference)] - uni_signal["EEG T4-{}".format(reference)]
	bi_signal[6] = uni_signal["EEG T4-{}".format(reference)] - uni_signal["EEG T6-{}".format(reference)]
	bi_signal[7] = uni_signal["EEG T6-{}".format(reference)] - uni_signal["EEG O2-{}".format(reference)]
	bi_signal[8] = uni_signal["EEG T3-{}".format(reference)] - uni_signal["EEG C3-{}".format(reference)]
	bi_signal[9] = uni_signal["EEG C3-{}".format(reference)] - uni_signal["EEG CZ-{}".format(reference)]
	bi_signal[10] = uni_signal["EEG CZ-{}".format(reference)] - uni_signal["EEG C4-{}".format(reference)]
	bi_signal[11] = uni_signal["EEG C4-{}".format(reference)] - uni_signal["EEG T4-{}".format(reference)]
	bi_signal[12] = uni_signal["EEG FP1-{}".format(reference)] - uni_signal["EEG F3-{}".format(reference)]
	bi_signal[13] = uni_signal["EEG F3-{}".format(reference)] - uni_signal["EEG C3-{}".format(reference)]
	bi_signal[14] = uni_signal["EEG C3-{}".format(reference)] - uni_signal["EEG P3-{}".format(reference)]
	bi_signal[15] = uni_signal["EEG P3-{}".format(reference)] - uni_signal["EEG O1-{}".format(reference)]
	bi_signal[16] = uni_signal["EEG FP2-{}".format(reference)] - uni_signal["EEG F4-{}".format(reference)]
	bi_signal[17] = uni_signal["EEG F4-{}".format(reference)] - uni_signal["EEG C4-{}".format(reference)]
	bi_signal[18] = uni_signal["EEG C4-{}".format(reference)] - uni_signal["EEG P4-{}".format(reference)]
	bi_signal[19] = uni_signal["EEG P4-{}".format(reference)] - uni_signal["EEG O2-{}".format(reference)]

	return bi_signal.T

def edf_to_csv(file, montage, n_first_data=None):
	"""Extracts the EEG signal from [file] (which is a .edf file) transfers it to a common bipolar montage and saves
	   it in a temporary .csv file
	   pre : file (pyedflib.EdfReader): open reader of a .edf file.
	   		 montage (string): type of unipolar montage used. Possibilities are '01_tcp_ar', '02_tcp_le', '03_tcp_ar_a',
	   		 				   and '04_tcp_le_a'
	   		 n_first_data (int or None): how many data to copy from each signal from the .edf to the .csv file. Starting 
	   		 							 from the beginning of the signal. Copies the whole signal if None.
	   post: data (np.ndarray - 2D): the data extracted from the .edf file and stored in the .csv file.
	   """
	# Channels names
	channels = file.getSignalLabels()
	# Channels signals
	signals = [file.readSignal(ch, n=n_first_data) for ch in range(len(channels))]
	length = signals[0].shape[0]
	labeled_signal = dict(zip(channels, signals))

	# Change the montage to a common bipolar montage from 4 possible unipolar montages.
	if montage == "02_tcp_le" or montage == "04_tcp_le_a":
		data = unipolar_to_bipolar(labeled_signal, length, "LE")
	if montage == "01_tcp_ar" or montage == "03_tcp_ar_a":
		data = unipolar_to_bipolar(labeled_signal, length, "REF")
	pd.DataFrame(data).to_csv('temp.csv',index=False)
	return data

def bool_files_with_seizure(file_list):
	"""Given a list of .edf file names, recovers the ones where a seizure occurs
	   pre : file_list (array of str): the list of .edf file to check.
	   post: bool_seizure (array of bool): an array with the same length as file_list with True at indexes of 
	         files with seizures and False otherwise."""
	bool_seizure = np.zeros(len(file_list), dtype=bool)
	for i in range(len(file_list)):
		# Find the .csv_bi file attached to the .edf file containing the events information.
		events_filename = file_list[i].replace(".edf", ".csv_bi")
		events = pd.read_csv(events_filename,skiprows = 5)
		# Set the seizure boolean to True of 'seiz' is one of the events listed.
		if 'seiz' in events["label"].unique():
			bool_seizure[i]=True
	return bool_seizure

def pre_ictal_limit(filename, frequency, margin=0):
	"""Given a .csv_bi containing a seizure event, return the number of samples of the first pre-ictal segment
	   (i.e. segment before a seizure that is not a seizure).
	   pre : filename (string): path to a .csv_bi file containing a seizure event.
	   		 frequency (int): the sampling frequency of the signal.
	   		 margin : number of seconds before a seizure to substract from the pre-ictal segment.
	   post: limit (int): number of signal samples belonging to the first pre-ictal segment of the file. Returns 0
	         if the result is negative after margin substraction.
	   """
	events = pd.read_csv(filename,skiprows = 5)
	seiz_start = events[events["label"]=="seiz"].iloc[0]["start_time"]
	return max(int((seiz_start-margin)*frequency),0)




# Pseudo data generation for pre-training ========================================================================

def derangement(length):
	"""Helper function. Provides the indexes for a derangement of an array of length [length].
	   A derangement of an array is a shuffling where no elements are sent back to their original place"""

	if length<2:
		return None
	# There is a 1/e chance for a shuffle to be a derangement. By repeating a shuffle and checking whether it is
	# a derangement, it is possible to obtain one fast.
	for _ in range(200):
		test = np.random.permutation(np.arange(length))
		for a, b in zip(test, np.arange(length)):
			if a == b:
					test[0]=-100
		if test[0]!=-100:
			return test

def gaussian_white_noise_eeg_generator(n_data, real_data, n_channels_affected):
	"""Generates [n_data] EEG data having up to [n_channels_affected] replaced by gaussian white noise.
	   pre : n_data (int): number of data points to generate.
	         real_data (np.ndarray): a sample of actual data to copy the shape from and to infer the 
	         						 white noise amplitude.
	         n_channels_affected (int): maximal number of channels replaced by white noise (random number 
	         							between 1 and [n_channels_affected])
	   post: signal (np.ndarray): pseudo EEG signal with [n_data] elements in the first axis, and the
	         second and third axis have the same shape as [real_data] second and third axis."""

	# Remove the DC component from the real data.
	prep_data = real_data - np.mean(real_data,axis=2,keepdims=True) 
	signal = np.empty((n_data, real_data.shape[1], real_data.shape[2]))
	for data in range(n_data):
		selected = real_data[np.random.randint(0,real_data.shape[0])]
		fake_channels = np.random.randint(0,real_data.shape[1],size=n_channels_affected)
		for channel in fake_channels:
			# Scale tailored such that the white noise has approximately the same amplitude as the real data.
			scale = np.std(prep_data[:,channel])/2
			selected[channel] = scale*np.random.normal(0,1,real_data.shape[2])
		signal[data] = selected
	return signal

def multisine_eeg_generator(frequency, n_data, n_freqs, real_data, n_channels_affected):
	"""Generates [n_data] EEG data having up to [n_channels_affected] replaced by a multisine.
	   pre : frequency (float): sampling frequency of the real_data.
	   		 n_freqs (int): number of frequency needed in the multisine.
	   		 n_data (int): number of data points to generate.
	         real_data (np.ndarray): a sample of actual data to copy the shape from.
	         n_channels_affected (int): maximal number of channels replaced by a multisine (random number 
	         							between 1 and [n_channels_affected])
	   post: signal (np.ndarray): pseudo EEG signal with [n_data] elements in the first axis, and the
	         second and third axis have the same shape as [real_data] second and third axis."""

	# Remove the DC component from the real data.
	prep_data = real_data - np.mean(real_data,axis=2,keepdims=True) 
	signal = np.empty((n_data, real_data.shape[1], real_data.shape[2]))
	for data in range(n_data):
		selected = real_data[np.random.randint(0,real_data.shape[0])]
		fake_channels = np.random.randint(0,real_data.shape[1],size=n_channels_affected)
		for channel in fake_channels:
			scale = np.std(prep_data[:,channel])
			amplitudes = np.random.normal(0,1,n_freqs)
			freqs=np.random.uniform(0,frequency/2,n_freqs)
			# Give different phases to avoid alignment (amplitude highly peaking when waves align)
			phases=np.random.uniform(0,2*np.pi,n_freqs)
			sines = np.array([amplitudes*np.sin(2*np.pi*freqs*t+phases) for t in np.arange(real_data.shape[2])])
			selected[channel] = zscore(np.sum(sines,axis=1))*scale
		signal[data] = selected
	return signal

def channel_swap_intradata(real_data, n_data=None):
	"""Creates [n_data] pseudo-data by shuffling the channel order of data from [real_data].
	   pre : real_data (np.ndarray): real data that will be used to generate pseudo-data by rearranging the channels.
	         n_data (int): number of data points to generate. If None, returns as many data as there are in real_data.
	   post: signal (np.ndarray): pseudo-data where the channel orders are shuffled compared to real data. 
	   		 [n_data] elements in the first axis, and the second and third axis have the same shape as [real_data] 
	   		 second and third axis."""

	if n_data is None:
		n_repeats = 1
		signal=np.empty(real_data.shape)
	else :
		n_repeats = n_data//real_data.shape[0] + 1
		signal=np.empty((n_data + real_data.shape[0],real_data.shape[1],real_data.shape[2]))

	for i in range(n_repeats):
		# Shuffle the data along the channel axis.
		signal[i*real_data.shape[0]:(i+1)*real_data.shape[0]]=real_data[:,np.random.permutation(np.arange(real_data.shape[1])),:]
	
	if n_data is None:
		return signal

	return signal[:n_data]

def channel_swap_interdata(real_data, n_swaps, n_data=None):
	"""Creates [n_data] pseudo-data by interverting some channels between two data points drawn at random in [real_data].
	   pre : real_data (np.ndarray): real data that will be used to generate pseudo-data by rearranging the channels.
	         n_swaps (int): maximum number of channels to be swapped in a pseudo-sample. 
	         n_data (int): number of data points to generate. If None, returns as many data as there are in real_data.
	   post: signal (np.ndarray): pseudo-data consisting a mixture of channels from two real data points. 
	   		 [n_data] elements in the first axis, and the second and third axis have the same shape as [real_data] 
	   		 second and third axis."""

	if n_data is None:
		n_repeats = 1
		signal=np.empty(real_data.shape)
	else :
		n_repeats = n_data//real_data.shape[0] + 1
		signal=np.empty((n_data + real_data.shape[0],real_data.shape[1],real_data.shape[2]))

	for i in range(n_repeats):
		# Matches the data with a derangement of it to generate swaps
		combo1 = real_data[:]
		combo2 = real_data[derangement(real_data.shape[0])]
		for j in range(real_data.shape[0]):
			# Do not swap half channels but only a number between 1 and [n_swaps].
			swapped_elems = np.random.choice(real_data.shape[1], np.random.randint(1,n_swaps+1), replace=False)
			combo1[j,swapped_elems] = combo2[j,swapped_elems]
		signal[i*real_data.shape[0]:(i+1)*real_data.shape[0]]=combo1
	
	if n_data is None:
		return signal

	return signal[:n_data]




# Layers of the model ======================================================================================================

class _EncoderLayer(tf.keras.layers.Layer):
# Inspired from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb
	"""Extends the Layer class from tf.keras. This class implements the architecture of a transformer layer.
		The call function runs the input through the layer"""
	
	def __init__(self, num_heads, projection_dim, transformer_units):
		"""num_heads (int): number of attention heads in the multihead attention layer.
			 projection_dim (int): input second axis dimension. Same as the projection dim of one patch.
			 transformer_units (list of int): dimensions of the feedforward neural network layers."""

		super().__init__()
		self.mlp_depth = len(transformer_units)
		self.mlp_dense = [tf.keras.layers.Dense(units, activation=tf.nn.gelu) for units in transformer_units]
		self.mlp_drop = [tf.keras.layers.Dropout(0.1) for units in transformer_units]

		self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)
		self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.add1 = tf.keras.layers.Add()
		self.add2 = tf.keras.layers.Add()

	def call(self, input):
		"""input (np.ndarray - 2D): the encoder input. The first axis are the patches and the second 
			are the patches projections."""
		# Nomalize input.
		x1 = self.norm1(input)
		# Attention embedding.
		attention_output = self.mha(x1,x1)
		# Add and norm block
		x2 = self.add1([attention_output,input])
		x3 = self.norm2(x2)

		# Feedworward NN
		for i in range(self.mlp_depth):
			x3 = self.mlp_dense[i](x3)
			x3 = self.mlp_drop[i](x3)
		# Skip connection
		output = self.add2([x3,x2])
		return output

class Encoder(tf.keras.layers.Layer):
# Inspired from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb
	"""Extends the Layer class from tf.keras. Stacks [n_layers] _EncoderLayer instance on top of one another. 
		 The call function runs the input through	the multiple layers."""

	def __init__(self, n_layers, num_heads, projection_dim, transformer_units):
		"""n_layers (int): number of _EncoderLayer instance to stack.
			 num_heads (int): number of attention heads in the multihead attention layer.
			 projection_dim (int): input second axis dimension. Same as the projection dim of one patch.
			 transformer_units (list of int): dimensions of the feedforward neural network layers."""

		super().__init__()
		self.n_layers = n_layers
		self.layers = [_EncoderLayer(num_heads, projection_dim, transformer_units) for _ in range(n_layers)]

	def call(self, input):
		"""input (np.ndarray or tf.Tensor - 2D): the encoder input. The first axis are the patches and the second 
			are the patches projections."""
		# The output of layer i-1 is input to the layer i. 
		for i in range(self.n_layers):
			input = self.layers[i](input)
		return input

class ParallelEncoder(tf.keras.layers.Layer):
	"""Extends the Layer class from tf.keras. Instantiates [n_channels] Encoder objects and stores them in an array. 
		 The call function takes each channel	of the input and runs it through its corresponding Encoder. The outputs 
		 are concatenated together."""
	def __init__(self, n_channels, n_layers, num_heads, projection_dim, transformer_units):
		"""n_channels (int): number of channels in the multichannel input signal.
			 n_layers (int): number of _EncoderLayer instance to stack.
			 num_heads (int): number of attention heads in the multihead attention layer.
			 projection_dim (int): input second axis dimension. Same as the projection dim of one patch.
			 transformer_units (list of int): size of the feedforward neural network layers."""

		super().__init__()
		self.n_channels = n_channels
		self.encoders = [Encoder(n_layers, num_heads, projection_dim, transformer_units) for _ in range(n_channels)]

	def call(self, input):
		"""input (np.ndarray - 3D): the encoder input. The first axis are the channels, the second axis are the 
			patches and the third are the patches projections."""
		to_return = []
		for i in range(self.n_channels):
			to_return.append(self.encoders[i](input[:,i]))
		return tf.keras.layers.Concatenate(axis=1)(to_return)

class DecisionLayer(tf.keras.layers.Layer):
# Inspired from https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_with_vision_transformer.ipynb
	"""Extends the Layer class from tf.keras. Implements a MultiLayer Perceptron (MLP) taking the output of a ParallelEncoder
		 object as input. The call function runs the input through the MLP to output [n_classes] probabilities. The ith probability 
		 is the probability of the input to be of class i."""
	
	def __init__(self, mlp_units, n_classes):
		"""mlp_units (list of int): size of the MLP layers. Length of mlp_units is the depth of the MLP
			 n_classes (int): number of output possible classes."""
		super().__init__()
		self.mlp_depth = len(mlp_units)
		self.mlp_dense = [tf.keras.layers.Dense(units, activation=tf.nn.gelu) for units in mlp_units]
		self.mlp_drop = [tf.keras.layers.Dropout(0.5) for units in mlp_units]

		self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.flat = tf.keras.layers.Flatten()
		self.drop = tf.keras.layers.Dropout(0.5)
		self.dense = tf.keras.layers.Dense(n_classes)

	def call(self, input):
		"""input (tf.Tensor - 3D): The MLP input. The first axis are the channels. The second and third axis are the output
			 of one Encoder instance for a projected patch as input."""
		# Normalize and flatten the input. self.drop allows to dropout some weights during training.
		x1 = self.drop(self.flat(self.norm(input)))
		for i in range(self.mlp_depth):
			x1 = self.mlp_dense[i](x1)
			x1 = self.mlp_drop[i](x1)
		return self.dense(x1)




# Data processing for training purposes ===============================================================================================

def load_and_split(path,start,stop):
	"""Loads the scalogram data from [path] and randomly splits it into training and test sets.
		 pre: path (string): path to the data to load.
					start (int): the first scalogram to load (scalograms under that number aren't loaded).
					stop (int): the last scalogram to load (scalograms over that number aren't loaded).
		 post: x_train (np.ndarray): training data. Array of data points with the same shapes as in [path] files.
					 y_train (np.array): training target. Same length as x_train.
					 x_test (np.ndarray): test data. Array of data points with the same shapes as in [path] files.
					 y_test (np.array): test target. Same length as x_test
		 """
	scalograms = np.array([np.load("{}/scal_spl{}.mat".format(path,i+1), allow_pickle = True) for i in range(start,stop)])
	target = np.load("{}/target.mat".format(path), allow_pickle = True)[start:stop]
	return train_test_split(scalograms,target, test_size = 0.05, random_state=4)

def gen_pre_training_data(real_path, fake_path, start, stop_real, stop_fake):
	"""Loads scalogram data from two sets, create a target vector with the first set being labeled 0 and de second 1, shuffles
		 both sets and split the result into training and test.
		 pre: real_path (string): path to the first dataset to load.
			  fake_path (string): path to the second dataset to load.
			  start (int): the first scalogram to load in each dataset (scalograms under that number aren't loaded).
			  stop_real (int): the last scalogram to load in the first dataset (scalograms over that number aren't loaded).
			  stop_fake (int): the last scalogram to load in the second dataset (scalograms over that number aren't loaded).
		 post: x_train (np.ndarray): training data. Array of data points with the same shapes as in [real_path] and [fake_path] files.
			   y_train (np.array): training target. Same length as x_train.
			   x_test (np.ndarray): test data. Array of data points with the same shapes as in [real_path] and [fake_path] files.
			   y_test (np.array): test target. Same length as x_test.
		 """
	scal_real = np.array([np.load("{}/scal_spl{}.mat".format(real_path,i+1), allow_pickle = True) for i in range(start,stop_real)])
	scal_fake = np.array([np.load("{}/scal_spl{}.mat".format(fake_path,i+1), allow_pickle = True) for i in range(start,stop_fake)])
	target = np.concatenate((np.zeros(scal_real.shape[0]),np.ones(scal_fake.shape[0])))
	# Shuffle to avoid having all the data from the first set on one end and of the second set on the other.
	shuffler = np.random.permutation(np.arange(len(target)))

	return train_test_split(np.concatenate((scal_real,scal_fake))[shuffler],target[shuffler], test_size=0.1)
	