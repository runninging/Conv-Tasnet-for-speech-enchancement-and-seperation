import librosa
import numpy as np
import fnmatch
import os
import random
import time
import sklearn.utils as sku
import scipy.signal
import numpy.random
from scipy.io import wavfile
import argparse
import yaml
import torch
from sklearn import preprocessing
import math
def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, pattern[1]):
            files.append(os.path.join(root, filename))
    return files

def find_dir(directory, pattern):
	dir = []
	for root, dirnames, filenames in os.walk(directory):
		if root.split('/')[-2] == pattern:
			dir.append(root)
	return dir		
	#print(dir)


def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find config file...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict

def norm_audio(audiofiles, noisefiles):
	'''Normalize the audio files
	used before training using a independent script'''
	for file in audiofiles:
		audio, sr = librosa.load(file, sr=16000)
		div_fac = 1 / np.max(np.abs(audio)) / 3.0
		audio = audio * div_fac
		librosa.output.write_wav(file, audio, sr)
	for file in noisefiles:
		audio, sr = librosa.load(file, sr=16000)
		div_fac = 1 / np.max(np.abs(audio)) / 3.0
		audio = audio * div_fac
		librosa.output.write_wav(file, audio, sr)

def mix(speech1,speech2,SNR):
    len1 = len(speech1)
    len2 = len(speech2)
    tot_len = max(len1, len2)
    
    if len1 < len2:
        rep = int(np.floor(len2 / len1))
        left = len2 - len1 * rep
        temp_audio = np.tile(speech1, [1, rep])
        temp_audio.shape = (temp_audio.shape[1],)
        speech1 = np.hstack((temp_audio, speech1[:left]))
        speech2 = np.array(speech2)
    else:
        rep = int(np.floor(len1 / len2))
        left = len1 - len2 * rep
        temp_noise = np.tile(speech2, [1, rep])
        temp_noise.shape = (temp_noise.shape[1],)
        speech2 = np.hstack((temp_noise, speech2[:left]))
        speech1 = np.array(speech1)
        
    fac = np.linalg.norm(speech1)/np.linalg.norm(speech2)/(10**(SNR*0.05))
    speech2 *= fac
    mix_speech = speech1 + speech2
    return mix_speech, speech1, speech2
	
def normalize_mean(x):
	return (x-x.mean())/x.std()

def unnormalize(x1,x2):
	return x1*x2.std()+x2.mean()

def zero_mean(x):
	return x - x.mean()

def l2_norm(x):
	return preprocessing.scale(x,axis=1,with_mean=False,with_std=True)

def make_same_length(speech,len_ref):
	len_ref = int(len_ref)
	len_s = len(speech)
	if len_s < len_ref:
		rep = int(np.floor(len_ref / len_s))
		left = len_ref - len_s * rep
		temp_speech = np.tile(speech, [1, rep])
		temp_speech.shape = (temp_speech.shape[1],)
		speech = np.hstack((temp_speech, speech[:left]))
	else:
		rep = int(np.floor(len_s / len_ref))
		add = len_ref * (rep+1) - len_s
		speech = np.hstack((speech, speech[:add]))
	return speech

def padlast(x, len_ref):
	len_x = len(x)
	fac = int(np.floor(len_x/len_ref))
	x_padlast = np.zeros((fac+1)*len_ref)
	x_padlast[:fac*len_ref] = x[:fac*len_ref]
	x_padlast[fac*len_ref:] = x[(len_x-len_ref):]
	return np.float32(x_padlast)

def padding(x, length):
	len_x = len(x)
	fac = int(np.floor(len_x/length))
	x_padded = np.zeros((fac+1)*length)
	x_padded[:len_x] = x
	return np.float32(x_padded)
        

class speech_preprocess(object):
	def __init__(self,
				 speech_dir,
				 train_save_path,
				 dev_save_path,
				 test_save_path,
				 num_data,
				 sr=8000,
				 N_L=40,
				 len_time=0.5,
				 is_norm=True):
		self.speech_dir = speech_dir
		self.train_save_path = train_save_path
		self.dev_save_path = dev_save_path
		self.test_save_path = test_save_path
		self.num_data = num_data
		self.sr = sr
		self.N_L = N_L
		self.len_time = len_time
		self.is_norm = is_norm

	def speech_segment(self,mix_dir,speech1_dir,speech2_dir):
		mix, _ = librosa.load(mix_dir, self.sr)
		speech1, _ = librosa.load(speech1_dir, self.sr)
		speech2, _ = librosa.load(speech2_dir, self.sr)

		len_mix = len(mix)
		len_ref = int(self.len_time*self.sr)

		if len_mix < len_ref:
			mix = make_same_length(mix, len_ref)
			speech1 = make_same_length(speech1,len_ref)
			speech2 = make_same_length(speech2,len_ref)
		else:
			mix = padlast(mix, len_ref)
			speech1 = padlast(speech1,len_ref)
			speech2 = padlast(speech2,len_ref)
		len_tot = len(mix)
		mix = np.reshape(mix, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech1 = np.reshape(speech1, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech2 = np.reshape(speech2, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		return mix, speech1, speech2

	def save_speech(self,mix_speech,speech1,speech2,save_path):
		num_speech = mix_speech.shape[0] 
		for ind in range(num_speech):
			np.savez(save_path+'_'+str(ind)+".npz",
			 		 mix_speech=mix_speech[ind,:,:],
			 		 speech1=speech1[ind,:,:],
			 		 speech2=speech2[ind,:,:])

	def data_generator(self):
		train_data_dir = os.path.join(self.speech_dir, "tr/mix/")
		train_mix_dirs = find_files(train_data_dir)
		print(train_data_dir)
		dev_data_dir = os.path.join(self.speech_dir, "cv/mix/")
		dev_mix_dirs = find_files(dev_data_dir)
		test_data_dir = os.path.join(self.speech_dir, "tt/mix/")
		test_mix_dirs = find_files(test_data_dir)

		print('#####generate train_data######')
		print(len(train_mix_dirs))
		ind = 0
		for mix_dir in train_mix_dirs:
			if ind%1000 == 0:
				print("{}/{} have done".format(ind,20000))
			wavname = mix_dir.split('/')[-1]
			speech1_dir = os.path.join(self.speech_dir, "tr/s1/" + wavname)
			speech2_dir = os.path.join(self.speech_dir, "tr/s2/" + wavname)
			mix, speech1, speech2 = self.speech_segment(mix_dir, speech1_dir, speech2_dir)
			input_path = os.path.join(self.train_save_path, wavname)
			self.save_speech(mix,speech1,speech2,input_path)
			ind += 1

		print('#####generate dev_data######')

		print(len(dev_mix_dirs))
		ind = 0
		for mix_dir in dev_mix_dirs:
			if ind%1000 == 0:
				print("{}/{} have done".format(ind,5000))
			wavname = mix_dir.split('/')[-1]
			speech1_dir = os.path.join(self.speech_dir, "cv/s1/" + wavname)
			speech2_dir = os.path.join(self.speech_dir, "cv/s2/" + wavname)
			mix, speech1, speech2 = self.speech_segment(mix_dir, speech1_dir, speech2_dir)
			input_path = os.path.join(self.dev_save_path, wavname)
			self.save_speech(mix,speech1,speech2,input_path)
			ind += 1

		print('#####generate test_data######')
		print(len(test_mix_dirs))
		ind = 0
		for mix_dir in test_mix_dirs:
			if ind%1000 == 0:
				print("{}/{} have done".format(ind,3000))
			wavname = mix_dir.split('/')[-1]
			speech1_dir = os.path.join(self.speech_dir, "tt/s1/" + wavname)
			speech2_dir = os.path.join(self.speech_dir, "tt/s2/" + wavname)			
			mix, speech1, speech2 = self.speech_segment(mix_dir, speech1_dir, speech2_dir)
			input_path = os.path.join(self.test_save_path, wavname)
			self.save_speech(mix,speech1,speech2,input_path)	
			ind += 1	
		
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TasNet by PyTorch ")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    config_dict = parse_yaml(args.config)
    data_config = config_dict["data_generator"]
    processor = speech_preprocess(**data_config)
    processor.data_generator()
