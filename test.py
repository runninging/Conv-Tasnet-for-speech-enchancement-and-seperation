import os
import argparse
import torch
from torch.autograd import Variable
from TasNET_model import TasNET
from utils import parse_yaml, find_files, zero_mean, make_same_length
from dataset import logger
import librosa
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources


def padding(x, length):
	len_x = len(x)
	fac = int(np.floor(len_x/length))
	x_padded = np.zeros((fac+1)*length)
	x_padded[:len_x] = x
	return np.float32(x_padded)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test(args):
    config_dict = parse_yaml(args.config)

    loader_config = config_dict["dataloader"]
    train_config = config_dict["trainer"]
    test_config = config_dict['test']
    data_config = config_dict["data_generator"]
    temp = config_dict["temp"]

    test_path  = test_config['test_load_path']
    test_save_path = test_config["test_save_path"]
    sr = data_config["sr"]
    N_L = data_config["N_L"]
    test_len_time = test_config["test_len_time"]

    #find test dirs
    test_dirs = find_files(os.path.join(test_path,'mix/'))
    test_dirs.sort()
    #load Tasnet model
    tasnet = TasNET()
    tasnet = torch.nn.DataParallel(tasnet, device_ids=[0])
    #tasnet = TasNET.load_model(test_config["test_model_path"])
    tasnet.to(device)
    model_dict = torch.load(test_config["test_model_path"])
    tasnet.load_state_dict(model_dict)
    tasnet.eval()

    logger.info("Testing...")
    #initialize
    num_test = 0
    tot = 0
    low = 0
    sdr_list = []    

    #Start test 
    with torch.no_grad():
        for test_dir in test_dirs:
            name = test_dir.split('/')[-1]
            speech1_dir = os.path.join(test_path,'s1/'+name)
            speech2_dir = os.path.join(test_path,'s2/'+name)

            #load mix, s1 and s2 data
            mix, _ = librosa.load(test_dir,sr)
            real1, _ = librosa.load(speech1_dir,sr)
            real2, _ = librosa.load(speech2_dir,sr)

            #save the mix data in target dir
            save_dir_mix = os.path.join(test_save_path,
            							"mix/"+name)
            #librosa.output.write_wav(save_dir_mix,mix,sr)
            
            #process data before the Tasnet
            len_mix = len(mix)
            mix = make_same_length(mix, N_L)
            mix = np.reshape(mix, [1,-1,N_L])

            #Separate mix audio with Tasnet
            mix = torch.from_numpy(mix)        
            if torch.cuda.is_available():
            	mix = mix.cuda()
            mix = Variable(mix) 
            speech1,speech2 = tasnet(mix)

            #translate the output to numpy in cpu	
            wave1 = speech1.to(torch.device("cpu"))
            wave2 = speech2.to(torch.device("cpu"))
            wave1 = wave1.view(-1,)
            wave2 = wave2.view(-1,)
            wave1 = zero_mean(wave1[:len_mix].numpy())/np.max(wave1[:len_mix].numpy())
            wave2 = zero_mean(wave2[:len_mix].numpy())/np.max(wave2[:len_mix].numpy())

            #Calculate the SDR with bss tools
            wave = [wave1,wave2]
            estimate = np.array(wave)
            real = [real1,real2]
            reference = np.array(real)
            sdr,sir,sar,_ = bss_eval_sources(estimate,reference) 
            sdr_list.append(np.mean(sdr))

            #Count the number of SDR lower than 5 and calculate the mean SDR
            if np.mean(sdr) < 5:
                low +=1
            num_test += 1
            tot += sdr
            mean = np.mean(tot)/(num_test)

            #Save the separated audio in the target dir
            save_dir1 = os.path.join(test_save_path,
            						"s1/"+name)
            save_dir2 = os.path.join(test_save_path,
            						"s2/"+name)
            #librosa.output.write_wav(save_dir1,wave1,sr)
            #librosa.output.write_wav(save_dir2,wave2,sr)
            
            if num_test%10 == 0:
                logger.info("The current SDR was {}/{}".format(mean, num_test))
                logger.info("SDR lower than 5 were {}/{}".format(low,num_test))

    #Print the SDR in the figure
    logger.info("Testing for all {} waves have done!".format(num_test))
    logger.info("The total mean SDR is {}".format(mean))
    xData = np.arange(1, len(sdr_list)+1, 1)
    sdr_list.sort()  
    yData = sdr_list
    plt.figure(num=1, figsize=(8, 6))
    plt.title('SDR of test samples', size=14)
    plt.xlabel('index', size=14)
    plt.ylabel('SDR', size=14)
    print(yData)
    plt.plot(xData, yData, color='b', linestyle='--', marker='o')
    plt.savefig('plot.png', format='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TasNET by PyTorch ")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    test(args)


 
