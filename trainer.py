import os
import librosa
import time
import warnings
import numpy as np
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataset import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TasNET_trainer(object):
    def __init__(self,
                 TasNET,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 num_epoches=20,
                 clip_norm=False,
                 sr=8000):
        self.TasNET = TasNET
        logger.info("TasNET:\n{}".format(self.TasNET))
        if type(lr) is str:
            lr = float(lr)
            logger.info("Transfrom lr from str to float => {}".format(lr))
        self.optimizer = torch.optim.Adam(
            self.TasNET.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                            'min', factor=0.5, patience=3,verbose=True)
        if torch.cuda.device_count() > 1:
            self.TasNET = nn.DataParallel(self.TasNET, device_ids=[0,1,2,3])
        self.TasNET.to(device)
        self.checkpoint = checkpoint
        self.num_epoches = num_epoches
        self.clip_norm = clip_norm
        self.sr = sr
        self.writer = SummaryWriter('./log')
        if self.clip_norm:
            logger.info("Clip gradient by 2-norm {}".format(clip_norm))

        if not os.path.exists(self.checkpoint):
            os.makedirs(checkpoint)

    def SISNR(self, output, target):
        #output:(128,4000)
        batchsize = np.shape(output)[0]
        target = target.view(batchsize,-1)
        output = output - torch.mean(output,1,keepdim=True)
        target = target - torch.mean(target,1,keepdim=True)

        s_shat = torch.sum(output*target,1,keepdim=True)
        s_2 = torch.sum(target**2,1,keepdim=True)
        s_target = (s_shat / s_2) * target   #(128,4000)

        e_noise = output - s_target    

        return 10*torch.log10(torch.sum(e_noise**2,1,keepdim=True)\
                    /torch.sum(s_target**2,1,keepdim=True))        #(128,1)


    def loss(self,output1,output2,target1,target2):
    	#PIT loss
        loss1 = self.SISNR(output1,target1)+self.SISNR(output2,target2)
        loss2 = self.SISNR(output1,target2)+self.SISNR(output2,target1)
        min = torch.min(loss1, loss2)   #(128,1)
        return torch.mean(min)        #scale

    def train(self, dataloader, epoch):
        self.TasNET.train()
        logger.info("Training...")
        tot_loss = 0
        tot_batch = len(dataloader)
        batch_indx = epoch*tot_batch

        for mix_speech, speech1, speech2 in dataloader:
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                mix_speech= mix_speech.cuda()
                speech1 = speech1.cuda()
                speech2 = speech2.cuda()

            mix_speech = Variable(mix_speech)
            speech1 = Variable(speech1)
            speech2 = Variable(speech2)


            output1, output2 = self.TasNET(mix_speech)
            cur_loss = self.loss(output1,output2,speech1,speech2)
            tot_loss += cur_loss.item()
            
            #write summary
            batch_indx += 1
            self.writer.add_scalar('train_loss', cur_loss, batch_indx)
            cur_loss.backward()
            if self.clip_norm:
                nn.utils.clip_grad_norm_(self.TasNET.parameters(),
                                         self.clip_norm)
            self.optimizer.step()
        return tot_loss / tot_batch, tot_batch

    def validate(self, dataloader, epoch):
        """one epoch"""
        self.TasNET.eval()
        logger.info("Evaluating...")
        tot_loss = 0
        tot_batch = len(dataloader)
        batch_indx = epoch*tot_batch
        #print(tot_batch)

        with torch.no_grad():
            for mix_speech,speech1,speech2 in dataloader:
                if torch.cuda.is_available():
                    mix_speech = mix_speech.cuda()
                    speech1 = speech1.cuda()
                    speech2 = speech2.cuda()

                mix_speech = Variable(mix_speech)
                speech1 = Variable(speech1)
                speech2 = Variable(speech2)

                output1, output2 = self.TasNET(mix_speech)
                cur_loss = self.loss(output1,output2,speech1,speech2)
                tot_loss += cur_loss.item()
                #write summary
                batch_indx += 1
                self.writer.add_scalar('dev_loss', cur_loss, batch_indx)
        return tot_loss / tot_batch, tot_batch

    def run(self, train_set, dev_set):
        init_loss, _ = self.validate(dev_set,0)
        logger.info("Start training for {} epoches".format(self.num_epoches))
        logger.info("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        torch.save(self.TasNET.state_dict(), os.path.join(self.checkpoint, 'TasNET_0.pkl'))
        for epoch in range(1, self.num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch = self.train(train_set, epoch)
            valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set, epoch)
            valid_end = time.time()
            self.scheduler.step(valid_loss)
            logger.info(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) |"
                " dev loss= {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, valid_loss, valid_end - valid_start,
                    valid_num_batch))
            save_path = os.path.join(
                self.checkpoint, "TasNET_{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.TasNET.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(self.num_epoches))
    
    def rerun(self, train_set, dev_set, model_path, epoch_done):
        self.TasNET.load_state_dict(torch.load(model_path))
        init_loss, _ = self.validate(dev_set,epoch_done)
        logger.info("Start training for {} epoches".format(self.num_epoches))
        logger.info("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        torch.save(self.TasNET.state_dict(), os.path.join(self.checkpoint, 'TasNET_0.pkl'))
        for epoch in range(epoch_done, self.num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch = self.train(train_set,epoch)
            valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set,epoch)
            valid_end = time.time()
            self.scheduler.step(valid_loss)
            logger.info(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) |"
                " dev loss= {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, valid_loss, valid_end - valid_start,
                    valid_num_batch))
            save_path = os.path.join(
                self.checkpoint, "TasNET_{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.TasNET.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(self.num_epoches))


