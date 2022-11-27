import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import functional as F



 
    
class LMUFFT_nccl(nn.Module):


    def __init__(self, hidden_size, seq_len, rnn_dropout = 0.5, num_head = 2):

        super(LMUFFT_nccl, self).__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn_dropout = rnn_dropout
        self.num_head = num_head
        
        self.weights_fft = nn.Parameter(torch.randn(self.seq_len//2+1, self.hidden_size,2))
        
        # self.weights_fft = nn.Parameter(torch.randn(self.num_head, self.seq_len//2+1, self.hidden_size // self.num_head,2))
        # self.weights_fft = nn.Parameter(torch.rand(1, self.seq_len//2+1, self.hidden_size // self.num_head,2))
    
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels = self.hidden_size*2 , out_channels = self.hidden_size, kernel_size = 3, padding=  1, groups = 1)# self.num_head)
        
        
        self.lmudropout = torch.nn.Dropout(p=self.rnn_dropout)#dropout_prob
        self.bn_1 = nn.BatchNorm1d(self.seq_len)#,affine = False)
        
        

    def combine_heads_local(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.hidden_size*2)
        return X
    
#     def combine_heads_local(self, X):
#         X = X.permute(3, 2,0,1)
#         X = X.reshape(X.size(0), X.size(1), X.size(2)*X.size(3))
#         return X

#     def split_heads_local(self, X):
#         X = X.reshape(X.size(0), X.size(1), X.size(2)//self.num_head, self.num_head)
#         X = X.permute(2, 3,0,1)
#         return X


    def forward(self, x):
  
        # batch_size, num_head, seq_len, input_size = x.shape
    
        # print(x.shape)
        # torch.Size([8, 8, 2048, 64])
        # input()
        
        # x = x.reshape(batch_size*num_head, seq_len, input_size)
        # fft_input = x.permute(0,1, 3, 2)
        # fft_input  = self.split_heads(x)
        
        
        # x = self.combine_heads_local(x)
        # fft_input =  # [batch_size, 1, seq_len]
        # fft_u = fft.rfft(fft_input, n =  self.seq_len, axis = -1)

        u = torch.mean(x,dim = -1,keepdim = True)
        # print(x.shape)
        
        fft_u = fft.rfft(u, n =  self.seq_len, axis = -2)#,norm = 'ortho')
        fft_u = torch.view_as_real(fft_u)
        # fft_u = fft.rfft2(x, dim=(- 1, - 2),norm = 'ortho')
        # fft_u = torch.view_as_real(fft_u)
                
            
        # print(fft_u.shape, self.weights_fft.unsqueeze(0).shape)
        
        # torch.Size([64, 2048, 33, 2]) torch.Size([1, 64, 1025, 2])
        
        # input()
        temp_real = fft_u[...,0]*self.weights_fft.unsqueeze(0)[...,0] - fft_u[...,1]*self.weights_fft.unsqueeze(0)[...,1]
        temp_image = fft_u[...,0]*self.weights_fft.unsqueeze(0)[...,1] + fft_u[...,1]*self.weights_fft.unsqueeze(0)[...,0]
        
        # print(fft_u.shape,self.weights_fft.unsqueeze(0).unsqueeze(0).shape)
        # input()
        # temp_real = fft_u[:,:,:,:,0]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,0] - fft_u[:,:,:,:,1]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,1]
        # temp_image = fft_u[:,:,:,:,0]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,1] + fft_u[:,:,:,:,1]*self.weights_fft.unsqueeze(0)[:,:,:,:seq_len//2+1,0]
        

        out_ft = torch.cat([temp_real.unsqueeze(-1),temp_image.unsqueeze(-1)],dim =  -1)
        out_ft = torch.view_as_complex(out_ft) # [batch_size, memory_size, seq_len+1]
        m = fft.irfft(out_ft, n =  self.seq_len, axis = -2)
        # m = fft.irfft2(out_ft,  dim=(- 1, - 2),norm = 'ortho')#s = [seq_len,input_size],
        # m = m.permute(0,1, 3, 2) # [batch_size, seq_len, memory_size]
        # m = m.reshape(x.shape[0],fft_input.shape[1],fft_input.shape[3],fft_input.shape[2])
        # m = self.combine_heads(m)
       
        
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        
        # print(input_h.shape)
        # input()
        # input_h = input_h.reshape(batch_size , num_head,  seq_len, input_size)
        # input_h = self.combine_heads_local(input_h)
        h =  self.tiny_conv_linear(input_h.permute(0,2,1)).permute(0,2,1)


        
        h = self.lmudropout(F.elu(self.bn_1(h)))
        # h = self.lmudropout((self.bn_1(h)))
        
        # h = h.reshape(batch_size , num_head,  seq_len, input_size)
        
        
        # print(batch_size, num_head, seq_len, input_size , h.shape)
        # input()
        # h = h.reshape(x.size(0),x.size(1), x.size(2), x.size(3))
        # h = self.combine_heads(h)
        return h#*np.sqrt(self.seq_len  / 2) #h#*np.sqrt(self.seq_len  / 2)   m+m_ori# 
# 