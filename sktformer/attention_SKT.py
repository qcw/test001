"""
--------------------------------------
"""

import torch
import torch.nn as nn
import math
import pdb

import dct
import lmu 

class DynamicAttention_SKTAttention(nn.Module):
    def __init__(self, config, maybe = 0):
        super().__init__()
        
        
        assert not (config.pooling_mode.lower() == 'cls' and config.cls_token)
        self.cls_from_seq = config.pooling_mode.lower() == 'cls'


        self.num_head = config.num_head
        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.seq_len = config.max_seq_len
        self.dp_rank = config.num_landmarks
        
        
        self.ln_1 = nn.LayerNorm(self.num_head * self.head_dim)
        self.ln_2 = nn.LayerNorm(self.num_head * self.head_dim)
        # self.ln_3 = nn.LayerNorm(self.num_head * self.head_dim)
        
        # self.ln_1 = nn.BatchNorm1d(self.seq_len )
        # self.ln_2 = nn.BatchNorm1d(self.seq_len )
        # self.ln_3 = nn.BatchNorm1d(self.seq_len )
        # self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        # self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        # self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        
        

        self.index_set_right =   torch.randperm(self.head_dim)
        self.index_set_right = self.index_set_right[:self.dp_rank] 
        
        # self.index_set_right =  [i*torch.log(self.head_dim)/self.dp_rank for i in self.index_set_right]
        # self.index_set_right = [torch.round(i) for i in self.index_set_right]
        # self.index_set_right = [i*(self.head_dim//(self.dp_rank - 1)) for i in range(self.dp_rank - 1)]
        # self.index_set_right.append(self.head_dim - 1)
        
        #random fourier r log(n)
        #sub fourier sqrt{r}log(n)
        
        self.index_set_left =   torch.randperm(self.seq_len)
        self.index_set_left = self.index_set_left[:self.dp_rank]
        
#         self.index_set_left = [i*(self.seq_len//(self.dp_rank - 1)) for i in range(self.dp_rank - 1)]
#         self.index_set_left.append(self.seq_len - 1)
        
        
        
        
        
#         self.tiny_conv_k =  torch.nn.Conv1d(in_channels = self.head_dim*self.num_head*2, out_channels = self.head_dim*self.num_head, kernel_size =self.dp_rank +  1, padding= self.dp_rank // 2,groups =  1)
#         self.weights_fft_real_k = nn.Parameter(torch.rand(self.seq_len,self.head_dim*self.num_head))
#         self.weights_fft_imag_k = nn.Parameter(torch.rand(self.seq_len,self.head_dim*self.num_head))
        
#         self.tiny_conv_v =  torch.nn.Conv1d(in_channels = self.head_dim*self.num_head*2, out_channels = self.head_dim*self.num_head, kernel_size =self.dp_rank +  1, padding= self.dp_rank // 2,groups =  1)
#         self.weights_fft_real_v = nn.Parameter(torch.rand(self.seq_len,self.head_dim*self.num_head))
#         self.weights_fft_imag_v = nn.Parameter(torch.rand(self.seq_len,self.head_dim*self.num_head))
# #         self.linear_dct1 = dct.LinearDCT(self.seq_len)
# #         self.linear_dct2 = dct.LinearDCT(self.head_dim)
        

        
#         # self.tiny_conv =  torch.nn.Conv1d(in_channels = self.num_head * self.head_dim, out_channels = self.num_head * self.head_dim, kernel_size = 16 + 1, padding=8, groups = self.num_head * self.head_dim)
        
# #         self.W = torch.nn.Parameter(torch.randn(self.num_head,self.seq_len,self.head_dim))
# #         self.W.requires_grad = True
        
        
# #         self.linear_idct1 = dct.LinearDCT(self.seq_len,type = 'idct')
# #         self.linear_idct2 = dct.LinearDCT(self.head_dim,type = 'idct')
        
#         if config.cls_token is True:
#             self.lmu_fft_k =lmu.LMUFFT_nccl(self.num_head * self.head_dim, self.num_head * self.head_dim, self.num_head * self.head_dim, self.seq_len+1, theta = self.seq_len+1 ,learnable = True,modes1 =32 )
#             self.lmu_fft_v =lmu.LMUFFT_nccl(self.num_head * self.head_dim, self.num_head * self.head_dim, self.num_head * self.head_dim, self.seq_len+1, theta = self.seq_len+1 ,learnable = True,modes1 =32)

#         else:
#             self.lmu_fft_k =lmu.LMUFFT_nccl(self.num_head * self.head_dim, self.num_head * self.head_dim, self.num_head * self.head_dim, self.seq_len, theta = self.seq_len ,learnable = True,modes1 =32)
#             self.lmu_fft_v =lmu.LMUFFT_nccl(self.num_head * self.head_dim, self.num_head * self.head_dim, self.num_head * self.head_dim, self.seq_len, theta = self.seq_len ,learnable = True,modes1 =32)

#         self.reversed_list = [self.seq_len - 1 - i for i in range(self.seq_len)]
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

       
        
    def forward(self,X, Q, K, V, mask,cls_embed=None):
                
        
        K = K * mask[:, None, :, None]
        V = V * mask[:,None, :, None]
                
        if cls_embed is not None:
            Q = torch.cat([self.split_heads(cls_embed),Q],dim = 2)
            K = torch.cat([self.split_heads(cls_embed),K],dim = 2)
            V = torch.cat([self.split_heads(cls_embed),V],dim = 2)
        # print('error-----------')
        # print(self.linear_dct2,self.linear_dct1,K.shape)
        # print(K.shape)
        # K = self.split_heads(self.large_conv(self.combine_heads(K).transpose(-1,-2)).transpose(-1,-2))
        # V = self.split_heads(self.large_conv(self.combine_heads(V).transpose(-1,-2)).transpose(-1,-2))
        # print(K.shape)
        # K = self.linear_dct1(self.linear_dct2(K).transpose(-1,-2)).transpose(-1,-2)+ K
        # V = self.linear_dct1(self.linear_dct2(V).transpose(-1,-2)).transpose(-1,-2)+ V
        # K_reverse = K[:,:,self.reversed_list,:]
        # K_reverse = self.split_heads(self.lmu_fft(self.combine_heads(K_reverse)))
        # K_reverse = K_reverse[:,:,self.reversed_list,:]
        
        
        # V_reverse = V[:,:,self.reversed_list,:]
        # V_reverse = self.split_heads(self.lmu_fft(self.combine_heads(V_reverse)))
        # V_reverse = V_reverse[:,:,self.reversed_list,:]
        # K = self.split_heads(self.lmu_fft_k(self.combine_heads(K))) #+ K_reverse
        # V = self.split_heads(self.lmu_fft_v(self.combine_heads(V))) #+ V_reverse
        # print(K.shape,self.W.shape)
        # K = self.W*K
        # V =V * self.W
        # K = self.linear_idct2(self.linear_idct1(K).transpose(-1,-2)).transpose(-1,-2)
        # V = self.linear_idct1(self.linear_idct1(V).transpose(-1,-2)).transpose(-1,-2)
        # print(K.shape)
        # K = self.split_heads(self.ln_1(self.combine_heads(K)))
        # V = self.split_heads(self.ln_3(self.combine_heads(V)))
        # K = dct.dct_real(dct.dct_real(K,dim = -1),dim = -2)
        # V = dct.dct_real(dct.dct_real(V,dim = -1),dim = -2)
        
        # error = torch.abs(Ktmp - K)
        # print('error: ', error.max())
        
        # linear_dct = LinearDCT(4096, 'dct')
# error = torch.abs(dct(x) - linear_dct(x))
# assert error.max() < 1e-3, (error, error.max())
        # K = dct.dct_real(K,dim = -2)
        # V = dct.dct_real(V,dim = -2)
#         K = self.combine_heads(K)
        
#         K1 = dct.dct_real(K,dim = -2)        
#         K1[:,self.dp_rank:,:] =0
        
#         K = torch.cat([dct.idct_real(K1,dim = -2), K], dim = -1)
#         K = self.tiny_conv1(K.permute(0,2,1)).permute(0,2,1)
#         K = self.split_heads(K)
        
#         V = self.combine_heads(V)
        
#         V1 = dct.dct_real(V,dim = -2)        
#         V1[:,self.dp_rank:,:] =0
        
#         V = torch.cat([dct.idct_real(V1,dim = -2), V], dim = -1)
        
#         V = self.tiny_conv2(V.permute(0,2,1)).permute(0,2,1)
#         V = self.split_heads(V)
#         K_feq = torch.fft.fft(K,n=self.seq_len ,axis = -2)
#         Knew = K_feq.real*self.weights_fft_real_k.unsqueeze(0) - K_feq.imag * self.weights_fft_imag_k.unsqueeze(0)
#         # Xnew[:,:]
#         K = torch.cat([Knew, K], dim = -1) 
#         K = self.tiny_conv_k(K.permute(0,2,1)).permute(0,2,1)

#         V_feq = torch.fft.fft(V,n=self.seq_len ,axis = -2)
#         Vnew = V_feq.real*self.weights_fft_real_v.unsqueeze(0) - V_feq.imag * self.weights_fft_imag_v.unsqueeze(0)
#         # Xnew[:,:]
#         V = torch.cat([Vnew, V], dim = -1) 
#         V = self.tiny_conv_v(V.permute(0,2,1)).permute(0,2,1)
        
        if self.dp_rank <= self.seq_len:
            K1 = K[:,:,self.index_set_left,:]
            V1 = V[:,:,self.index_set_left,:]
        else:
            K1 = K
            V1 = V

            
        # batch, head_number, seq_len, hidden_dim
        dots = Q @ K1.transpose(-1,-2)  
        # batch, head_number, seq_len, sub_seq_len
        # batch, head_number, sub_seq_len, hiddem_dim
        dots = dots / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(dots,dim=-1)
        attn = self.drop_attn(attn)
        
        #### right part ####        
        Q2 = Q.transpose(-1,-2)
        # V = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,V1))))
        # K2 = dct.dct_real(K,dim = -1)
        # V2 = dct.dct_real(V,dim = -1)
        # V = torch.matmul(attn,V1)
        if self.dp_rank <= self.head_dim:

            K2 = K[:,:,:,self.index_set_right]
            V2 = V[:,:,:,self.index_set_right]
        else:
            K2 = K
            V2 = V
    
        dots_r = Q2 @ K2
        dots_r = dots_r / math.sqrt(self.seq_len)
        attn_r = nn.functional.softmax(dots_r,dim=-1).transpose(-1,-2)
        attn_r = self.drop_attn(attn_r)

        X =self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,V1))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V2,attn_r))))/2
        # X =  torch.matmul(attn,V1)/2 + torch.matmul(V2,attn_r)/2
        
        
        
        
#         K = self.linear_dct1(self.linear_dct2(K).transpose(-1,-2)).transpose(-1,-2)#+ K
#         V = self.linear_dct1(self.linear_dct2(V).transpose(-1,-2)).transpose(-1,-2)#+ V
#         # print(K.shape,self.W.shape)
#         # K = self.W*K
#         # V =V * self.W
#         # K = self.linear_idct2(self.linear_idct1(K).transpose(-1,-2)).transpose(-1,-2)
#         # V = self.linear_idct1(self.linear_idct1(V).transpose(-1,-2)).transpose(-1,-2)
#         # print(K.shape)
#         # K = self.split_heads(self.ln_1(self.combine_heads(K)))
#         # V = self.split_heads(self.ln_3(self.combine_heads(V)))
#         # K = dct.dct_real(dct.dct_real(K,dim = -1),dim = -2)
#         # V = dct.dct_real(dct.dct_real(V,dim = -1),dim = -2)
        
#         # error = torch.abs(Ktmp - K)
#         # print('error: ', error.max())
        
#         # linear_dct = LinearDCT(4096, 'dct')
# # error = torch.abs(dct(x) - linear_dct(x))
# # assert error.max() < 1e-3, (error, error.max())
#         # K = dct.dct_real(K,dim = -2)
#         # V = dct.dct_real(V,dim = -2)
#         if self.dp_rank <= self.seq_len:
#             K1 = K[:,:,self.index_set_left,:]
#             V1 = V[:,:,self.index_set_left,:]
#         else:
#             K1 = K
#             V1 = V

            
#         # batch, head_number, seq_len, hidden_dim
#         dots = Q @ K1.transpose(-1,-2)  
#         # batch, head_number, seq_len, sub_seq_len
#         # batch, head_number, sub_seq_len, hiddem_dim
#         dots = dots / math.sqrt(self.head_dim)
#         attn = nn.functional.softmax(dots,dim=-1)
#         attn = self.drop_attn(attn)
        
#         #### right part ####        
#         Q2 = Q.transpose(-1,-2)
#         # V = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,V1))))
#         # K2 = dct.dct_real(K,dim = -1)
#         # V2 = dct.dct_real(V,dim = -1)
#         # V = torch.matmul(attn,V1)
#         if self.dp_rank <= self.head_dim:

#             K2 = K[:,:,:,self.index_set_right]
#             V2 = V[:,:,:,self.index_set_right]
#         else:
#             K2 = K
#             V2 = V
    
#         dots_r = Q2 @ K2
#         dots_r = dots_r / math.sqrt(self.seq_len)
#         attn_r = nn.functional.softmax(dots_r,dim=-1).transpose(-1,-2)
#         attn_r = self.drop_attn(attn_r)

#         X =  X + self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,V1))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V2,attn_r))))/2
        
        
        
        
        # X = self.split_heads(self.ln_2(self.combine_heads(torch.matmul(V2,attn_r))))
        # X =  torch.matmul(attn,V1)/2 + torch.matmul(V2,attn_r)/2 
        # self.split_heads(self.ln_3(self.combine_heads(X)))/3+ 
        #self.split_heads(self.ln_3(self.combine_heads(X))) +  
        
        
        ### QKV
        ### QK_1^T(V_1, V_2)Q^TK_2
        # batch, head_number, 1024, hidden_dim, sub_seq_len = 8, 16
        
        #32*32  1024, -> (1024,3) -> W -> Q,K,V (1024,3)
        
        #Q(1024,3), K,V FFT(1024,3), Q ->lookup-> KV -> <softmax(<Q,K>).V>
        
        #Q(1024,3), K,V DCT(64,3), DCT(1024,3), Q ->lookup-> KV -> <softmax(<Q,K>).V>
            
            
        #Q CUR, C,R, O(rlog(n))
        
        return X


