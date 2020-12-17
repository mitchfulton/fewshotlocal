import torch
import torch.nn as nn

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        #nn.BatchNorm3d(out_dim),
        nn.InstanceNorm3d(out_dim,affine=True),
        nn.PReLU(),)


def conv_trans_block_3d(in_dim, out_dim):
    return nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, output_padding=0)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        #nn.BatchNorm3d(out_dim),
        nn.InstanceNorm3d(out_dim,affine=True),
        nn.PReLU())


class AE(nn.Module):
    def __init__(self,in_dim,out_dim,num_filters):
        super(AE,self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.PReLU()
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters , activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        
        
        # upsampling
        self.trans_1 = conv_trans_block_3d(self.num_filters,self.num_filters)
        self.up_1 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters,self.num_filters)
        self.up_2 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters,self.num_filters)
        self.up_3 = conv_block_2_3d(self.num_filters , self.num_filters, activation)
        
        self.out = conv_block_3d(self.num_filters, self.out_dim, activation)
        
    def forward(self,x):
        #downsample
        x = self.down_1(x)
        x = self.pool_1(x)
        
        x = self.down_2(x)
        x = self.pool_2(x)
        
        x = self.down_3(x)
        x = self.pool_3(x)
        
        x = self.down_4(x)
        
        #work back up and concat between layers
        x = self.trans_1(x)
        x = self.up_1(x)
        
        x = self.trans_2(x)
        x = self.up_2(x)
        
        x = self.trans_3(x)
        x = self.up_3(x)
        
        x = self.out(x)
        
        return x
        
        
        
        
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 128
  x = torch.Tensor(1, 1, image_size, image_size, image_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = AE(1,2,32)
  
  out = model(x)
  print("out size: {}".format(out.size()))
        
        
        
        
        
        
