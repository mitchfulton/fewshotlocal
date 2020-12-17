import torch
import torch.nn as nn

def conv_block_3d(in_dim, out_dim, activation,stride=1):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        #nn.BatchNorm3d(out_dim),
        nn.InstanceNorm3d(out_dim,affine=True),
        nn.PReLU(),)


def conv_trans_block_3d(in_dim, out_dim,stride=2):
    return nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=stride, padding=0, output_padding=0)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation,stride=1):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation,stride),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        #nn.BatchNorm3d(out_dim),
        nn.InstanceNorm3d(out_dim,affine=True),
        nn.PReLU())


class UNet(nn.Module):
    def __init__(self,in_dim,out_dim,num_filters):
        super(UNet,self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.PReLU()
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation,stride=2)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters , activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters , self.num_filters , activation)
        
        
        # upsampling
        self.trans_1 = conv_trans_block_3d(self.num_filters,self.num_filters)
        self.up_1 = conv_block_2_3d(self.num_filters*2 , self.num_filters , activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters,self.num_filters)
        self.up_2 = conv_block_2_3d(self.num_filters*2 , self.num_filters , activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters,self.num_filters,stride=2)
        self.up_3 = conv_block_2_3d(self.num_filters*2 , self.num_filters, activation)
        
        self.out = conv_block_3d(self.num_filters, self.out_dim, activation)
        
    def forward(self,x):
        #downsample
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        
        down_4 = self.down_4(pool_3)
        
        #work back up and concat between layers
        trans_1 = self.trans_1(down_4)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)
        
        out = self.out(self.trans_3(up_3))
        
        return out
        
        
        
        
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 128
  x = torch.Tensor(1, 1, image_size, image_size, image_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(8)
  
  out = model(x)
  print("out size: {}".format(out.size()))
        
        
        
        
        
        
