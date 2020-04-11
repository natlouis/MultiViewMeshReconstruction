#Incorporate Recurrent 3D Convlutional LSTM from 3D-R2N2
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, get_norm
from torch import nn
from torch.nn import functional as F

from torch.nn import Linear, LeakyReLU, Sigmoid, Tanh
from r2n2_lib.layers import FCConv3DLayer_torch, Unpool3DLayer, \
                       SoftmaxWithLoss3D, BN_FCConv3DLayer_torch

class RecurrentVoxelHead(nn.Module):
    def __init__(self, cfg):
        super(RecurrentVoxelHead, self).__init__()

        # fmt: off
        self.voxel_size = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE #48
        conv_dims       = cfg.MODEL.VOXEL_HEAD.CONV_DIM #256
        num_conv        = cfg.MODEL.VOXEL_HEAD.NUM_CONV #4
        input_channels  = cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS #resnet50 - 2048
        self.norm       = cfg.MODEL.VOXEL_HEAD.NORM
        # fmt: on

        assert self.voxel_size % 2 == 0

        self.conv_norm_relus = []
        # From ResNet backbone feature extractor
        prev_dim = input_channels

        # Recurrent layers
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        #define the FCConv3DLayers in 3d convolutional gru unit
        #self.n_convfilter = [96, 128, 256, 256, 256, 256]


        #number of filters for each 3d convolution layer in the decoder
        #self.n_deconvfilter = [128, 128, 128, 64, 32, 2]

        self.input_shape  = None #unused 
        self.n_gru_vox    = 4
        self.n_fc_filters = [1024] #the filter shape of the 3d convolutional gru unit
        self.n_h_feat  = 128 #number of features for output tensor

        self.h_shape      = (self.batch_size, self.n_h_feat, self.n_gru_vox, self.n_gru_vox, self.n_gru_vox) #the size of the hidden state
        self.conv3d_filter_shape = (self.n_h_feat, self.n_h_feat, 3, 3, 3) #the filter shape of the 3d convolutional gru unit

        self.recurrent_layer = recurrent_layer(self.input_shape, input_channels, \
                                 self.n_fc_filters, self.h_shape, self.conv3d_filter_shape)

        self.reduce_dim = Conv2d(
                self.n_h_feat*self.n_gru_vox,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
        '''
        for k in range(num_conv):
            conv = Conv2d(
                prev_dim,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("voxel_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            prev_dim = conv_dims

        '''

        # Deconvolutional layers
        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.predictor = Conv2d(conv_dims, self.voxel_size, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for voxel prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        V = self.voxel_size

        '''
        x = F.interpolate(x, size=V // 2, mode="bilinear", align_corners=False)
        for layer in self.conv_norm_relus:
            x = layer(x)
        '''

        #initialize the hidden state and update gate
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)

        #a list used to store intermediate update gate activations
        u_list = []

        """
        x is the input and the size of x is (num_views, batch_size, feature_size).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network. 
        """
        #TODO: Need x to be a list of features one for each view
        x = x.unsqueeze(0) #NOTE: Temporarily add single time dimension

        for time in range(x.size(0)):
            gru_out, update_gate = self.recurrent_layer(x[time], h, u, time)
            
            h = gru_out
            
            u = update_gate
            #u_list.append(u)
        
        x = h

        '''
        R2N2 outputs a 5-D tensor: (batch_size, z, f/g, x, y)
        z,x,y serve as the occupancy probabilities for each grid position, and 
        f/g is whether that occupancy belongs to foreground or background.
        
        MeshRCNN doesn't output f/g, just occupancy prob on for all grid positions.
        This means that creating a 3D convolutional LSTM in R2N2 could've been unnecessary.

        - Nate 
        ''' 
        #Flatten from 5-D to 4-D.
        x = x.flatten(1,2) 

        #Linearly interpolate spatial dimension to 24 x 24
        x = F.interpolate(x, size=V // 2, mode="bilinear", align_corners=False)
        x = self.reduce_dim(x)

        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        return x

    def initHidden(self, h_shape):
        h = torch.zeros(h_shape)
        if torch.cuda.is_available():
            h = h.type(torch.cuda.FloatTensor)
        return h

class recurrent_layer(nn.Module):
    def __init__(self, input_shape, input_channels, \
                 n_fc_filters, h_shape, conv3d_filter_shape):
        super(recurrent_layer, self).__init__()

        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        
        #find the input feature map size of the fully connected layer
        #fc7_feat_w, fc7_feat_h = self.fc_in_featmap_size(input_shape, num_pooling=6)
        fc7_feat_w, fc7_feat_h = 5,5

        #define the fully connected layer
        self.fc7 = Linear(int(input_channels * fc7_feat_w * fc7_feat_h), n_fc_filters[0])
        
        
        #define the FCConv3DLayers in 3d convolutional gru unit
        #conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        self.t_x_s_update = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_s_reset = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        self.t_x_rs = BN_FCConv3DLayer_torch(n_fc_filters[0], conv3d_filter_shape, h_shape)
        
    def forward(self, x, h, u, time):
        #for recurrent batch normalization, time is the current time step
        """
        x is the input and the is the output of the encoder (ResNet).
        h and u is the hidden state and activation of last time step respectively.
        This function defines the forward pass of the recurrent layer of the network.
        """

        x = x.view(x.size(0), -1) #flatten output from encoder

        fc7 = self.fc7(x)
        rect7 = self.leaky_relu(fc7)
        
        t_x_s_update = self.t_x_s_update(rect7, h, time)
        t_x_s_reset = self.t_x_s_reset(rect7, h, time)
        
        update_gate = self.sigmoid(t_x_s_update)
        complement_update_gate = 1 - update_gate
        reset_gate = self.sigmoid(t_x_s_reset)
        
        rs = reset_gate * h
        t_x_rs = self.t_x_rs(rect7, rs, time)
        tanh_t_x_rs = self.tanh(t_x_rs)
        
        gru_out = update_gate * h + complement_update_gate * tanh_t_x_rs
        
        return gru_out, update_gate
        
    #infer the input feature map size, (height, width) of the fully connected layer
    def fc_in_featmap_size(self, input_shape, num_pooling):
        #fully connected layer
        img_w = input_shape[2]
        img_h = input_shape[3]
        #infer the size of the input feature map of the fully connected layer
        fc7_feat_w = img_w
        fc7_feat_h = img_h
        for i in range(num_pooling):
            #image downsampled by pooling layers
            #w_out= np.floor((w_in+ 2*padding[0]- dilation[0]*(kernel_size[0]- 1)- 1)/stride[0]+ 1)
            fc7_feat_w = np.floor((fc7_feat_w + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
            fc7_feat_h = np.floor((fc7_feat_h + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
        return fc7_feat_w, fc7_feat_h
