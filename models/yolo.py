import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.loss import SigmoidBin

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class HorizonDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, ch=()):  # detection layer
        super(HorizonDetect, self).__init__()
        self.no = 2  # number of outputs (y-intercept and slope)
        self.nl = len(ch)  # number of detection layers (expected to be 1)
        # Add Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output conv layer
        self.m = nn.ModuleList(nn.Conv2d(x, self.no, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export

        # Assuming self.nl is always 1 for HorizonDetect based on typical usage
        if self.nl != 1:
             print(f"Warning: HorizonDetect received {self.nl} input feature maps, expected 1.")
        
        # Process the first (and likely only) input feature map
        x_in = x[0] 

        # Apply GAP
        x_pooled = self.pool(x_in)

        # Apply final conv layer
        x_out = self.m[0](x_pooled)

        # Remove spatial dimensions (1, 1)
        x_out = x_out.view(x_out.shape[0], -1) # Shape becomes (bs, self.no)

        if self.training:
            # Return the tensor directly (bs, self.no)
            # Apply sigmoid to stabilize training
            out = x_out.sigmoid() 
        else: # Inference
            # Apply sigmoid and return tensor (or potentially tuple like other heads?)
            # For now, just sigmoid and return tensor for consistency with training
            y = x_out.sigmoid()
            # Note: Inference path might need adjustment based on how it's used later
            # The original inference path created `z` and potentially concatenated. 
            # Returning just the tensor seems more logical for horizon prediction.
            out = y

        return out

    # Remove convert method as it's likely not needed for horizon output
    # def convert(self, z): ...


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class IKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), nkpt=17, ch=(), inplace=True, dw_conv_kpt=False):  # detection layer
        super(IKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det=(nc + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        self.no = self.no_det+self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)
        
        if self.nkpt is not None:
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else: #keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt==0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else :
                x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class IBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):  # detection layer
        super(IBin, self).__init__()
        self.nc = nc  # number of classes
        self.bin_count = bin_count

        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        # classes, x,y,obj
        self.no = nc + 3 + \
            self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()   # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()
        
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        #self.x_bin_sigmoid.use_fw_regression = True
        #self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True
        
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                

                #px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                #py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                #y[..., 0] = px
                #y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph
                
                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)
                
                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch_initial = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.ch = parse_model(deepcopy(self.yaml), [ch_initial])  # model, savelist, channels list
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors for all necessary heads
        self.stride = None # Initialize self.stride
        for i, m in enumerate(self.model):
            if isinstance(m, (Detect, IDetect, IAuxDetect, IBin, IKeypoint)):
                s = 256  # 2x min stride
                
                # Calculate stride using a dummy forward specific to this head
                # This requires the head module `m` to have a `forward` method
                # that can accept a dummy tensor of appropriate channel count.
                # We need the number of input channels `ch_in` to the head `m`.
                # Let's retrieve it from the `ch` list created by `parse_model`.
                # Note: `parse_model` appends output channels, so `ch[m.f]` or `ch[m.f[0]]` might work.
                
                try:
                    # Determine input channels `ch_in` based on `m.f`
                    if isinstance(m.f, int):
                        # Handle negative 'f' indices correctly relative to current layer 'i'
                        # Note: ch[k] stores the output channels of layer k-1.
                        source_layer_index = i + m.f if m.f < 0 else m.f 
                        c1 = self.ch[source_layer_index + 1] # Get output channel of the source layer
                    elif isinstance(m.f, list):
                         # For multi-input heads like IDetect, need list of channel counts
                         # ch_in = [self.model[j].c2 for j in m.f] # Placeholder - NEEDS PROPER FIX
                         pass # Defer calculation
                    else:
                         raise ValueError(f"Unexpected layer.f type: {type(m.f)}")
                    
                    # ---> Let's skip stride calculation FOR NOW and fix parse_model first.
                    # dummy_input = torch.zeros(1, ch_in, s, s)
                    # head_output = m(dummy_input)
                    # output_shape = head_output[0].shape if isinstance(head_output, tuple) else head_output.shape
                    # m.stride = torch.tensor([s / output_shape[-2]])
                    
                    # Placeholder stride assignment until parse_model is fixed
                    if not hasattr(m, 'stride') or m.stride is None:
                        logger.warning(f"Layer {i} ({m.type}): Stride not calculated dynamically, assigning default.")
                        if isinstance(m, (Detect, IDetect)) and len(m.anchors) == 3:
                             m.stride = torch.tensor([8., 16., 32.])
                        else:
                             m.stride = torch.tensor([8.]) # Default single stride
                
                except Exception as e:
                    logger.warning(f"Layer {i} ({m.type}): Error during stride setup - {e}. Assigning default.")
                    if not hasattr(m, 'stride') or m.stride is None:
                         m.stride = torch.tensor([8.]) # Default single stride
                
                # Handle anchor grid setup for detection heads
                if hasattr(m, 'anchor_grid') and hasattr(m, 'stride') and m.stride is not None:
                    check_anchor_order(m)
                    # Ensure stride has correct dimensions for division
                    stride_view = m.stride.view(-1, 1, 1)
                    # Ensure anchor grid matches number of strides
                    nl_stride = stride_view.shape[0]
                    nl_anchor = m.anchor_grid.shape[0]
                    if nl_stride == nl_anchor:
                         m.anchors /= stride_view
                    elif nl_anchor % nl_stride == 0: # Check if anchors are grouped per stride
                        na = m.anchor_grid.shape[2]
                        m.anchors = m.anchors.view(nl_stride, -1, 2)
                        m.anchor_grid = m.anchor_grid.view(nl_stride, 1, -1, 1, 1, 2)
                        m.anchors /= stride_view
                    else:
                         logger.warning(f"Layer {i} ({m.type}): Mismatch between strides ({nl_stride}) and anchors ({nl_anchor}). Skipping scaling.")
                
                # Store the main detection stride (assuming IDetect or Detect is primary)
                if isinstance(m, (IDetect, Detect)) and self.stride is None:
                     self.stride = m.stride
                
                # Initialize biases for the specific head type (ensure stride exists)
                if hasattr(m, 'stride') and m.stride is not None:
                    if isinstance(m, Detect) or isinstance(m, IDetect):
                        self._initialize_biases(head=m)
                    elif isinstance(m, IAuxDetect):
                        self._initialize_aux_biases(head=m)
                    elif isinstance(m, IBin):
                        self._initialize_biases_bin(head=m)
                    elif isinstance(m, IKeypoint):
                        self._initialize_biases_kpt(head=m)
                else:
                    logger.warning(f"Layer {i} ({m.type}): Skipping bias initialization due to missing stride.")

            # We might need similar logic for HorizonDetect if it needs stride?
            # Currently HorizonDetect doesn't use self.stride in its forward pass.
            
        # Ensure a primary stride was found if expected
        if self.stride is None:
             logger.warning("Could not determine primary model stride.")

        # Init weights, biases (general initialization)
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        detection_output = None
        horizon_output = None

        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # Get input(s) from stored outputs `y`
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Debug print before calling the module - REMOVED
            # print(f"DEBUG: Before m(x): Layer {m.i}, Module Type: {m.type}, Input Type: {type(x)}") 

            # Run the current module
            x = m(x)

            # Always store the output tensor `x` for potential use by subsequent layers.
            y.append(x)

            # Keep track of the specific head outputs
            if isinstance(m, IDetect):
                detection_output = x
                # print(f"DEBUG: Assigned detection_output (type: {type(detection_output)}) ") # REMOVED
            elif isinstance(m, HorizonDetect):
                horizon_output = x
                # print(f"DEBUG: Assigned horizon_output (type: {type(horizon_output)}) ") # REMOVED

        # --- Profiling code removed for clarity --- 

        # Return based on which heads were found
        if detection_output is not None and horizon_output is not None:
            # During training, heads return lists/tensors, inference might return tuple
            # Handle potential tuples from inference mode if needed, though training is primary concern here
            det_out = detection_output[0] if isinstance(detection_output, tuple) and not self.training else detection_output
            hor_out = horizon_output[0] if isinstance(horizon_output, tuple) and not self.training else horizon_output
            # print(f"DEBUG: Returning Detection (type: {type(det_out)}) and Horizon (type: {type(hor_out)})" ) # REMOVED
            return det_out, hor_out 
        elif detection_output is not None:
            # print(f"DEBUG: Returning only Detection (type: {type(detection_output)})") # REMOVED
            # Handle potential tuple from inference mode
            return detection_output[0] if isinstance(detection_output, tuple) and not self.training else detection_output
        elif horizon_output is not None:
             # print(f"DEBUG: Returning only Horizon (type: {type(horizon_output)})") # REMOVED
             # Handle potential tuple from inference mode
             return horizon_output[0] if isinstance(horizon_output, tuple) and not self.training else horizon_output
        else:
            # print(f"DEBUG: Returning last computed x (type: {type(x)}) ") # REMOVED
            # Fallback: return the last computed output if no specific head output was identified
            return x

    def _initialize_biases(self, head, cf=None):
        m = head # Use passed head module
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(m.m, m.stride):
             b = mi.bias.view(m.na, -1) 
             b.data[:, 4] += math.log(8 / (640 / s) ** 2) 
             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum()) 
             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, head, cf=None):
        m = head # Use passed head module
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, mi2, s in zip(m.m, m.m2, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, head, cf=None):
        m = head # Use passed head module
        bc = m.bin_count
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            b[:, (0,1,2,bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, head, cf=None):
        m = head # Use passed head module
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(m.m, m.stride):
             b = mi.bias.view(m.na, -1) 
             b.data[:, 4] += math.log(8 / (640 / s) ** 2) 
             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum()) 
             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch_initial):  # model_dict, initial_input_channels(e.g., [3])
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, ch = [], [], ch_initial # layers, savelist, channel list (ch has initial input channels at ch[0])
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        
        # ----- Refactored Channel Calculation Logic -----
        args_orig = deepcopy(args) # Save original args from YAML for logging/reference
        
        # 1. Calculate input channels `c1` or list `c1_list`
        if isinstance(f, int):
            # Handle negative 'f' indices correctly relative to current layer 'i'
            # Note: ch[k] stores the output channels of layer k-1.
            source_layer_index = i + f if f < 0 else f 
            c1 = ch[source_layer_index + 1] # Get output channel of the source layer
        else: # f is a list
            # Handle negative indices within the list
            c1_list = [ch[i + x + 1] if x < 0 else ch[x+1] for x in f]
            c1 = c1_list
            
        # 2. Calculate output channels `c2` and determine `constructor_args` based on module `m`
        constructor_args = [] # Initialize
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, 
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC, 
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, 
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            # These modules generally take (c1, c2, *other_args_from_yaml)
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = args_orig[0] # Target output channels from YAML args[0]
            if c2 != no: c2 = make_divisible(c2 * gw, 8)
            constructor_args = [c1, c2, *args_orig[1:]] 
            # Special handling for CSP blocks needing 'n' inserted
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC, STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                constructor_args.insert(2, n); n = 1 # Insert n repeats, set n=1 for nn.Sequential wrapping
        
        elif m is nn.BatchNorm2d:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 # Output channels = input channels
            constructor_args = [c1] # BN constructor takes num_features=c1
        
        elif m is Concat:
            if not isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Expected list of input channels, got {c1}.")
            c2 = sum(c1) # Output channels = sum of list of input channels
            constructor_args = args_orig # Concat constructor takes dimension from YAML args
        
        elif m is Shortcut:
            if not isinstance(c1, list) or len(c1) != 2: raise TypeError(f"Layer {i} ({m.__name__}): Expected list of 2 input channels, got {c1}.")
            c2 = c1[0] # Output channels = channels of first input in list
            constructor_args = args_orig # Shortcut constructor takes dimension from YAML args
            
        elif m is Foldcut:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 // 2 # Output channels = input / 2
            constructor_args = args_orig # Foldcut constructor takes dimension from YAML args
            
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]: # Heads taking ..., ch
            if not isinstance(c1, list): c1 = [c1] # Ensure c1 is a list for heads
            c2 = -1 # Output channels not relevant for main ch list progression
            constructor_args = args_orig # Head constructors take nc, anchors, etc. from YAML args
            constructor_args.append(c1) # Append the calculated list of input channels `c1`
            # Handle anchor int format
            if m in [Detect, IDetect] and len(constructor_args) > 1 and isinstance(constructor_args[1], int):
                 constructor_args[1] = [list(range(constructor_args[1] * 2))] * len(f)
        
        elif m is HorizonDetect: # Head taking ch
             if not isinstance(c1, list): c1 = [c1]
             c2 = -1
             constructor_args = args_orig # Takes no args from YAML initially
             constructor_args.append(c1) # Appends list of input channels
             
        elif m is ReOrg:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 * 4
            constructor_args = args_orig # ReOrg constructor takes no args
            
        elif m is Contract:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 * args_orig[0] ** 2
            constructor_args = args_orig # Contract constructor takes gain
            
        elif m is Expand:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 // args_orig[0] ** 2
            constructor_args = args_orig # Expand constructor takes gain
            
        elif m is nn.Upsample:
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 # Upsample keeps the channel count
            constructor_args = args_orig # Upsample takes scale_factor, mode from YAML
            
        elif m is MP: # MaxPool
            if isinstance(c1, list): raise TypeError(f"Layer {i} ({m.__name__}): Received list of input channels {c1}, but expected a single int.")
            c2 = c1 # MaxPool keeps the channel count
            constructor_args = args_orig # MP constructor takes k from YAML
            
        else: # Fallback for unhandled modules
            # This is risky, assume output=input and constructor takes YAML args directly
            print(f"WARNING: Layer {i}, Module {m.__name__}: Unhandled in parse_model channel calculation. Assuming c2=c1 and constructor takes YAML args.")
            c2 = c1[0] if isinstance(c1, list) else c1
            constructor_args = args_orig

        # 3. Construct the module
        m_ = nn.Sequential(*[m(*constructor_args) for _ in range(n)]) if n > 1 else m(*constructor_args)
        # ---------------------------------------------

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # Log using constructor_args for accuracy, consider showing args_orig too?
        log_args = str(constructor_args) # Or maybe format differently? 
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, log_args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2) # Append calculated output channels
        
        # ---> DEBUG PRINT <-----
        # if i >= 0 and i <= 5: # Check first few layers
        #      print(f"DEBUG parse_model: Layer={i}, From={f}, Module={t}, C1={c1}, C2={c2}, ConstructorArgs={constructor_args}")
        #      print(f"DEBUG parse_model: Layer={i}, Appended c2={c2}, New Ch Len={len(ch)}, Last 5 ch={ch[-5:]}") 
        # elif i >= 70 and i <= 82: # Check layers around the error
        #      print(f"DEBUG parse_model: Layer={i}, From={f}, Module={t}, C1={c1}, C2={c2}, ConstructorArgs={constructor_args}")
        #      print(f"DEBUG parse_model: Layer={i}, Appended c2={c2}, New Ch Len={len(ch)}, Last 5 ch={ch[-5:]}") 

    return nn.Sequential(*layers), sorted(list(set(save))), ch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
    
    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
