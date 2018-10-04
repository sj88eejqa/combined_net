import torch.nn.functional as F
import torch.nn as nn
import torch as t
import ssd as ssd

def get_conv_model(model_name='resnext50'):
    return{
        #'resnext18':ResNeXt(Bottlenect,[2,2,2,2],32),
        #'resnext34':ResNeXt(Bottlenect,[3,4,6,3],32),
        'resnext50':ResNeXt(Bottlenect,[3,4,6,3],32),
        'resnext101':ResNeXt(Bottlenect,[3,4,23,3],32),
        'resnext152':ResNeXt(Bottlenect,[3,8,36,3],32)
    }[model_name.lower()]

def get_ssd_part():
    return SSD_Last()

def get_panss_part(in_c=256,image_shape=(256,256),n_class=21):
    return PANSS(in_c,image_shape,n_class)

class SEblock(nn.Module):
    def __init__(self, channels):
        ratio = 16
        super(SEblock, self).__init__()
        self.excit_block = nn.Sequential(
            nn.Conv2d(channels,channels//ratio,1,1,0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//ratio,channels,1,1,0),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w = x.shape
        x = F.avg_pool2d(x, (h,w), stride=1, padding=0)
        x = self.excit_block(x)
        return x

class FPN(nn.Module):
    def __init__(self,reduce_factor = 256):
        super(FPN,self).__init__()
        reduce_factor = reduce_factor
        channel = [2048,1024,512,256]
        self.reduces = nn.Sequential(*[nn.Conv2d(i,reduce_factor,kernel_size=1,stride=1,padding=0,bias=False) for i in channel])
        self.fusions = nn.Sequential(*[nn.Conv2d(reduce_factor,reduce_factor,kernel_size=3,padding=1,stride=1,bias=False) for _ in range(3)])

    def forward(self,stages):
        s2,s3,s4,s5 = stages
        p5 = self.reduces[0](s5)
        p4 = self.fusions[0](self.reduces[1](s4)+p5)
        p3 = self.fusions[1](self.reduces[2](s3)+p4)
        p2 = self.fusions[2](self.reduces[3](s2)+F.upsample(p3,(64,64),mode='bilinear',align_corners=True))

        return [p2,p3,p4,p5]

class PAN_Button_Up(nn.Module):
    def __init__(self,channel = 256):
        super(PAN_Button_Up,self).__init__()
        channel = channel
        self.reduce = nn.Conv2d(channel,channel,kernel_size=3,stride=2,padding=1,bias=False)
        self.fusions = nn.Sequential(*[nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False) for _ in range(3)])

    def forward(self, Ps):
        p2,p3,p4,p5 = Ps
        n2 = p2
        n3 = self.fusions[0](p3+self.reduce(n2))
        n4 = self.fusions[1](p4+n3)
        n5 = self.fusions[2](p5+n4)
        return [n2,n3,n4,n5]

class Bottlenect(nn.Module):
    expansion = 2
    def __init__(self,inplanes,planes,stride=1,downsample=None,dilation=1,num_group=32):
        super(Bottlenect,self).__init__()
        self.conv1_1x1_bn_relu = nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2_3x3_bn_relu = nn.Sequential(
            nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=dilation,dilation=dilation,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3_1x1_bn = nn.Sequential(
            nn.Conv2d(planes,planes*2,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(planes*2)
        )
        self.downsample = downsample
        self.se_block = SEblock(planes*2)

    def forward(self, x):
        residual = x

        out = self.conv1_1x1_bn_relu(x)
        out = self.conv2_3x3_bn_relu(out)
        out = self.conv3_1x1_bn(out)
        weight = self.se_block(out)
        out *= weight
        if self.downsample is not None:
            residual = self.downsample(x)
        #print("out shape: {}, residual shapeL {}".format(out.shape,residual.shape))
        out += residual
        return F.relu(out)

class ResNeXt(nn.Module):
    def __init__(self,block,layers,num_group=32):
        super(ResNeXt,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # (N,64,64,64)
        self.stage2 = self._make_layer(block,128,layers[0],num_group) # (N,256,64,64)
        self.stage3 = self._make_layer(block,256,layers[1],num_group,stride=2) # (N,512,32,32)
        self.stage4 = self._make_layer(block,512,layers[2],num_group,dilation=2) # (N,1024,32,32)
        self.stage5 = self._make_layer(block,1024,layers[3],num_group,dilation=4)  # (N,2048,32,32)

        self.fpn = FPN(reduce_factor=256)
        self.pan_bu = PAN_Button_Up(channel=256)

    def _make_layer(self,block,planes,blocks,num_group,stride=1,dilation=1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                    kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample,num_group=num_group))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes,dilation=dilation,num_group=num_group))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)

        ps = self.fpn([stage2,stage3,stage4,stage5])
        ns = self.pan_bu(ps)
        return ns


class SSD_Last(nn.Module):
    def __init__(self):
        super(SSD_Last,self).__init__()
        ori_ssd = ssd.SSD300()
        self.fit_ssd = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=2,dilation=2,bias=False)
        self.norm4 = ori_ssd.norm4
        self.conv5s = nn.Sequential(*[getattr(ori_ssd,conv) for conv in [
            'conv5_1','conv5_2','conv5_3',
        ]])
        self.conv6_7 = nn.Sequential(*[getattr(ori_ssd,conv) for conv in[
            'conv6','conv7',
        ]])
        self.conv8s = nn.Sequential(*[getattr(ori_ssd,conv) for conv in[
            'conv8_1','conv8_2',
        ]])
        self.conv9s = nn.Sequential(*[getattr(ori_ssd,conv) for conv in [
            'conv9_1','conv9_2',
        ]])
        self.conv10s = nn.Sequential(*[getattr(ori_ssd,conv) for conv in[
            'conv10_1','conv10_2',
        ]])
        self.conv11s = nn.Sequential(*[getattr(ori_ssd,conv) for conv in[
            'conv11_1','conv11_2',
        ]])
        self.multibox = getattr(ori_ssd,'multibox')

    def forward(self, x):
        x = self.fit_ssd(x)
        hs = []
        hs.append(self.norm4(x))

        h = F.max_pool2d(x, kernel_size=2, stride=1, ceil_mode=True)
        
        h = self.conv_relu(3,self.conv5s,h)
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = self.conv_relu(2,self.conv6_7,h)
        hs.append(h)

        h = self.conv_relu(2,self.conv8s,h)
        hs.append(h)

        h = self.conv_relu(2,self.conv9s,h)
        hs.append(h)

        h = self.conv_relu(2,self.conv10s,h)
        hs.append(h)

        h = self.conv_relu(2,self.conv11s,h)
        hs.append(h)

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds
    
    def conv_relu(self,i,convs,x):
        for j in range(i):
            x = F.relu(convs[j](x))
        return x

# Feature Pyramid Attention module
class FPA(nn.Module):
    def __init__(self,in_c=256):
        super(FPA,self).__init__()
        self.conv7x7 = nn.Conv2d(in_c,in_c//2,kernel_size=7,stride=2,padding=3,bias=False)
        self.conv7x7_r = nn.Conv2d(in_c//2,in_c//2,kernel_size=7,stride=1,padding=3,bias=False)

        self.conv5x5 = nn.Conv2d(in_c//2,in_c//2,kernel_size=5,stride=2,padding=2,bias=False)
        self.conv5x5_r = nn.Conv2d(in_c//2,in_c//2,kernel_size=5,stride=1,padding=2,bias=False)
        
        self.conv3x3 = nn.Conv2d(in_c//2,in_c//2,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv3x3_r = nn.Conv2d(in_c//2,in_c//2,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.skip = nn.Conv2d(in_c,in_c//2,kernel_size=1,stride=1,padding=0,bias=False)
        self.gp_conv1x1 = nn.Conv2d(in_c,in_c//2,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self,x):
        down = self.conv7x7(x)
        down_r = self.conv7x7_r(down)

        down_5 = self.conv5x5(down)
        donw_5_r = self.conv5x5_r(down_5)

        down_3 = self.conv3x3(down_5)
        down_3_r = F.upsample(self.conv3x3_r(down_3),(8,8),mode='bilinear')
        
        down_5_r = F.upsample(donw_5_r+down_3_r,(16,16),mode='bilinear')
        down_r = F.upsample(down_r+down_5_r,(32,32),mode='bilinear')

        out = self.skip(x)*down_r
        gp = F.upsample(self.gp_conv1x1(F.avg_pool2d(x,(32,32),stride=1,padding=0)),(32,32),mode='bilinear')
        
        return t.cat([out,gp],1)

# Global Attention Upsample module
class GAU(nn.Module):
    def __init__(self,in_c=256):
        super(GAU,self).__init__()
        self.conv3x3 = nn.Conv2d(in_c,in_c,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv1x1 = nn.Conv2d(in_c,in_c,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self,low,high):
        attention = self.conv1x1(F.avg_pool2d(high,high.shape[-2:],stride=1,padding=0))
        out = self.conv3x3(low)*attention
        return out+high

### Pyramid Attention Network for Semantic Segmentation
class PANSS(nn.Module):
    def __init__(self,in_c=256,image_shape=(256,256),n_class=21):
        self.image_shape = image_shape
        super(PANSS,self).__init__()
        self.fpa = FPA(in_c)
        self.GAUs = nn.Sequential(*[GAU(in_c) for _ in range(3)])
        self.alis_conv = nn.Sequential(
            nn.Conv2d(in_c,in_c//2,3,1,1),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True)
        )
        self.scores = nn.Conv2d(in_c//2,n_class,3,1,1,bias=False) 

    def _pass_gau(self,low,high,gau,upsample=False):
        if upsample:
            high = F.upsample(high,low.shape[-2:],mode='bilinear')
        return gau(low,high)+high

    def forward(self,features):
        n2,n3,n4,n5 = features
        n5 = self.fpa(n5)
        n4 = self._pass_gau(n4,n5,self.GAUs[0],upsample=False)
        n3 = self._pass_gau(n3,n4,self.GAUs[1],upsample=False)
        n2 = self._pass_gau(n2,n3,self.GAUs[2],upsample=True)
        out = self.alis_conv(F.upsample(n2,tuple(i//2 for i in self.image_shape),mode='bilinear')) # (N,128,128,128)
        out = F.upsample(out,self.image_shape,mode='bilinear') # (N,128,256,256)
        return self.scores(out)

# only for debugging
if __name__ == "__main__":
    import torch as t
    from torch.autograd import Variable
    a = Variable(t.ones((1,3,256,256))).cuda()
    resnext = get_conv_model().cuda() # default resnext5
    out = resnext(a)[-1]
    print(out.shape) # (N, 256, 32 ,32)
    ssd_last = get_ssd_part().cuda()
    loc, conf = ssd_last(out)
    print("loc :{}, conf :{}".format(loc.shape,conf.shape))