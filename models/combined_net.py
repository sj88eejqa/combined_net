import models.utility as ut
import torch.nn as nn


class Combined_Net(nn.Module):
    def __init__(self, backbone="resnext50",image_shape=(256,256),n_class=22):
        super(Combined_Net,self).__init__()
        self.resnext = ut.get_conv_model(backbone) # default resnext50
        self.ssd_part = ut.get_ssd_part()
        self.panss_part = ut.get_panss_part(image_shape=image_shape,n_class=22)

    def forward(self, x):
        ### conv part
        n2,n3,n4,n5 = self.resnext(x)

        ### ssd part
        loc, conf = self.ssd_part(n5)

        ### panss part
        scores = self.panss_part([n2,n3,n4,n5])

        return loc,conf,scores

### only for debugging
if __name__ == "__main__":
    from torch.autograd import Variable
    import torch as t
    model = Combined_Net().cuda()
    a = Variable(t.ones((1,3,256,256))).cuda()
    loc,conf,scores = model(a)
    print("loc :{}, conf :{}, scores:{}".format( loc.shape,conf.shape,scores.shape))