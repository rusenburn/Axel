import torch as T

def intprod(xs):
    out = 1
    for x in xs:
        out*=x
    return out

def NormConv2d(conv2d:T.nn.Conv2d,scale=1,bias=True):
    conv2d.weight.data *= scale/conv2d.weight.norm(dim=(1,2,3),p=2,keepdim=True)
    if bias:
        conv2d.bias.data*=0
    return conv2d
def NormLinear(linear:T.nn.Linear,scale=1.0,bias=True):
    linear.weight.data *= scale/linear.weight.norm(dim=1,p=2,keepdim=True)
    if bias:
        linear.bias.data *= 0
    return linear