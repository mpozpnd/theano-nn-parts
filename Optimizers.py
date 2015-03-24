import numpy as np
import theano
import theano.tensor as T



from collections import OrderedDict


def adadelta(rho_,eps_,cost,params,L1_rate,L2_rate):
        updates = OrderedDict()
        gparams = [T.grad(cost + L1_rate * param.norm(L=1) + L2_rate * param.norm(L=2) , param) for param in params]
        gparams_mean=[gparam.mean() for gparam in gparam in gparams]

        eg2=[theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]
        dxt=[theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]
        ex2=[theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]

        for gparam, param, eg2_,ex2_ in zip(gparams,params,eg2,ex2):
         updates[eg2_]=T.cast(rho_ * eg2_ +(1. -rho_)*(gparam **2) ,theano.config.floatX)
         dparam = -T.sqrt((ex2_ + eps_)/(updates[eg2_] + eps_)) * gparam
         updates[ex2_]=T.cast(rho_ * ex2_ + (1. -rho_)*(dparam **2),theano.config.floatX)
         updates[param]=T.cast(param + dparam,theano.config.floatX)
        return updates,eg2,ex2

def sgd(lr,cost,params,L1_rate,L2_rate):
    gparams = [T.grad(cost + L1_rate * param.norm(L=1) + L2_rate * param.norm(L=2) , param) for param in params]
    gparams_mean=[gparam.mean() for gparam in gparams]
    updates = [(param,param - lr * gparam) for param,gparam in zip(params,gparams)]
    return updates,gparams_mean

def sgd_momentum(lr,momentum,cost,params,L1_rate,L2_rate):
    gparams = [T.grad(cost + L1_rate * param.norm(L=1) + L2_rate * param.norm(L=2) , param) for param in params]
    #gparams_mean=[gparam.mean() for gparam in gparam in gparams]
    ex2=[theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]

    for gparam,param,ex2_ in zip(gparams,params,ex2):
        updates[ex2_]=ex2_
        updates[param]=T.cast(param + momentum * ex2_ - lr * gparam,theano.config.floatX)
    return updates,ex2





