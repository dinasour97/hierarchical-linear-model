import numpy as np

#一些辅助计算函数
def c_inv_func(xi,sigma2_y,sigma2_u):
    #求c^(-1)的函数
    p=xi.shape[1]
    return np.linalg.inv(np.dot(xi.T,xi)+sigma2_y/(sigma2_u)*np.identity(p))
def u_star_func(xi,yi,wi,gamma,sigma2_y,sigma2_u):
    #求u期望的函数
    c_inv=c_inv_func(xi,sigma2_y,sigma2_u)
    u_star=c_inv.dot(xi.T.dot(yi-xi.dot(wi.dot(gamma))))
    return u_star
def u_square_func(xi,sigma2_u,sigma2_y):
    #求u方差的函数
    c_inv=c_inv_func(xi,sigma2_y,sigma2_u)
    return sigma2_y*c_inv
def dj_func(yi,xi,wi,gamma):
    #求dj的函数
    return yi-xi.dot(wi.dot(gamma))