import numpy as np
import pandas as pd
from .funcs import *

class hierarchical(object):
    def __init__(self,y,x,w):
        if len(y)!=len(x) or len(y)!=len(w):
            raise ValueError('不匹配的组数')
        self.J=len(y)
        self.x=x
        self.y=y
        #总样本数
        n=0
        for yi in y:
            n+=yi.shape[0]
        self.n=n
        #样本特征数
        p=x[0].shape[1]
        for xi in x:
            if p!=xi.shape[1]:
                raise Exception('特征数量不匹配')
        self.p=p
        #组特征数
        pq=w[0].shape[1]
        for wi in w:
            if pq!=wi.shape[1]:
                raise Exception('组特征数不匹配')
        self.w=w#w
        q=int(pq/p)
        self.q=q
        
        self.gamma=np.random.rand(p*q,1)+1#gamma
        self.sigma2_y=1#sigma_y
        self.sigma2_u=1#sigma_u
    def sigma2_y_update(self,sigma2_y,sigma2_u,gamma):
        #y的方差的迭代函数
        num=0
        b=np.zeros(shape=(self.p,self.p))
        for xi,yi,wi in zip(self.x,self.y,self.w):
            u_star=u_star_func(xi,yi,wi,gamma,sigma2_y,sigma2_u)
            c_inv=c_inv_func(xi,sigma2_y,sigma2_u)
            di=dj_func(yi,xi,wi,gamma)
            a=di-xi.dot(u_star)
            b+=xi.T.dot(xi)*sigma2_y*c_inv
            num+=a.T.dot(a)
        num=num+np.trace(b)
        #pdb.set_trace()
        return num/self.n
    def sigma2_u_update(self,sigma2_y,sigma2_u,gamma):
        #u的方差的迭代函数
        num=0
        for xi,yi,wi in zip(self.x,self.y,self.w):
            u_star=u_star_func(xi,yi,wi,gamma,sigma2_y,sigma2_u)
            u_square=u_square_func(xi,sigma2_u,sigma2_y)
            a=u_star.T.dot(u_star)
            b=np.trace(u_square)
            num+=a+b
        return num/(self.J*self.p)
    def gamma_update(self,sigma2_u,sigma2_y,gamma,lamda):
        #gamma的迭代函数
        num_vecter=np.zeros(shape=(gamma.shape[0],gamma.shape[1]))
        num_matrix=np.zeros(shape=(self.p*self.q,self.p*self.q))
        condition0=np.zeros(shape=(gamma.shape[0],gamma.shape[1]))
        for xi,yi,wi in zip(self.x,self.y,self.w):
            u_star=u_star_func(xi,yi,wi,gamma,sigma2_y,sigma2_u)
            a=wi.T.dot(xi.T.dot(yi))
            b=wi.T.dot(xi.T.dot(xi.dot(u_star)))
            c=wi.T.dot(xi.T.dot(xi.dot(wi)))
            condition0=condition0+a-b-c.dot(gamma)
            num_vecter=num_vecter+a-b
            num_matrix=num_matrix+c
        if np.linalg.matrix_rank(num_matrix)!=num_matrix.shape[0]:
            raise ValueError('矩阵不满秩！')
        gamma_hat=np.linalg.inv(num_matrix).dot(num_vecter-lamda*sigma2_y/2*np.sign(gamma))
        for i in range(gamma.shape[0]):
            if abs(condition0[i,0])<lamda*sigma2_y/2:
                gamma[i,0]=0
            else:
                gamma[i,0]=gamma_hat[i,0]
        return gamma
    def update(self,tol,max_iter,lamda):
        #将所有系数更新一次的函数
        dist=1
        n=0
        while dist>tol and n<max_iter:
            sigma2_y_new=self.sigma2_y_update(self.sigma2_y,self.sigma2_u,self.gamma.copy())[0][0]
            sigma2_u_new=self.sigma2_u_update(self.sigma2_y,self.sigma2_u,self.gamma.copy())[0][0]
            gamma_new=self.gamma_update(self.sigma2_u,self.sigma2_y,self.gamma.copy(),lamda)
            dist=np.sum(np.square(gamma_new-self.gamma))
            n+=1
            if n%100==0:
                print('已经迭代'+str(n)+'次')
            
            self.sigma2_y=sigma2_y_new
            self.sigma2_u=sigma2_u_new
            self.gamma=gamma_new
        if n>=max_iter:
            print('模型未收敛')
        else:
            print('模型收敛')
        return None