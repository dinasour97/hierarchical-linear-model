import numpy as np

#生成测试数据
#输入方差和期望，生成数据的函数
class data_gen(object):
    def __init__(self,J,p,q,n):
        self.J=J
        self.p=p
        self.q=q
        self.n=n
    def x_gen(self,mu,sigma):
        #每一组x都是list的一个元素
        x_list=[]
        for j in range(self.J):
            mu_x=np.zeros(self.p)+mu
            sigma_x=np.eye(self.p)*sigma
            x_else=np.random.multivariate_normal(mu_x,sigma_x,self.n)
            x_1=np.zeros(self.n)+1
            x = np.column_stack((x_1,x_else))
            x_list.append(x)
        return x_list
    def w_gen(self,mu,sigma):
        #每一组w都是list的一个元素
        #是经过格式变换的形式
        w_list=[]
        for j in range(self.J):
            mu_w=np.zeros(self.q)+mu
            sigma_w=np.eye(self.q)*sigma
            w_else=np.random.multivariate_normal(mu_w,sigma_w,1)
            w=np.insert(w_else,0,1)
            I=np.eye(self.p+1)
            w=np.kron(I,w)
            w_list.append(w)
        return w_list
    def y_gen(self,gamma,sigma_y,sigma_u,w,x):
        #根据x,gamma,w生成y
        if gamma.shape[0]!=(self.p+1)*(self.q+1) or gamma.shape[1]!=1:
            raise ValueError('错误的gamma')
        y_list=[]
        for j in range(self.J):
            #pdb.set_trace()
            u=np.random.normal(0,sigma_u,self.p+1).reshape(-1,1)
            e=np.random.normal(0,sigma_y,self.n).reshape(-1,1)
            beta=w[j].dot(gamma)+u
            y=x[j].dot(beta)+e
            y_list.append(y)
        return y_list