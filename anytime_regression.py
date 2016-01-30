import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

def vanilla_ridge_regression(X_train, y_train, lam, flag='vector'):
    if flag == 'vector':
        d = X_train.shape[1]
        inv = np.linalg.inv(np.dot(X_train.T,X_train)+lam*np.eye(d))
        w = np.dot(np.dot(inv,X_train.T),y_train)
    else:
        d = 1
        inv = np.linalg.inv(np.dot(X_train.T,X_train)+lam*np.eye(d))
        w = np.dot(inv*X_train,y_train)
    return w

def anytime_regularization(X_train,y_train,lam,p):
    n,d = X_train.shape
    dp = np.diag((1-p)/p)
    mean = np.tile(X_train.mean(axis=0),(n,1))
    XX = np.dot(X_train.T,X_train)
    dXX = np.diag(np.diag(XX))
    MM = np.dot(mean.T,mean)
    mm = np.dot(mean,dp)
    pMMp = np.dot(mm.T,mm)
    dMM = np.diag(np.diag(MM))
    #XX = np.cov(X_train.T)    
    #inv = np.linalg.inv(XX+pMMp+np.dot(dp,dXX+dMM)+lam*np.eye(X_train.shape[1]))
    inv = np.linalg.inv(XX+np.dot(dp,dXX)+lam*np.eye(X_train.shape[1]))
    w = np.dot(np.dot(inv,X_train.T),y_train)
    return w

def anytime_regression_l2(X_train, y_train, lam, p):
    dp = np.diag(p) #B
    dpp = np.diag(p-1.0/2*p*p)
    #dpp = np.diag(p)
    xp = np.dot(X_train,dp)
    inv = np.linalg.inv(np.dot(np.dot(dpp,X_train.T),X_train) + 1.0/2*np.dot(xp.T,xp)+lam*np.eye(X_train.shape[1]))
    #inv = np.linalg.inv(np.dot(np.dot(X_train.T,X_train),dpp) +lam*np.eye(X_train.shape[1]))
    w = np.dot(np.dot(inv,xp.T),y_train)
    return w 

def simulate_regression(X_train, y_train, lam, p):
    dp = np.diag(p)
    Xp = np.dot(X_train, dp)
    XX = np.dot(Xp.T,Xp)
    diag_XX = np.diag(np.diag(XX))
    X_s=np.dot(diag_XX,np.diag(1/p)) + XX - diag_XX
    inv = np.linalg.inv(X_s + lam*np.eye(X_train.shape[1]))
    w = np.dot(np.dot(inv,Xp.T),y_train)
    return w 

def vanilla_lasso_regression(X_train,y_train,lam):
    clf = lm.Lasso(alpha=lam)
    clf.fit(X_train,y_train)
    w = clf.coef_
    return w

def query_statistical_learning(X_train,y_train,lam):
    dic = {}
    n,d = X_train.shape
    assert d ==2
    w1=vanilla_ridge_regression(X_train[:,0], y_train, lam,'scalar')
    dic['1'] = w1
    w2=vanilla_ridge_regression(X_train[:,1], y_train, lam, 'scalar')
    dic['2'] = w2
    w12 = vanilla_ridge_regression(X_train,y_train,lam)
    dic['12'] = w12
    dic['0'] = np.zeros((d,))
    return dic
          
    

def predict(X_test,w,flag='regression'):
    if flag == 'regression':
        return np.dot(X_test,w)
    elif flag == 'classification':
        y_pred = np.dot(X_test,w)
        y_p = (y_pred >0.5).astype('int32')
        y_n =-1*(y_pred <-0.5).astype('int32')
        y_pred = y_p+y_n
        
        for i in range(y_pred.shape[0]):
            if y_pred[i]==0:
                rand = np.random.rand()
                if rand>0.5:
                    y_pred[i]=1
                else:
                    y_pred[i]=-1
        return y_pred

    
def qsl_predict(X_test,dic,flag ='classification'):
    n,d = X_test.shape
    assert d == 2
    y_pred = np.zeros((n,))
    for i in range(n):
        if (X_test[i,0]==1 or X_test[i,0]== -1) and (X_test[i,1]==1 or X_test[i,1]==-1):
            y_pred[i]=np.dot(X_test[i,:],dic['12'])
        elif X_test[i,0]==1 or X_test[i,0]== -1:
            y_pred[i]= X_test[1,0]*dic['1'] 
        elif X_test[i,1]==1 or X_test[i,1]== -1:
            y_pred[i]= X_test[1,1]*dic['2'] 
        else:
            y_pred[i]=0
        
        if flag == 'classification':
            if y_pred[i]>0.2:
                y_pred[i] = 1
            elif y_pred[i]<-0.2:
                y_pred[i] = -1
            else:
                rand = np.random.rand()
                if rand >0.5:
                    y_pred[i] = 1
                else:
                    y_pred[i] = -1
    return y_pred

def prediction_error(X_test,y_test,w,flag='regression'):
    N = X_test.shape[0]
    if flag == 'regression':
        dif = np.dot(X_test, w) - y_test
        error = 1.0/N*np.sqrt(np.sum(np.power(dif,2)))
        #error = np.sqrt(np.sum(np.power(dif,2)))
    elif flag == 'classification':
        y_pred = predict(X_test,w,flag)
        error = 1- np.mean(y_pred==y_test)
    elif flag == 'qsl':
        y_pred = qsl_predict(X_test,w)
        error = 1-np.mean(y_pred==y_test)
    elif flag == 'qsl_reg':
        y_pred = qsl_predict(X_test,w,'regression')
        dif = y_pred-y_test
        error = 1.0/N*np.sqrt(np.sum(np.power(dif,2)))
    return error

def modify_data(X,X_train,p):
    XX = np.array(X)
    M = np.mean(X_train,axis=0)
    X_mean = np.mean(X,axis=0)
    num = X.shape[0] - (p*X.shape[0]).astype('int32')
    for i in range(num.shape[0]):
        perm = np.random.permutation(X.shape[0])[0:num[i]]
        XX[perm, i] = M[i]
        #XX[perm,i] = 0
    return XX


def generate_synthetic_data_regression(n=1000,d=2, sigma=np.array([0.1,0.1]),w = np.array([0.5,0.5]),p = np.array([0.9,0.1]),noise = 0.01):
    #x = np.zeros((n,))
    X = np.zeros((n,d))
    for i in range(d):
        x = np.random.rand(n) * np.random.randint(1,10)
        x1 = x + np.random.normal(0,sigma[i],x.shape[0])
        X[:,i] = x1
    y = np.dot(X,w) + np.random.normal(0,noise,X.shape[0])
    X_train = np.array(X[0:n/2,:])
    y_train = np.array(y[0:n/2])
    X_test = np.array(X[n/2:,:])
    y_test = np.array(y[n/2:])
    return x, X_train,y_train, X_test, y_test


def generate_synthetic_data(n=2000,d=2,p=np.array([0.2,0.9])):
    a = [-1,1]
    y=np.random.choice(a,(n,)) # randomly generate labels, Non-baise dataset
    #mask = (y==0).astype('int32') ## increase number of positive labels
    #y = y+mask
    X = np.tile(y.reshape(n,1),(1,d)) ## all correct label
    if d >2:
        temp = (1-np.sort(np.random.rand(d)))*0.5
        order = np.argsort(p)
        rate = temp[order]
    else:
        if p[0]<=p[1]:
            rate = np.array([0.1,0.4]) 
        else:
            rate = np.array([0.4,0.1])
    for i in range(n):
        for j in range(d):
            random = np.random.rand()
            if random<rate[j]:
                if X[i,j]==1:
                    X[i,j] = -1
                else:
                    X[i,j] = 1
    return X,y                

def plot(error,lam,label='exp'):
    fig, ax = plt.subplots()
    ax.plot(lam, error, label=label)
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.plot()

    
'''
#unfinished
def cross_validate(X_train,y_train,X_val,y_val,p,lam,flag):
    n = lam.shape[0]
    dic = {'vrr':1,'qsl':1,'samp':1}
    error_list = [10,10,10]
    for i in range(n):
        w = vanilla_ridge_regression(X_train,y_train,lam[i])
        error = prediction_error(X_val,y_val,w,'classification')
        if error < error_list[0]:
            error_list[0] = error
            dic['vrr'] = lam[i]
            
        w = vanilla_ridge_regression(X_train,y_train,lam[i])
        error = prediction_error(X_val,y_val,w,'classification')
        if error < error_list[0]:
            error_list[0] = error
            dic['vrr'] = lam[i]
'''           
        


##### Don't need this function for now   
def normalize(X):
    n,d = X.shape
    sum_rows = np.sum(X,axis=0)
    mean_rows = np.mean(X,axis=0)
    XX = (X-mean_rows)/sum_rows
    return XX
    
        