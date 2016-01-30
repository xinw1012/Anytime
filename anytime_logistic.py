import numpy as np


class anytime_logistic:
    def __init__(self,lam = 1):
        self.theta = None
        self.iters = 1000
        self.learn_rate = 0.01
        self.lam = lam
        self.loss = 0
        self.trained = False
    
    def sigmoid(self,X,theta):
        e = 1.0/(1+np.exp(-1*np.dot(X,theta)))
        return e
    
    def fit(self, X, y, p,flag='anytime'):
        assert X.shape[0] == y.shape[0]
        N,d = X.shape
        M = np.tile(X.mean(axis=0),(N,1)) #(N,d)
        
        theta = self.theta
        #theta = np.random.rand(d)
        theta = np.zeros((d,)) # fixed starting point
        iters = self.iters
        learn_rate = self.learn_rate
        lam = self.lam
        loss = self.loss
        
        if flag == 'anytime':
            #mean = np.mean(X)
            XTX = np.diag(np.diag(np.dot(X.T,X)))
            MTM = np.diag(np.diag(np.dot(M.T,M)))
            #XTX = np.diag(np.diag(np.cov(X.T)))
            #MTM = np.diag(np.diag(np.cov(M.T)))
            dp = np.diag((1-p)/p)
            #print 'XTX',XTX
            #print 'dp', dp
            coef =  1.0/N*(np.dot(dp,XTX+MTM))+ lam*np.eye(d) 
            #coef =  1.0/N*(np.dot(dp,XTX))+ lam*np.eye(d)
            print 'coef'
            print coef
        else:
            coef = lam*np.eye(d)
            print 'coef'
            print coef

        
        for i in range(iters):
            h = self.sigmoid(X,theta)
            grad = 1.0/N * np.dot(X.T, h-y) + np.dot(coef,theta)
            theta -= learn_rate*grad
            loss_new = 1.0/N *(-np.dot(y.T,np.log(h))-np.dot((1-y.T),np.log(1-h))) \
            + np.dot(np.dot(theta.T, coef),theta)
            
            #print loss_new
            if np.isnan(loss_new):
                break
            if abs(loss_new-loss)<1e-4:
                loss = loss_new
                self.loss = loss
                break
            loss = loss_new
            self.loss = loss
            #print loss

        self.theta = theta
        self.trained = True
        #print 'theta'
        #print self.theta   
            
    def predict(self,X):
        theta = self.theta
        h = self.sigmoid(X,theta)        
        y1 = (h>0.6).astype('int32')
        y2 = (h>=0.4).astype('int32')
        y_pred = (y1+y2)/2.0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == 0.5:
                random = np.random.rand()
                if random>=0.5:
                    y_pred[i]=1
                else:
                    y_pred[i]=0
        return y_pred

    
def generate_synthetic_data(n=2000,d=2,p=np.array([0.2,0.9])):
    a = [-1,0,1]
    y=np.random.choice(a,(n,)) # randomly generate labels
    mask = (y==0).astype('int32') ## increase number of positive labels
    y = y+mask
    X = np.tile(y.reshape(n,1),(1,d)) ## all correct label
    if d >2:
        temp = (1-np.sort(np.random.rand(d)))*0.5
        order = np.argsort(p)
        rate = temp[order]
    else:
        if p[0]<=p[1]:
            rate = np.array([0.05,0.6])
        else:
            rate = np.array([0.6,0.05])
    for i in range(n):
        for j in range(d):
            random = np.random.rand()
            if random<rate[j]:
                if X[i,j]==1:
                    X[i,j] = -1
                else:
                    X[i,j] = 1
    return X,y                
   
def modify_data(X_test,X_train,p, flag='avg'):
    M = X_train.mean(axis=0)
    XX= np.array(X_test)
    n,d = X_test.shape
    num = n-(n*p).astype('int32')
    for i in range(d):
        perm = np.random.permutation(n)[0:num[i]]
        if flag == 'avg':
            XX[perm,i]=M[i]
        else:
            XX[perm,i]=0
    return XX    


########### Need modification ###############    
def cross_validation(X,y,p,n_folds=10,flag='anytime'):
    lam = np.logspace(-4,5,n_folds)
    split_X = np.split(X,n_folds)
    split_y = np.split(y,n_folds)
    accuracy = []
    for i in range(len(split_X)):
        split_new_X = list(split_X)
        split_new_y = list(split_y)
        X_validation = split_new_X[i]
        y_validation = split_new_y[i]
        del split_new_X[i]
        del split_new_y[i]
        X_train = np.ma.concatenate(split_new_X)
        y_train = np.concatenate(split_new_y)
        #test model 
        model = anytime_logistic(lam[i])
        model.fit(X_train,y_train,p,flag)
        y_pred = anytime.predict(X_validation)
        acc = np.mean(y_pred == y_validation)
        accuracy.append(acc)
    print 'accuracies'
    print accuracy
    index = np.argmax(accuracy)
    print 'best lambda: ', lam[index]
    return lam[index]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    