import json
import random
import numpy as np

def getGausian(point,mu_k,cov_k):
    point=np.array(point)
    mu_k=np.array(mu_k)
    cov_k=np.array(cov_k)
    cov_k=np.reshape(cov_k,(2,2))
    gau_val = (np.linalg.det(cov_k)**-0.5)*(np.exp(-0.5*(np.transpose(point-mu_k).dot(np.linalg.inv(cov_k).dot(point-mu_k)))))
    return gau_val;

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    phi = [[0 for x in range(K)] for y in range(len(X))] 
    # Run 100 iterations of EM updates
    for t in range(100):
        for i in range(len(X)):
            phitemp=[]
            point = X[i]
            for k in range(K):
                phitemp.append(pi[k]*getGausian(X[i],mu[k],cov[k]))
            sumPhitemp=sum(phitemp)
            
            phitemp = [x / sumPhitemp for x in phitemp]
            #print(len(phitemp))
            phi[i]=phitemp
        
        #print(len(phi),len(phi[0]))    
        #maximization pi
        Narray=[]
        for i in range(K):
            tempSum=0
            for j in range(len(X)):
                tempSum+=phi[j][i]
            Narray.append(tempSum)
        pi = [x / len(X) for x in Narray]
        
        #maximization mu
        for i in range(K):
            tempSum=[0,0]
            for j in range(len(X)):
                point=X[j]
                for l in range(len(point)):
                    tempSum[l]+=phi[j][i]*point[l]
            mu[i] = [temp/Narray[i] for temp in tempSum]
            
        #maximization cov
        for i in range(K):
            tempSum=[0,0,0,0]
            tempSum=np.array(tempSum)
            for j in range(len(X)):
                #distort=[a-b for a,b in zip(X[j],mu[i])]
                point=np.array(X[j])
                mu_t=np.array(mu[i])
                distort = np.subtract(point,mu_t)
                #distort=np.array(distort)
                #print(distort.shape)
                distort=np.reshape(distort,(2,1))
                distort=np.dot(distort,np.transpose(distort))
                
                distort=np.reshape(distort,(4))
                #print("after flatten",distort.shape)
                distort=np.multiply(phi[j][i],distort)
                tempSum=np.add(tempSum,distort)
            tempSum = np.divide(tempSum,Narray[i])
            cov[i]=tempSum.tolist()
    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
