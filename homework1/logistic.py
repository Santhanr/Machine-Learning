"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is logistic_train, logistic_test, and feature_square.
"""

import json
import numpy
import sys


###### Q6.1 ######
def logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features. 
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector of logistic regression and initialized to be all 0s
    - b: a scalar corresponding to the bias of logistic regression model
    - step_size: step size (or learning rate) used in gradient descent
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w and b.
    """
    # you need to fill in your solution here
    Xtrain = numpy.asarray(Xtrain)
    ytrain = numpy.asarray(ytrain)
    b = 0.0
    w = numpy.array([0.0] * Xtrain.shape[1])
    for j in range(0,max_iterations):
        sum_grad_bias = 0.0
        sum_grad_coeff = numpy.array([0.0] * Xtrain.shape[1])
        for i in range(0,Xtrain.shape[0]):
            temp=numpy.dot(w,Xtrain[i])
            temp=temp+b
            temp=float(numpy.exp(-1*temp))
            temp=temp+1.0
            temp=1.0/temp
            temp=temp-float(ytrain[i])
            sum_grad_bias=sum_grad_bias+temp
            coeff=numpy.multiply(temp,Xtrain[i])
            sum_grad_coeff=numpy.add(sum_grad_coeff,coeff)
        avg_grad_bias=sum_grad_bias/float(Xtrain.shape[0])
        avg_grad_coeff=numpy.divide(sum_grad_coeff,float(Xtrain.shape[0]))
        b=b-(float(step_size)*avg_grad_bias)
        coeff_update=numpy.multiply(float(step_size), avg_grad_coeff)
        w=numpy.subtract(w,coeff_update)
    return w, b

###### Q6.2 ######
def logistic_test(Xtest, ytest, w, b, t = 0.5):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features. 
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of logistic regression and learned by logistic_train()
    - b_l: a scalar, which is the bias of logistic regression and learned by logistic_train()
    - t: threshold, when you get the prediction from logistic regression, it should be real number from 0 to 1. Make all prediction less than t to 0 and otherwise make to 1 (Binarize)
    
    Returns:
    - testing accuracy.
    """
    # you need to fill in your solution here
    Xtest = numpy.asarray(Xtest)
    ytest = numpy.asarray(ytest)
    hits=0
    for i in range(0,Xtest.shape[0]):
        temp = numpy.dot(Xtest[i], w)
        temp = temp+b
        temp = numpy.exp(-1 * temp)
        temp = temp + 1
        temp = 1 / temp
        if(temp<t):
            pred=0
        else:
            pred=1
        if(pred==ytest[i]):
            hits=hits+1
    test_acc=float(hits)/float(Xtest.shape[0])
    return test_acc

###### Q6.3 ######
def feature_square(Xtrain, Xtest):
    """
    - Xtrain: training features, consists of num_train data points, each of which contains a D-dimensional feature
    - Xtest: testing features, consists of num_test data points, each of which contains a D-dimensional feature

    Returns:
    - element-wise squared Xtrain and Xtest.
    """
    Xtrain_s=numpy.square(Xtrain)
    Xtest_s=numpy.square(Xtest)
    return Xtrain_s, Xtest_s


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_toydata(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, test_set = data_set['train'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    return Xtrain, ytrain, Xtest, ytest

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = 0
        else:
            ytrain[i] = 1
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = 0
        else:
            ytest[i] = 1

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest

def inti_parameter(Xtrain):
    m, n = numpy.array(Xtrain).shape
    w = numpy.array([0.0] * n)
    b = 0
    step_size = 0.1
    max_iterations = 500
    return w, b, step_size, max_iterations

def logistic_toydata1():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata1.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def logistic_toydata2():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json')

    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def logistic_toydata2s():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json') # squared data

    Xtrain_s, Xtest_s = feature_square(Xtrain, Xtest)
    w, b, step_size, max_iterations = inti_parameter(Xtrain_s)


    w_l, b_l = logistic_train(Xtrain_s, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest_s, ytest, w_l, b_l)

    return test_acc

def logistic_mnist():

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)
    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc


def main():

    test_acc = dict()

    #=========================toydata1===========================
    
    test_acc['toydata1'] = logistic_toydata1() # results on toydata1
    print('toydata1, test acc = %.4f \n' % (test_acc['toydata1']))

    #=========================toydata2===========================
    
    test_acc['toydata2'] = logistic_toydata2() # results on toydata2
    
    print('toydata2, test acc = %.4f \n' % (test_acc['toydata2']))
    test_acc['toydata2s'] = logistic_toydata2s() # results on toydata2 but with squared feature 
    
    print('toydata2 w/ squared feature, test acc = %.4f \n' % (test_acc['toydata2s']))
    
    #=========================mnist_subset=======================

    test_acc['mnist'] = logistic_mnist() # results on mnist
    print('mnist test acc = %.4f \n' % (test_acc['mnist']))

    
    with open('logistic.json', 'w') as f_json:
        json.dump([test_acc], f_json)

if __name__ == "__main__":
    main()
