
import numpy as np
import time

def our_pca(input_data: np.ndarray, info_amount: float):
    
    #This function supposes the input data is
    #a numpy.ndarray with shape (observations, axes)
    
    pca_start = time.time()
    
    #Exiting if the information amount to be kept is 
    #out of range
    if info_amount <= 0 or info_amount >= 1:
        return "Information amount out of range, must be in (0:1)"
    
    '''
    print("Original matrix:")
    print(input_data)
    print("\n")
    '''
    
    #computing mean of each column/feature
    #adding an extra axis 
    mean = np.sum(input_data, axis=0)
    mean = mean[:, np.newaxis]
    mean = mean / input_data.shape[0]
    
    #we use h tu substract the mean out of 
    #it's representing column
    h = np.ones((input_data.shape[0], 1))
    
    #Substracting the mean from data
    data_no_mean = input_data - np.dot(h, mean.T)
    '''
    print("Original matrix with mean substracted:")
    print(data_no_mean)
    print("\n")
    '''
    
    #Computing the cross-covavriance matrix
    #we use n-1 instead of n to account for Bessel's bias correction
    #The easy way: covariance_matrix = np.cov(input_data)
    covariance_matrix = np.dot(data_no_mean.T, data_no_mean) / (input_data.shape[0] -1)
    
    '''
    print("Covariance matrix:")
    print(covariance_matrix)
    print("\n")
    '''
    
    #Extracting the eigen values and vectors of 
    #the cross-covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    
    eigen_values = np.abs(eigen_values)
    eigen_vectors = np.abs(eigen_vectors)

    #determining indexes of eigen values sorted in descending order
    indexes = eigen_values.argsort()[::-1]  
    
    #eigen values sorted in descending order 
    eigen_values[::-1].sort()
    
    #ratio to know when the specified amount of information
    #has been reached
    ratio = 0
    
    #index of the last principal component kept according to 
    #the specified amount of information
    pc_index = 0
    
    #determining the components to be kept according to
    #the specified amount of information
    EIGEN_SUM = np.sum(eigen_values)
    while(ratio < info_amount):
        pc_index = pc_index + 1
        ratio = np.sum(eigen_values[0:pc_index]) / EIGEN_SUM
        
    '''
    print(ratio)
    print(pc_index)
    '''
    
    #computing the transform matrix ordering in a 
    #descending order the eigen vectors
    #by their correspondig eigen values
    transform_matrix = eigen_vectors[:,indexes[0 : pc_index]]

    '''
    print("Transform matrix:")
    print(transform_matrix)
    print("\n")
    '''
    '''
    print("Original matrix in transformed space:")
    print(np.dot(input_data, transform_matrix))
    '''
    
    #printing out the time spent in this function
    pca_end = time.time()
    print("Time spent in pca function [min]: ", (pca_end - pca_start) / 60)
    
    #returning the input data in the new basis
    return (np.dot(input_data, transform_matrix), eigen_values)