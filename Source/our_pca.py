import numpy as np
import time

def our_pca(input_data: np.ndarray):
    
    #This function supposes the input data is
    #a numpy.ndarray with shape (observations, axes)
    
    pca_start = time.time()
    
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

    #computing the transform matrix ordering in a 
    #descending order the eigen vectors
    #by their correspondig eigen values
    idexes = eigen_values.argsort()[::-1]   
    transform_matrix = eigen_vectors[:,idexes]

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
    return (np.dot(input_data, transform_matrix), np.sort(eigen_values)[::-1])