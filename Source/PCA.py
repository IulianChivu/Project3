import numpy as np
import time

print("\n")


start = time.time()

'''
m = [[1, 4, 21, -9, 1], [7, 5, 5, 8, 0], 
     [9, 4, 2, 12, 1], [10, 3, 0, 3, 0], [12, 5, 1, -2, 0]
    ]
m = [[1, 1, 1], [1, 2, 1], [1, 3, 2], [1, 4, 3]]
'''

m = [[5, -5], [4, -3], [3,-4], [-4,3], [-3,4], [-5, 5]]



m = np.array(m)
print("original matrix:")
print(m)
print("\n")

mean = list()

for p in range(m.shape[1]):
    mean.append(np.sum(m[:, p]) / m.shape[0])
    
mean = (np.array(mean))
mean = mean[:, np.newaxis]
#print(mean)
#print("\n")

h = np.ones((m.shape[0], 1))
#print(h)
#print("\n")
#print(np.dot(h, mean.T))

#print(m)
m_no_mean = m - np.dot(h, mean.T)
print("Original matrix with mean substracted:")
print(m_no_mean)
print("\n")

covariance_matrix = np.dot(m_no_mean.T, m_no_mean) / (m.shape[0] - 1)
print("Covariance matrix:")
print(covariance_matrix)
print("\n")

eigen_values, eigen_vectors= np.linalg.eig(covariance_matrix)

idexes = eigen_values.argsort()[::-1]   
transform_matrix = eigen_vectors[:,idexes]

print("Transform matrix:")
print(transform_matrix)
print("\n")

print("Matrix in transformed space:")
print(np.dot(m, transform_matrix))


end = time.time()
print("\nTime elapsed: "+str(end-start)+" seconds")
