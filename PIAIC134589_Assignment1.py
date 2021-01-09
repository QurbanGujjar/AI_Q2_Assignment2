#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[3]:


import numpy as np


# 2. Create a null vector of size 10 

# In[5]:


np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[ ]:


np.arange(10,49)


# 4. Find the shape of previous array in question 3

# In[ ]:


x=np.arange(10,49)
x.shape


# 5. Print the type of the previous array in question 3

# In[ ]:


print(x.dtype)


# 6. Print the numpy version and the configuration
# 

# In[ ]:


np.__version__


# 7. Print the dimension of the array in question 3
# 

# In[ ]:


print(x.ndim)


# 8. Create a boolean array with all the True values

# In[ ]:


np.full((5),True,dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[ ]:


# import numpy as np
np.arange(0,12).reshape(2,-1)


# 10. Create a three dimensional array
# 
# 

# In[ ]:


np.arange(0,12).reshape(3,2,-1)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[ ]:


z=np.arange(0,12)
z[: : -1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[ ]:


nullVactor= np.zeros((10))
nullVactor[4]=1
nullVactor

# 13. Create a 3x3 identity matrix
# In[ ]:


np.eye(3,3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[ ]:


arr = np.array([1, 2, 3, 4, 5],dtype='f8')
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[ ]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[ ]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
c=arr1==arr2
c


# 17. Extract all odd numbers from arr with values(0-9)

# In[ ]:


oddValues=np.arange(0,10)
oddValues[oddValues%2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[ ]:


oddValues=np.arange(0,10)
oddValues[oddValues%2==1]=-1
oddValues


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[ ]:


arr = np.arange(10)
arr[5:9]=12
arr

# 20. Create a 2d array with 1 on the border and 0 inside
# In[28]:


# x=np.arange(0,36).reshape(1,6,-1)
x=np.ones((5,5))
x[1:-1,1:-1] = 0
x


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[32]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[36]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,:,:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[40]:


arr=np.arange(10).reshape(2,-1)
arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[42]:


arr=np.arange(10).reshape(2,-1)
arr[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[61]:


colm=np.arange(10).reshape(2,-1)
colm[0:2,3:4]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[71]:


random=np.random.rand(10,10)
# print(random)
print(random.max())
print(random.min())


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[76]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.union1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[78]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(np.in1d(a, b))[0]


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[166]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[names!='Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[168]:


mask = (names != 'Bob') & (names != 'Will')
data[mask]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[164]:


np.arange(1,16).reshape(5,3)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[147]:


arr = np.arange(1,17).reshape(2,2,4)
arr


# 33. Swap axes of the array you created in Question 32

# In[163]:


np.swapaxes(arr,1,0)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[145]:


arr=np.random.rand(10)
sqrt=np.sqrt(arr)
sqrt[sqrt<0.5]=0
sqrt


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[139]:


a1=np.random.rand(12)
a2=np.random.rand(12)
a3=np.maximum(a1,a2)
a3


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[129]:


names = np.array(['Will', 'Bob','Bob', 'Joe', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[124]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a, b)
result
get_ipython().run_line_magic('pinfo', 'np.setdiff1d')


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[112]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
sampleArray
SA=np.delete(sampleArray,2,1)
sampleArray = np.insert(SA, 1, newColumn, axis=1)
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[80]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[122]:


arr=np.random.random(20)
arr.cumsum()


# In[ ]:




