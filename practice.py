import numpy as np
import torch


def relu(x):
	return np.maximum(0, x)

class Practice:
	print("Hello World")
	

	a = np.array([1, 2, 3, 4])
	b = np.array([10, 20, 30, 40])
	
	print("a:", a)
	print("b:", b)

	res = a + b
	print("adding", res)
	

	res2 = a - b
	print("subtracting:", res2)

	
	res3 = a * b
	print("multiply element wise: ", res3)

	res4 = b / a
	print("divide b by a: ", res4)


	res5 = np.square(a)
	print("square every elem in a", res5)


	angles = np.array([0, np.pi/4, np.pi/2, np.pi])
	
	res6 = np.sin(angles)
	print(res6)

	res7 = np.cos(angles)
	print(res7)

	res8 = np.exp(angles)
	print(res8)

	res9 = np.log(angles + 1)
	print(res9)

	arr = np.array([2, 5, 8, 10, 3, 7])

	print("elems greater than 5:", arr[arr > 5])

	print("elems divisible by 2:", arr[(arr % 2) == 0])
	arr[arr < 5] = 0
	print("replacing all values less than 5 with 0", arr)

	mat = np.array([[1, 2, 3], [4, 5, 6]])

	print("mat:", mat)

	res10 = np.sum(mat)
	print("sum: ", res10)

	res11 = np.mean(mat, axis=1)
	print("Mean of each row:", res11)

	res12 = np.max(mat, axis=0)
	print("max of each column:", res12)

	res13 = np.std(mat)
	print("std dev or arr", res13)

	# normalize vector
	data = np.array([10, 20, 30, 40, 50])
	
	print("This is data:", data)

	std = np.std(data)
	mean = np.mean(data)
	
	norm = (data - mean) / std
	print("normalized vector data:", norm)

	np_arr = np.array([-3, -1, 0, 2, 4])
	print("relu'd arr:", relu(np_arr))



