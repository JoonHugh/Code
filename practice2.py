import numpy as np

class Practice2:
	arr = np.array([1, 2, 3, 4, 5])
	arr2 = np.array([[1, 2, 3], [4, 5, 6]])
	arr3 = np.array([True, False, True])
	
	print("numpy sum:", np.sum(arr))
	print("numpy mean:", np.mean(arr))
	print("numpy max:", np.max(arr))
	print("numpy min:", np.min(arr))
	print("np.std:", np.std(arr))
	print("numpy product:", np.prod(arr))

	print("numpy sum columns:", np.sum(arr2, axis=0))
	print("numpy sum rows:", np.sum(arr2, axis=1))

	print("cumaltative sum:", np.cumsum(arr))
	print("cumalative product:", np.cumprod(arr))

	print("Is true in array?:", np.any(arr3))
	print("Is the array all true?", np.all(arr3))
	
	print("----------------")
	print("PRACTICE:")
	
	np_arr = np.array([3, 6, 9, 12, 15])
	
	print("np_arr:", np_arr)
	print("sum:", np.sum(np_arr))
	print("mean:", np.mean(np_arr))
	print("std dev:", np.std(np_arr))
	print("product of all elems:", np.prod(np_arr))
	print("max and min respectively:", np.max(np_arr), np.min(np_arr))
	
	print("-----------------\n PRACTICE 2")
	
	mat = np.array([[2, 4, 6], [1, 3, 5]])
	print("mat:", mat)
	print("sum of each column:", np.sum(mat, axis=0))
	print("mean of each row:", np.mean(mat, axis=1))
	print("max value per column:", np.max(mat, axis=0))
	
	print("-----------------\nPRACTICE3")
	x = np.array([1, 2, 3, 4, 5])
	print("x:", x)
	print("cum sum:", np.cumsum(x))
	print("cumalative prod:", np.cumprod(x))

	print("---------------\nPRACTICE 4")
	nums = np.array([5, 10, 15, 20])
	print("nums:", nums)
	print("All elems divisible by 5?", np.all((nums % 5) == 0))
	print("Any elems greater than 18?", np.any(nums > 18))
	print("Al elems are even?", np.all(nums % 2))

	print("-------------\nBONUS:")
	
	np.random.seed(0)
	a = np.random.randint(1, 101, (5, 5))
	print("a", a)
	
	print("row with highest sum:", np.max(np.sum(a, axis=1)))
	print("column with lowest mean:", np.min(np.mean(a, axis=0)))
	print("norm each column:", (a - np.mean(a, axis=0)) / np.std(a))
