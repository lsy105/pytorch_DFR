import numpy as np
def To1DArray(arr):
    result = ""
    row = arr.shape
    for num in arr:
        result += str(num) + ','
    return result

def To2DArray(arr):
    result = ""
    row, col = arr.shape
    for r in range(row):
        array = '{'
        for c in range(col):
            array += str(arr[r][c]) + ','
        array += '},\n'
        result += array
    return result

def To3DArray(arr):
    result = ""
    ch, row, col = arr.shape
    for i in range(ch):
        array = '{'
        array += To2DArray(arr[i])
        array += '},\n'
        result += array
    return result

def To4DArray(arr):
    result = ""
    batch, ch, row, col = arr.shape
    for i in range(batch):
        array = '{'
        array += To3DArray(arr[i])
        array += '},\n'
        result += array
    return result
    

def ToCArray(arr):
        result = ''
        dim = len(arr.shape) 
        if dim == 2:
            return To2DArray(arr)
        elif dim == 3:
            return To3DArray(arr)
        elif dim == 4:
            return To4DArray(arr)
        else:
            return To1DArray(arr)
"""
test = np.random.rand(3,2,2,2)
f1 = open('3d.txt', 'w')
f1.write(ToCArray(test))
f1.close()
"""
