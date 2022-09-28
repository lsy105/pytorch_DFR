def Test(A):
    for i in range(1, len(A)):
        if A[i] < A[i - 1]:
            j = i
            while j > 0 and A[j] < A[j - 1]:
                temp = A[j - 1]
                A[j - 1] = A[j]
                A[j] = temp
                j -= 1
    return A

A = [1,2,3,4,5,6,8,9,13,10,11,7,12]
print(A)
print(Test(A))
x = 123123.1231231
print('%.2f' % x)
