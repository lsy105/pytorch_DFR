import os
import numpy as np
class QParams(object):
    def __init__(self, name, directory='params'):
        self.name = name
        self.dir = directory
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def FindMaxMin(self, vec):
        vec = vec.flatten()
        minval = 0
        maxval = 0
        for item in vec:
            minval = min(minval, item)
            maxval = max(maxval, item)
        return minval, maxval

    def ChooseQuantization(self, minval, maxval, num_bits=8):
        minval = min(minval, 0.0)
        maxval = max(maxval, 0.0)
        qmin = 0
        qmax = 2**num_bits - 1
        scale = (maxval - minval) / (qmax - qmin)
        initial_zero_point = qmin - minval / scale
        Q_zero_point = 0
        if initial_zero_point < qmin:
            Q_zero_point = qmin
        elif initial_zero_point > qmax:
            Q_zero_point = qmax
        else:
            Q_zero_point = round(initial_zero_point)
        return scale, Q_zero_point

    def ChooseLogQuantization(self, minval, maxval, num_bits):
        minval = min(minval, 0.0)
        maxval = max(maxval, 0.0)
        qmin = 0
        qmax = 2**num_bits - 1
        scale = (maxval - minval) / (qmax - qmin)
        scale = np.power(2, np.round(np.log2(scale)))
        initial_zero_point = qmin - minval / scale
        Q_zero_point = 0
        if initial_zero_point < qmin:
            Q_zero_point = qmin
        elif initial_zero_point > qmax:
            Q_zero_point = qmax
        else:
            Q_zero_point = round(initial_zero_point)
        return scale, Q_zero_point

    def ToCArray(self, vec, row, col):
        result = ''
        for r in range(row):
            array = '{'
            for c in range(col):
                array += str(int(vec[r * col + c])) + ','
            array += '},\n'
            result += array
        return result

    def ToCFloatArray(self, vec, row, col):
        result = ''
        for r in range(row):
            array = '{'
            for c in range(col):
                array += '%.6f,' % vec[r * col + c]
            array += '},\n'
            result += array
        return result
    
    def GetFloat(self, A, row=None, col=None):
        if A is not None:
            A = A.flatten()
        path = self.dir + '/' + self.name
        fin = open(path + '.txt', 'w')
        if A is not None:
            vec = []
            for idx, item in enumerate(A):
                vec.append(item)
            result = self.ToCFloatArray(vec, row, col)
            print(result, file=fin)
        fin.close()

    def Quantize(self, A=None, minval=None, maxval=None, row=None, col=None, num_bits=8):
        if A is not None:
            A = A.flatten()
        path = self.dir + '/' + self.name
        fin = open(path + '.txt', 'w')
        if minval is None and maxval is None:
            minval, maxval = self.FindMaxMin(A)
        scale, Q_zero_point = self.ChooseQuantization(minval, maxval, num_bits)
        if A is not None:
            vec = []
            for idx, item in enumerate(A):
                q = Q_zero_point + item / scale
                q = max(0.0, min(2**num_bits - 1, q))
                vec.append(q)
            result = self.ToCArray(vec, row, col)
            print(result, file=fin)
        print("parameters:", file=fin) 
        print(str(scale) + " " + str(Q_zero_point), file=fin) 
        fin.close()
        return scale, Q_zero_point

    def LogQuantize(self, A=None, minval=None, maxval=None, row=None, col=None):
        if A is not None:
            A = A.flatten()
        path = self.dir + '/' + self.name
        fin = open(path + '.txt', 'w')
        if minval is None and maxval is None:
            minval, maxval = self.FindMaxMin(A)
        scale, Q_zero_point = self.ChooseLogQuantization(minval, maxval)
        if A is not None:
            vec = []
            for idx, item in enumerate(A):
                q = Q_zero_point + item / scale
                q = max(0.0, min(255.0, q))
                vec.append(q)
            result = self.ToCArray(vec, row, col)
            print(result, file=fin)
        print("parameters:", file=fin) 
        print(str(scale) + " " + str(Q_zero_point), file=fin) 
        fin.close()
        return scale, Q_zero_point

    def BinaryQuantize(self, A=None, minval=None, maxval=None, row=None, col=None):
        if A is not None:
            A = A.flatten()
        path = self.dir + '/' + self.name
        fin = open(path + '.txt', 'w')
        if A is not None:
            vec = []
            for idx, item in enumerate(A):
                q = np.sign(item)
                if q == -1:
                    q = 0
                vec.append(q)
            result = self.ToCArray(vec, row, col)
            print(result, file=fin)
        fin.close()

