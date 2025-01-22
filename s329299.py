# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in fornp.mulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray: 
    
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray: 
    return  1

def f3(x: np.ndarray) -> np.ndarray: 
    return np.add(np.subtract(np.multiply(-(x[1]), abs(np.multiply(x[1], abs(x[1])))), -(np.subtract(np.cos(np.multiply(x[0], x[0])), x[2]))), np.subtract(np.add(np.subtract(-(x[2]), -(np.add(np.subtract(np.multiply(-(x[2]), 0.931), -(max(np.multiply(x[0], x[0]), np.subtract(x[0], x[0])))), np.sqrt(np.subtract(x[0], x[0]))))), np.sqrt(max(0.0000000001,np.divide(np.tan(np.tan(-(-0.751))), -0.155)))), -(max(np.multiply(x[0], x[0]), np.subtract(x[0], x[0])))))

def f4(x: np.ndarray) -> np.ndarray: 

    def fake_expa(a, x):
        if np.isscalar(a) and np.isscalar(x):
            if a < 0.001:
                a = 0.001
            if a > 10:
                a = 10
            if x < 0.001:
                x = 0.001
            if x > 8:
                x = 8
            result = a**x
            return min(result, 10**8)
        else:
            return np.array([fake_expa(ai, xi) for ai, xi in zip(a, x)])
        
    def fake_modf(x):
        if np.isscalar(x):
            return x - int(x)
        else:
            return np.array([fake_modf(xi) for xi in x])
    return  np.add(np.add(np.cos(x[1]), np.add(np.add(0.8696509541, np.add(np.add(np.cos(np.negative(x[1])), np.cos(x[1])), np.add(np.cos(x[1]), np.add(np.tan(0.8696509541), np.cos(x[1]))))), np.add(np.cos(np.negative(x[1])), fake_modf(np.cos(np.negative(x[1])))))), fake_expa(np.tan(0.8696509541), np.tan(0.8696509541)))

def f5(x: np.ndarray) -> np.ndarray: 
    return np.multiply(np.multiply(np.subtract(np.add(x[0], x[1]), np.add(x[1], x[0])), np.multiply(np.divide(x[1], x[0]), np.subtract(0.384, 0.629))), np.subtract(-0.720, np.subtract(np.add(-0.502, x[1]), np.subtract(x[1], -0.558))))

def f6(x: np.ndarray) -> np.ndarray: 
    return np.subtract(x[1], np.multiply(np.subtract(x[0], x[1]), 0.695))

def f7(x: np.ndarray) -> np.ndarray: 
        
    return  np.add(np.add(np.cos(x[1]), np.add(np.add(np.cos(x[1]), np.add(np.add(np.cos(x[1]), np.add(np.cos(x[1]), np.add(np.tan(0.990), np.add(np.cos(x[1]), np.cos(x[1]))))), np.cos(x[1]))), 0.939)), np.tan(0.683))

def f8(x: np.ndarray) -> np.ndarray: 
    return np.multiply(max(max(np.multiply(np.multiply(x[5], np.abs(np.subtract(-0.37017066891978700482, x[5]))), np.negative(np.subtract(np.cos(np.sin(np.negative(x[5]))), x[5]))), np.abs(max(np.multiply(max(np.abs(x[4]), np.abs(-0.48552610290267739224)), np.negative(np.multiply(x[5], np.abs(x[5])))), 0.75612783915036763105))), np.abs(0.05160937174649449233)), np.subtract(np.divide(x[5], 0.05160937174649449233), np.subtract(np.add(x[5], x[0]), np.multiply(x[5], np.abs(np.negative(x[5]))))))
