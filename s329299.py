

import numpy as np


def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray:  
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray: 
    return ((756207.936*x[2]+1218853.827*x[1]+1747375.474*np.arctan(0.4426*x[2]))*0.994*np.cos(0.502*x[0])+1247989.7133*np.sin(0.0269*x[1])+3000251.7633*np.arctan(1.089*x[0]))


def f3(x: np.ndarray) -> np.ndarray:
    return np.round(((((x[2] * -3.5011236) - -4.2145568) + (-(x[1]) * np.square(x[1]))) - ((9.8732836e-07 - x[0]) * (x[0] + x[0]))) - 0.2143514, 7)

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
    return (np.cos(0.560802 * x[1]) * np.exp(1.27210 * x[0]) * np.arctan(0.3330 * x[0]) * (8.11230e-10 * 3.71410e-11 * 1.90540e-11)) / (np.exp(-0.98120 * x[1]) * (9.14440e-10 * 3.12660e-10 * 1.04880 * x[0]))

def f6(x: np.ndarray) -> np.ndarray: 
    return np.subtract(x[1], np.multiply(np.subtract(x[0], x[1]), 0.695))

def f7(x: np.ndarray) -> np.ndarray:
    term1 = (x[1] * x[0]) * (1.298260 - np.sin((0.704873 - (x[1] * x[0])) * -0.125456))
    term2 = ((0.941172 / np.power(0.919048, np.remainder(x[0], 0.179080))) *
             np.power(np.power(1.007577, np.remainder(x[0], -0.545329)), ((x[1] - 0.545329) * x[0]))
            ) - np.abs(np.arctan(np.arctan(x[0] - x[1])))
    term3 = (1.774334 * (0.543759 + np.power(1.027270, np.remainder(-1.646242, x[1])))) + \
            np.exp(np.power(0.970442, np.remainder(x[0], 0.965901)) + np.power(0.956161, (x[0] - x[1])))
    return np.exp(np.square(0.194125 * (term1 + term2 * term3)))

def f8(x: np.ndarray) -> np.ndarray: 
    return 0.001*np.abs(x[2])+2933.8781*np.tan(0.28*x[5])-574.14*np.abs(x[4])+409.5379*np.cos(5.5364e-7*x[0])+406.5744*np.cos(0.084*x[1])-1848.7071*np.sin(0.641*x[5])-624.0142*np.cos(1.0164*x[4])+45.4951*x[3]
