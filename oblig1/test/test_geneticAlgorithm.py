import random

from src.genericAlgorithm import *

# Test if crash
a = b = None
n = 1000

for k in range(n):
    a = [i for i in range(1,random.randint(10, 1000))]
    b = a[:]
    while a == b:
        random.shuffle(a)
        random.shuffle(b)
    # print("here")
    tmp = pmx(a,b)
    if None in tmp:
        print(f"Err: len(a):{len(a)}, len(b):{len(b)}, k:{k}\n    a:{a}\n    b:{b}\n    tmp:{tmp}")
    if k % (n//10) == 0:
        print(f"{k}")