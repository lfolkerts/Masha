import time
import random
import numpy as np
from time import time

from concrete import fhe

rng = np.random.default_rng()
compile_size = 150 

b_sim = True
b_run = True

def encrypted_compare(y_pred, ranks):
    args = y_pred
    #argmax
    argcnt = arr_size

    for argi in range(argcnt):
        for argj in range(argi+1, argcnt):
            dist = (args[argi] > args[argj])
            ranks[argi] += dist
            ranks[argj] += (1-dist)
    return ranks

print(f"Compile Size: {compile_size}")
for arr_size in range(4, 24):
    compiler = fhe.Compiler(encrypted_compare, {"y_pred": "encrypted", "ranks":"encrypted"})
    y_pred = rng.integers(0,8,size=(compile_size,arr_size)) 
    
    st = time() 
    inputset = []
    for isi in range(compile_size):
        inputset.append((y_pred[isi], np.zeros(arr_size).astype(int)))
    circuit = compiler.compile(inputset)
    print(f"Compile Time: {(time()-st)} for vector length {arr_size}")
    
    st=time()
    if b_sim:    
        for si in range(compile_size):
            circuit.simulate(y_pred[si], np.zeros(arr_size).astype(int))
        print(f"Sim Time: {(time()-st)} for vector length {arr_size}")
   
    st=time()
    if b_run:
        homomorphic_evaluation = circuit.encrypt_run_decrypt(y_pred[0], np.zeros(arr_size).astype(int))
        print(f"Run Time: {(time()-st)} for vector length {arr_size}")
    print()

