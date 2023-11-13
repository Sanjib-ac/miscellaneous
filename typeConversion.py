import numpy as np
import csv
from tqdm import tqdm
import cupy as cp
import time
from datetime import datetime
from sys import getsizeof
import matplotlib.pyplot as plt


def generateFile(filename: str, datatype=np.float64, size=1000001):
    """Generate a .csv file with float64 """
    datatype = datatype
    # size = size
    f = open("output.csv", "a")
    rng = np.random.default_rng()
    x = [rng.random(size=size, dtype=np.float64)]
    y = np.sin(x, dtype=np.float64)
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(y)
    print(f'Data:{y[0][0]}, datatype: {y[0].dtype}; data :{x[0][0]}, datatype: {x[0].dtype}')


def readFile():
    data = np.loadtxt("output.csv", delimiter=",")
    for i in range(10):
        print(f'Data: {data[0]}, type: {data[0].dtype}')


def typeConvF64_F16(times=1):
    t_start = time.time()
    data = np.loadtxt("output.csv", delimiter=",", dtype=np.float64)
    print(f'No data points - {len(data)}; type: {data.dtype}')
    # output_filename = datetime.now().strftime("%d_%m_%H_%M_%S") + '.csv'
    output_filename = datetime.now().strftime("%H_%M_%S") + '.csv'
    # f_open = open('./output/' + output_filename, 'a')

    for _ in tqdm(range(times)):
        data_16 = [data.astype(dtype=np.float16)]
        # with open('./output/' + output_filename, 'a') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerows(data_16)

    print(f'CPU time taken for {times} times conversion:{time.time() - t_start}, type: {data_16[0].dtype}')
    # print(f'Data before conversion{data[0]}, type={data[0].dtype} '
    #       f'after conversion:{data_16[0][0]}, type={data_16[0][0].dtype}')

    t_start = time.time()
    data_cpu = np.loadtxt("output.csv", delimiter=",", dtype=np.float64)

    for _ in tqdm(range(times)):
        data_gpu = cp.asarray(data_cpu)
        data_gpu_16 = data_gpu.astype(dtype=np.float16)
        data_back_cpu = cp.asnumpy(data_gpu_16)
    # print(f'data_cpu:{data_cpu[0]}')
    print(f'GPU time taken for {times} times conversion:{time.time() - t_start}, type: {data_back_cpu.dtype}')


'Main function'
if __name__ == "__main__":
    # generateFile('output.csv')
    # readFile()
    typeConvF64_F16(times=1000)
