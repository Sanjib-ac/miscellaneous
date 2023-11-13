#
# # A bit late, but you can make this happen easily by defining a text item that ignores item transforms. It is important that you use setParentItem with this, similar to how you would with a legend item:
#
# import pyqtgraph as pg
#
# pg.mkQApp()
#
# plot = pg.PlotWidget()
# moving_item = pg.TextItem("Moving", anchor=(0.5, 0.5))
# still_item = pg.TextItem("Still", anchor=(0.5, 0.5))
#
# still_item.setFlag(still_item.GraphicsItemFlag.ItemIgnoresTransformations)
# # ^^^ This line is necessary
#
# plot.addItem(moving_item, ignoreBounds=True)
# # Use this instead of `plot.addItem`
# still_item.setParentItem(plot.plotItem)
#
# # This position will be in pixels, not scene coordinates since transforms are ignored.
# # You can use helpers like `mapFromScene()` etc. to translate between pixels
# # and viewbox coordinates
# still_item.setPos(300, 300)
#
# plot.show()
#
# pg.exec()
# from time import sleep
# import sys
#
# for i in range(21):
#     sys.stdout.write('\r')
#     # the exact output you're looking for:
#     sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
#     sys.stdout.flush()
#     sleep(0.25)
import time

import numpy as np
import csv
from sys import getsizeof
import matplotlib.pyplot as plt

# f = open("output.csv", "a")
# f1 = open("output1.csv", "a")
# size = 1000001
# rng = np.random.default_rng()
# x = [rng.random(size=size, dtype=np.float64)]
# x1 = rng.random(size=size, dtype=np.float32)
# y = np.sin(x, dtype=np.float64)
# print(getsizeof(x))
# # for i in range(len(x)-1):
# #     f.write(str(y[i]) + ',')
# #     f1.write(str(x1[i]) + ',')
#
# with open("output.csv", 'w') as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)
#
#     # writing the data rows
#     csvwriter.writerows(x)
# print(getsizeof(f))
# # with open('outfile.dat', 'a') as outfile:
# #     x.tofile(outfile)
# f.close()
# f1.close()
# print(f'')
# # plt.plot(y)
# # plt.show()

#
import numpy as np

rng = np.random.default_rng()
x = rng.random(size=1, dtype=np.float64)
y = x.astype(np.float16)
z = x.astype(np.int16)
print(f'Data:{x}, dtatype: {x.dtype}; data :{y}, datatype: {y.dtype}, data :{z}, datatype: {z.dtype}')

times = 1000
# t_start = time.time()
# for i in range(times):
#     data = np.loadtxt("output.csv", delimiter=",", dtype=np.float64)
#     data_16 = data.astype(dtype=np.float16)
#
#     # for j in range(len(data) - 1):
#     #     print(f'{data[j]}, {data[j].dtype}')
#     #     print(f'{data_16[j]}, {data_16[j].dtype}')
#
# print(f'Time taken for {times} loops:{time.time() - t_start}')

from tqdm import tqdm
import cupy as cp
t_start = time.time()
data = np.loadtxt("output.csv", delimiter=",", dtype=np.float64)
for i in tqdm(range(times)):
    data_16 = data.astype(dtype=np.float16)
print('Done')
print(f'Time taken for {times} loops:{time.time() - t_start}')

t_start = time.time()
data_cpu = np.loadtxt("output.csv", delimiter=",", dtype=np.float64)
for i in tqdm(range(times)):

    data_gpu = cp.asarray(data_cpu)
    data_gpu_16 = data_gpu.astype(dtype=np.float16)
    data_back_cpu = cp.asnumpy(data_gpu_16)
print('Done')
# print(f'data_cpu:{data_cpu[0]}')
print(f'Time taken for {times} loops:{time.time() - t_start}')
