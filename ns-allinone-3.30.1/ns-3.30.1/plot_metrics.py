#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math
import statistics

args = sys.argv[1:]
print("args", args)

# LAST_N = 100_000
LAST_N = 1_000_000_000

def windowize(data, n=10000):
	new_data = []
	for i in range(int(math.floor(len(data)/n))):
		new_data.append(data[i*n:(i+1)*n])
	return [statistics.mean(item) for item in new_data]

max_x = float("inf")
max_y = float("inf")

# for arg in args:
if len(args) > 1:
	max_x = float(args[1])
if len(args) > 2:
	max_y = float(args[2])

arg = args[0]
print("Yeah, plotting")

df = pd.read_csv(arg, sep=";", header=0)

last_part = arg.split("/")[-1].split(".")[0]
# print("last_part", last_part)

extracted_data = {}
for col in df:
	extracted_data[col] = df[col].tolist()

for col in extracted_data.keys():
	data = np.array(extracted_data[col])[-LAST_N:]
	x = np.array(list(range(len(data))), dtype=np.float64)
	x /= 1000000.0
	plt.plot(x, data, linestyle="", marker=",", label=col)

	plt.plot(windowize(x), windowize(data), label=col+" averaged")

	intermed_path = "/".join(arg.split("/")[:-1])+"/plots/"
	os.makedirs(intermed_path, exist_ok=True)
	# plt.grid(True)
	plt.tight_layout()
	if max_x < float("inf"):
		plt.xlim(0,min(max_x, np.max(x)))
	if max_y < float("inf"):
		plt.ylim(0,min(max_y, np.max(data)))
	plt.xlabel("millions of training flows")
	plt.ylabel(col)
	plt.legend(loc="upper right")
	plt.tight_layout()
	plt.savefig(intermed_path+last_part+"_"+col+".png", bbox_inches = 'tight', pad_inches = 0, dpi=200)
	plt.close()