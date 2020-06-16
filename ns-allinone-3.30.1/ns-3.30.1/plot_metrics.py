#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math
import statistics

args = sys.argv[1:]

# LAST_N = 100_000
LAST_N = 1_000_000_000

def windowize(data, n=400):
	new_data = []
	for i in range(int(math.floor(len(data)/n))):
		new_data.append(data[i*n:(i+1)*n])
	return [statistics.mean(item) for item in new_data]

for arg in args:
	print("Yeah, plotting")

	df = pd.read_csv(arg, sep=";", header=0)

	last_part = arg.split("/")[-1].split(".")[0]
	# print("last_part", last_part)

	extracted_data = {}
	for col in df:
		extracted_data[col] = df[col].tolist()

	for col in extracted_data.keys():
		data = np.array(extracted_data[col])[-LAST_N:]
		x = np.array(list(range(len(data))))
		plt.plot(x, data, linestyle="", marker=",")

		plt.plot(windowize(x), windowize(data))

		intermed_path = "/".join(arg.split("/")[:-1])+"/plots/"
		os.makedirs(intermed_path, exist_ok=True)
		# plt.grid(True)
		plt.tight_layout()
		plt.savefig(intermed_path+last_part+"_"+col+".png", bbox_inches = 'tight', pad_inches = 0)
		plt.close()