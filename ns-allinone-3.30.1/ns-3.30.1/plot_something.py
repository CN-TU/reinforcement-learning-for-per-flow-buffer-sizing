#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

paths = sys.argv[1:]

for path in paths:
	with open(path, "r") as f:
		lines = f.readlines()

	lines = [item for item in lines if item!="" and item != "\n"]
	tuples = [[float(subitem) for subitem in item.split(" ")] for item in lines]
	results = list(zip(*tuples))
	ys = results[1:]
	for i, y in enumerate(ys):
		# print("i", i, "y", y)
		if i > 0:
			plt.twinx()
		plt.plot(results[0], y)
	appropriate_dir = "/".join(path.split("/")[:-1])+"/"
	file_name = ".".join(path.split("/")[-1].split(".")[:-1])+".pdf"
	plt.savefig(appropriate_dir+file_name)
	# plt.show()
	plt.close()