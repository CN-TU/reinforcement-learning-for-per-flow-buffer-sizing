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
		# if i > 0:
		# 	plt.twinx()
		if i == 0 or not ("FqCoDelQueueDisc" in path):
			if i==0:
				label = "queue size"
			else:
				label = "max. queue size"
			plt.plot(results[0], y, label=label)
	plt.xlabel("time (s)")
	plt.ylabel("queue length (packets)")
	plt.legend()
	appropriate_dir = "/".join(path.split("/")[:-1])+"/"
	file_name = ".".join(path.split("/")[-1].split(".")[:-1])+".pdf"

	plt.tight_layout()
	plt.savefig(appropriate_dir+file_name, bbox_inches = 'tight', pad_inches = 0)
	# plt.show()
	plt.close()