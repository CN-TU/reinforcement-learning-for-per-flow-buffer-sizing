#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import statistics
import math
import numpy as np

def windowize(data, n=10):
	new_data = []
	for i in range(int(math.floor(len(data)/n))):
		new_data.append(data[i*n:(i+1)*n])
	return [statistics.median(item) for item in new_data]

cc_int = re.compile("(cc\_(\d+)\_)")

paths = sys.argv[1:]

cc_mapping = {0: "New Reno", 1: "Bic"}
full_names = {"bw": "bandwidth", "delay": "delay"}
units = {"bw": "Mbit/s", "delay": "ms"}



for path in paths:
	df = pd.read_csv(path, sep=";", header=0)

	relevant_thing = "bw" if "bw" in path else "delay"

	match = cc_int.search(path)
	cc_string = match.group(1)
	cc_name = cc_mapping[int(match.group(2))]
	print(cc_name, relevant_thing)
	path = path.replace(cc_string, f"{cc_name}_")
	path = path.replace("bw", full_names["bw"])
	path = path.replace("delay", full_names["delay"])

	average_queue_length = df["average_queue_length"].tolist()
	average_max_queue_length = df["average_max_queue_length"].tolist()
	x = df[relevant_thing].tolist()

	print("correlation max", np.corrcoef(x, average_max_queue_length)[0,1])
	print("correlation avg", np.corrcoef(x, average_queue_length)[0,1])
	print("avg max", statistics.mean(average_max_queue_length))
	print("avg avg", statistics.mean(average_queue_length))

	# average_queue_length = windowize(df["average_queue_length"].tolist())
	# average_max_queue_length = windowize(df["average_max_queue_length"].tolist())
	# x = windowize(df[relevant_thing].tolist())

	plt.plot(x, average_queue_length, label="queue length")
	plt.plot(x, average_max_queue_length, label="max queue length")
	appropriate_dir = "/".join(path.split("/")[:-1])+"/plots/"
	os.makedirs(appropriate_dir, exist_ok=True)

	file_name = ".".join(path.split("/")[-1].split(".")[:-1])+".pdf"
	plt.grid(True)
	plt.xlabel(f"{full_names[relevant_thing]} ({units[relevant_thing]})")
	plt.ylabel("avg. queue size (packets)")
	plt.legend()
	plt.title(cc_name)
	plt.savefig(appropriate_dir+file_name)
	plt.close()
