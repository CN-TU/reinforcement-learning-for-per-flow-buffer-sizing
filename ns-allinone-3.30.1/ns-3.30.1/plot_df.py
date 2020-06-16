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
	print("path", path)
	if os.path.isdir(path):
		continue
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
	average_throughput = np.array(df["average_throughput"].tolist())/1000000*8
	average_max_queue_length = df["average_max_queue_length"].tolist()
	x = df[relevant_thing].tolist()

	print("correlation max", np.corrcoef(x, average_max_queue_length)[0,1])
	# print("correlation avg", np.corrcoef(x, average_queue_length)[0,1])
	print("avg max", statistics.mean(average_max_queue_length))
	# print("avg avg", statistics.mean(average_queue_length))
	print("throughput avg", np.mean(average_throughput))
	print("queue avg", np.mean(average_queue_length))

	# average_queue_length = windowize(df["average_queue_length"].tolist())
	# average_max_queue_length = windowize(df["average_max_queue_length"].tolist())
	# x = windowize(df[relevant_thing].tolist())

	fig, ax1 = plt.subplots()

	things = []
	things += (ax1.plot(x, average_queue_length, label="queue length"))
	if (np.array(average_max_queue_length) > -1).all():
		things += (ax1.plot(x, average_max_queue_length, label="max queue length"))

	ax2 = ax1.twinx()
	things += (ax2.plot(x, average_throughput, label="throughput", color="red"))

	appropriate_dir = "/".join(path.split("/")[:-1])+"/plots/"
	os.makedirs(appropriate_dir, exist_ok=True)

	file_name = ".".join(path.split("/")[-1].split(".")[:-1])+".pdf"
	# plt.grid(True)
	ax1.set_xlabel(f"{full_names[relevant_thing]} ({units[relevant_thing]})")
	ax1.set_ylabel("queue size (packets)")
	ax2.set_ylabel("throughput (Mbit/s)")
	ax1.set_ylim(ymin=0)
	ax2.set_ylim(ymin=0)
	# ax1.legend(loc=0)
	# ax2.legend(loc=1)
	# print("things", things)
	plt.legend(things, [l.get_label() for l in things], loc="lower right")
	# plt.title(cc_name)
	plt.tight_layout()
	plt.savefig(appropriate_dir+file_name, bbox_inches = 'tight', pad_inches = 0)
	plt.close()
