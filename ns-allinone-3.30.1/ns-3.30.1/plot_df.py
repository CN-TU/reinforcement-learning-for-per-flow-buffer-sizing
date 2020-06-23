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

common_path_regex = re.compile("(^.*)\_cc")

paths = sys.argv[1:]

cc_mapping = {0: "New Reno", 1: "Bic"}
full_names = {"bw": "bandwidth", "delay": "delay"}
units = {"bw": "Mbit/s", "delay": "ms"}

aggregate_results = {}

for path in paths:
	print("path", path)
	if os.path.isdir(path):
		continue
	df = pd.read_csv(path, sep=";", header=0)

	relevant_thing = "bw" if "bw" in path else "delay"

	average_queue_length = np.array(df["average_queue_length"].tolist())
	max_queue_length = np.array(df["max_measured_queue_length"].tolist())
	average_throughput = np.array(df["average_throughput"].tolist())/1000000*8
	average_max_queue_length = np.array(df["average_max_queue_length"].tolist())
	x = df[relevant_thing].tolist()

	match_for_path = common_path_regex.search(path).group(1)
	if match_for_path not in aggregate_results:
		aggregate_results[match_for_path] = {}
	if "throughput" not in aggregate_results[match_for_path]:
		aggregate_results[match_for_path]["throughput"] = []
	if "queue" not in aggregate_results[match_for_path]:
		aggregate_results[match_for_path]["queue"] = []
	if "max_queue" not in aggregate_results[match_for_path]:
		aggregate_results[match_for_path]["max_queue"] = []
	aggregate_results[match_for_path]["throughput"].append(average_throughput)
	aggregate_results[match_for_path]["queue"].append(average_queue_length)
	aggregate_results[match_for_path]["max_queue"].append(max_queue_length)

	match = cc_int.search(path)
	cc_string = match.group(1)
	cc_name = cc_mapping[int(match.group(2))]
	print(cc_name, relevant_thing)
	path = path.replace(cc_string, f"{cc_name}_")
	path = path.replace("bw", full_names["bw"])
	path = path.replace("delay", full_names["delay"])

	if "RLQueueDisc" in path:
		print("correlation max", np.corrcoef(x, average_max_queue_length)[0,1])
		print("avg max", statistics.mean(average_max_queue_length))
	print("throughput avg", np.mean(average_throughput))
	print("queue avg", np.mean(average_queue_length))
	print("max queue avg", np.mean(max_queue_length))

	if "max_queue_avg_for_cc" not in aggregate_results[match_for_path]:
		aggregate_results[match_for_path]["max_queue_avg_for_cc"] = {}
	if cc_name not in aggregate_results[match_for_path]["max_queue_avg_for_cc"]:
		aggregate_results[match_for_path]["max_queue_avg_for_cc"][cc_name] = []

	aggregate_results[match_for_path]["max_queue_avg_for_cc"][cc_name].append(average_max_queue_length)

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
	ax1.set_xlabel(f"{full_names[relevant_thing]} ({units[relevant_thing]})")
	ax1.set_ylabel("queue size (packets)")
	ax2.set_ylabel("throughput (Mbit/s)")
	ax1.set_ylim(ymin=0)
	ax2.set_ylim(ymin=0)
	plt.legend(things, [l.get_label() for l in things], loc="lower right")
	plt.tight_layout()
	plt.savefig(appropriate_dir+file_name, bbox_inches = 'tight', pad_inches = 0)
	plt.close()

for key in aggregate_results:
	print("key", key)
	print("avg_throughput", np.mean(np.concatenate(aggregate_results[key]["throughput"])), "avg_queue", np.mean(np.concatenate(aggregate_results[key]["queue"])), "max_queue", np.mean(np.concatenate(aggregate_results[key]["max_queue"])))
	print([(item, np.mean(aggregate_results[match_for_path]["max_queue_avg_for_cc"][item])) for item in aggregate_results[match_for_path]["max_queue_avg_for_cc"]])