# -*-  Mode: Python; -*-
# /*
#  * Copyright (c) 2016 NITK Surathkal
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License version 2 as
#  * published by the Free Software Foundation;
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#  *
#  * Ported to Python by: Mohit P. Tahiliani <tahiliani@nitk.edu.in>
#  */

# Network topology
#
#        n0 ----------- n1
#             500 Kbps
#              5 ms
#
#  - Flow from n0 to n1 using BulkSendApplication.
#  - Tracing of queues and packet receptions to file "tcp-bulk-send-py.tr"
#    and pcap tracing available when tracing is turned on.

import ns.core
import ns.point_to_point
import ns.internet
import ns.applications
import ns.network

def main (argv):
	#
	# Allow the user to override any of the defaults at
	# run-time, via command-line arguments
	#
	cmd = ns.core.CommandLine ()
	cmd.tracing = "False"
	cmd.maxBytes = 0
	cmd.AddValue ("tracing", "Flag to enable/disable tracing")
	cmd.AddValue ("maxBytes", "Total number of bytes for application to send")
	cmd.Parse (sys.argv)

	tracing = cmd.tracing
	maxBytes = int (cmd.maxBytes)

	#
	# Explicitly create the nodes required by the topology (shown above).
	#
	("Create nodes.")
	nodes = ns.network.NodeContainer ()
	nodes.Create (2)

	#
	# Explicitly create the point-to-point link required by the topology (shown above).
	#
	print("Create channels.")
	pointToPoint = ns.point_to_point.PointToPointHelper ()
	pointToPoint.SetDeviceAttribute ("DataRate", ns.core.StringValue ("500Kbps"))
	pointToPoint.SetChannelAttribute ("Delay", ns.core.StringValue ("5ms"))

	devices = pointToPoint.Install (nodes)

	#
	# Install the internet stack on the nodes
	#
	stack = ns.internet.InternetStackHelper ()
	stack.Install (nodes)

	#
	# We've got the "hardware" in place.  Now we need to add IP addresses.
	#
	print("Assign IP Addresses.")
	address = ns.internet.Ipv4AddressHelper ()
	address.SetBase (ns.network.Ipv4Address ("10.1.1.0"), ns.network.Ipv4Mask ("255.255.255.0"))
	i = address.Assign (devices)

	print("Create Applications.")
	#
	# Create a BulkSendApplication and install it on node 0
	#
	port = 9  # well-known echo port number

	source = ns.applications.BulkSendHelper ("ns3::TcpSocketFactory", ns.network.InetSocketAddress (i.GetAddress (1), port))

	# Set the amount of data to send in bytes.  Zero is unlimited.
	source.SetAttribute ("MaxBytes", ns.core.UintegerValue (maxBytes))
	sourceApps = source.Install (nodes.Get (0))
	sourceApps.Start (ns.core.Seconds (0.0))
	sourceApps.Stop (ns.core.Seconds (10.0))

	#
	# Create a PacketSinkApplication and install it on node 1
	#
	sink = ns.applications.PacketSinkHelper ("ns3::TcpSocketFactory", ns.network.InetSocketAddress (ns.network.Ipv4Address.GetAny (), port))
	sinkApps = sink.Install (nodes.Get (1))
	sinkApps.Start (ns.core.Seconds (0.0))
	sinkApps.Stop (ns.core.Seconds (10.0))

	#
	# Set up tracing if enabled
	#
	if tracing == "True":
		ascii = ns.network.AsciiTraceHelper ()
		pointToPoint.EnableAsciiAll (ascii.CreateFileStream ("tcp-bulk-send-py.tr"))
		pointToPoint.EnablePcapAll ("tcp-bulk-send-py", False)

	#
	# Now, do the actual simulation.
	#
	print("Run Simulation.")
	ns.core.Simulator.Stop (ns.core.Seconds (10.0))
	ns.core.Simulator.Run ()
	ns.core.Simulator.Destroy ()
	print("Done.")

	sink1 = ns.applications.PacketSink (sinkApps.Get (0))
	print("Total Bytes Received:", sink1.GetTotalRx ())

if __name__ == '__main__':
    import sys
    main (sys.argv)
