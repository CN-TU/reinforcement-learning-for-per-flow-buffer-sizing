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
import ns.network
import ns.point_to_point
import ns.internet
import ns.applications
import ns.traffic_control
import ns.internet_apps
import os

# import pdb; pdb.set_trace()

directory = "MixTrafficRL/"
numSenders = 1
# Hopefully, only one sender enables a simpler architecture and higher performance.
assert numSenders==1
stopTime = 10
queueDiscType = "FifoQueueDisc"
queueDisc = "ns3::"+queueDiscType
bottleneckBandwidth = "20Mbps"
delay = 10
# actual_delay = int(round(delay/2))
actual_delay = delay/2
print("actual_delay*2", actual_delay*2)
bottleneckDelay = f"{actual_delay}ms"
# accessBandwidth = "1000Mbps"
# accessDelay = "1ms"

def CheckQueueSize (queue, queue_disc_type):
	qSize = queue.GetCurrentSize().GetValue()
	# check queue size every 1/10 of a second
	ns.core.Simulator.Schedule (ns.core.Seconds (0.1), CheckQueueSize, queue, queue_disc_type)

	with open((directory + queue_disc_type + "/queueTraces/queue.plotme"), "a") as fPlotQueue:
		fPlotQueue.write(f"{ns.core.Simulator.Now ().GetSeconds ()} {qSize}\n")

# def CwndTrace (stream, oldCwnd, newCwnd):
# 	stream.write(f"{Simulator.Now ().GetSeconds ()} {newCwnd / 1446.0}")

# def TraceCwnd (queue_disc_type):
# 	for i in range(numSenders):
# 		asciiTraceHelper = ns.network.AsciiTraceHelper()
# 		stream = asciiTraceHelper.CreateFileStream (directory+queue_disc_type + "/cwndTraces/S1-" + str(i + 1) + ".plotme")
# 		ns.core.Config.ConnectWithoutContext ("/NodeList/" + str (i) + "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", ns.core.MakeBoundCallback(CwndTrace,stream))

def main (argv):

	# Sender
	tcpSender = ns.network.NodeContainer ()
	tcpSender.Create (numSenders)

	# # Gateway
	# gateway = ns.network.NodeContainer ()
	# gateway.Create (1)

	# Sink
	sink = ns.network.NodeContainer ()
	sink.Create (1)

	# TODO: Set default section

	ns.core.Config.SetDefault ("ns3::TcpSocket::SndBufSize", ns.core.UintegerValue (1 << 20))
	ns.core.Config.SetDefault ("ns3::TcpSocket::RcvBufSize", ns.core.UintegerValue (1 << 20))
	ns.core.Config.SetDefault ("ns3::TcpSocket::DelAckTimeout", ns.core.TimeValue (ns.core.Seconds (0)))
	ns.core.Config.SetDefault ("ns3::TcpSocket::InitialCwnd", ns.core.UintegerValue (1))
	ns.core.Config.SetDefault ("ns3::TcpSocketBase::LimitedTransmit", ns.core.BooleanValue (False))
	ns.core.Config.SetDefault ("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue (1446))
	ns.core.Config.SetDefault ("ns3::TcpSocketBase::WindowScaling", ns.core.BooleanValue (True))
	ns.core.Config.SetDefault (queueDisc + "::MaxSize", ns.network.QueueSizeValue (ns.network.QueueSize ("100p")))
	ns.core.Config.SetDefault ("ns3::TcpL4Protocol::SocketType", ns.core.StringValue ("ns3::TcpBic"))

	internet = ns.internet.InternetStackHelper()
	internet.InstallAll ()

	# tchPfifo = ns.traffic_control.TrafficControlHelper()
	# handle = tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc")
	# tchPfifo.AddInternalQueues (handle, 3, "ns3::DropTailQueue", "MaxSize", ns.core.StringValue("1000p"))

	tch = ns.traffic_control.TrafficControlHelper()
	tch.SetRootQueueDisc (queueDisc)

	# accessLink = ns.point_to_point.PointToPointHelper()
	# accessLink.SetDeviceAttribute ("DataRate", ns.core.StringValue(accessBandwidth))
	# accessLink.SetChannelAttribute ("Delay", ns.core.StringValue(accessDelay))

	bottleneckLink = ns.point_to_point.PointToPointHelper()
	bottleneckLink.SetDeviceAttribute ("DataRate", ns.core.StringValue(bottleneckBandwidth))
	bottleneckLink.SetChannelAttribute ("Delay", ns.core.StringValue(bottleneckDelay))

	# devices_list = [None] * numSenders
	# for i in range(numSenders):
	# 	devices_list[i] = accessLink.Install(tcpSender.Get(i), gateway.Get(0))
	# 	tchPfifo.Install(devices_list[i])
	# devices = [ns.network.NetDeviceContainer(item) for item in devices_list]

	devices_list = [None] * numSenders
	queue_discs_list = []
	for i in range(numSenders):
		devices_list[i] = bottleneckLink.Install(tcpSender.Get(i), sink.Get (0))
		queue_discs_list.append(tch.Install(devices_list[i]))
	devices = [ns.network.NetDeviceContainer(item) for item in devices_list]

	# devices_sink = ns.network.NetDeviceContainer()
	# devices_sink = accessLink.Install (gateway.Get (1), sink.Get (0))
	# tchPfifo.Install (devices_sink)

	# devices_gateway = ns.network.NetDeviceContainer()
	# devices_gateway = bottleneckLink.Install (gateway.Get (0), gateway.Get (1))
	# # Install QueueDisc at gateway
	# queueDiscs = tch.Install (devices_gateway)

	address = ns.internet.Ipv4AddressHelper()
	address.SetBase (ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.0"))

	interfaces_list = [None]*numSenders
	# interfaces_sink = ns.internet.Ipv4InterfaceContainer()
	# interfaces_gateway = ns.internet.Ipv4InterfaceContainer()

	for i in range(numSenders):
		address.NewNetwork ()
		interfaces_list [i] = address.Assign (devices[i])
	interfaces = [ns.internet.Ipv4InterfaceContainer(item) for item in interfaces_list]

	# address.NewNetwork ()
	# interfaces_gateway = address.Assign (devices_gateway)

	# address.NewNetwork ()
	# interfaces_sink = address.Assign (devices_sink)

	ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables ()

	port = 50000
	sinkLocalAddress =  ns.network.Address(ns.network.InetSocketAddress (ns.network.Ipv4Address.GetAny (), port))
	sinkHelper = ns.applications.PacketSinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress)

	# print("alive")
	remoteAddress = ns.network.AddressValue(ns.network.InetSocketAddress (interfaces[0].GetAddress (1), port))

	# ns.core.Config.SetDefault ("ns3::V4Ping::Verbose", ns.core.BooleanValue (True))
	# ping = ns.internet_apps.V4PingHelper (interfaces[0].GetAddress (1))
	# pingers = ns.network.NodeContainer()
	# pingers.Add (tcpSender)
	# apps = ping.Install(pingers)
	# apps.Start(ns.core.Seconds(0))
	# apps.Stop(ns.core.Seconds(10))

	ftp = ns.applications.BulkSendHelper("ns3::TcpSocketFactory", ns.network.Address ())
	ftp.SetAttribute ("Remote", remoteAddress)
	ftp.SetAttribute ("SendSize", ns.core.UintegerValue(1000))

	sourceApp = ftp.Install (tcpSender)
	sourceApp.Start (ns.core.Seconds (0))
	sourceApp.Stop (ns.core.Seconds (stopTime))

	sinkHelper.SetAttribute ("Protocol", ns.core.TypeIdValue(ns.internet.TcpSocketFactory.GetTypeId ()))
	sinkApp = sinkHelper.Install (sink)
	sinkApp.Start (ns.core.Seconds (0))
	sinkApp.Stop (ns.core.Seconds (stopTime))

	queue = queue_discs_list[0].Get (0)

	# os.makedirs(directory+queueDiscType+"/cwndTraces", exist_ok=True)
	os.makedirs(directory+queueDiscType+"/queueTraces", exist_ok=True)

	ns.core.Simulator.ScheduleNow (CheckQueueSize, queue,queueDiscType)

	# ns.core.Simulator.Schedule (ns.core.Seconds (0.1), TraceCwnd,queueDiscType)

	ns.core.Simulator.Stop (ns.core.Seconds (stopTime))
	ns.core.Simulator.Run ()
	ns.core.Simulator.Destroy ()

if __name__ == '__main__':
		import sys
		main (sys.argv)
