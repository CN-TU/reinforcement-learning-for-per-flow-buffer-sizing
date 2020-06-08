/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 NITK Surathkal
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Shefali Gupta <shefaligups11@ogmail.com>
 *         Jendaipou Palmei <jendaipoupalmei@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/ipv6-routing-table-entry.h"
#include "ns3/internet-module.h"
#include "ns3/tcp-header.h"
#include "ns3/traffic-control-module.h"
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include "ns3/rl-net.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <sys/shm.h>
#include <sys/wait.h>
#include <semaphore.h>
#include <random>

const bool ACTOR_CRITIC = false;
double LR = 0.01;
double alpha = 0.01;
const int nproc = 20;

using namespace std;
using namespace ns3;

fstream log_file;

auto net = make_shared<RlNet>();
// auto net_throughput = make_shared<RlNet>();
// auto net_queue = make_shared<RlNet>();
auto net_reward = make_shared<RlNetReward>();

#define CC_LOWER 0
#define CC_UPPER 1


double reward_normalizer = 1000000;
// double queue_normalizer = 10;

uint32_t mtu = 1446;
uint32_t every_nth = 10;
uint64_t iteration = 0;
double queue_sampling_interval = 0.1;
ofstream output;

string dir = "results/";
string file_name;
string date;
int numSenders = 1;
int shm_key;

void
CheckQueueSize (Ptr<QueueDisc> queue, string queue_disc_type, string path)
{
  double qSize = queue->GetNBytes ();
  double qPacket = queue->GetNPackets();
  // check queue size every 1/10 of a second
  Simulator::Schedule (Seconds (queue_sampling_interval), &CheckQueueSize, queue, queue_disc_type, path);

  ofstream fPlotQueue (dir + queue_disc_type + "/queueTraces/"+path+"_queue.plotme", ios::out | ios::app);
  // fPlotQueue << Simulator::Now ().GetSeconds () << " " << qSize << endl;
  fPlotQueue << Simulator::Now ().GetSeconds () << " " << qPacket << " " << qSize << endl;
  fPlotQueue.close ();
}

static void
CwndTrace (Ptr<OutputStreamWrapper> stream, uint32_t oldCwnd, uint32_t newCwnd)
{
  // *stream->GetStream () << Simulator::Now ().GetSeconds () << " " << newCwnd / 1446.0 << endl;
  *stream->GetStream () << Simulator::Now ().GetSeconds () << " " << newCwnd << endl;
}

static void
TraceCwnd (string queue_disc_type, string path)
{
  for (uint8_t i = 0; i < numSenders; i++)
    {
      AsciiTraceHelper asciiTraceHelper;
      Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (dir + queue_disc_type + "/cwndTraces/"+ path +"_S1-" + to_string (i + 1) + ".plotme");
      Config::ConnectWithoutContext ("/NodeList/" + to_string (i) + "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", MakeBoundCallback (&CwndTrace,stream));
    }
}

const int shared_memory_multiplier = ACTOR_CRITIC ? 1 : 2;

int shmid;
struct shmseg *shmp;
const int input_len = FeatureVector::queueVectorLength*2+FeatureVector::queueMaxVectorLength+FeatureVector::arrivalVectorLength+FeatureVector::departureVectorLength+FeatureVector::dropVectorLength;
struct wrapper {
  double inputs[nproc*input_len*shared_memory_multiplier];
  double rewards[nproc*shared_memory_multiplier];
  double avg_throughputs_after[nproc*shared_memory_multiplier];
  double avg_queues_after[nproc*shared_memory_multiplier];
  int64_t decisions[nproc*shared_memory_multiplier];
  int64_t chosen_queue_sizes[nproc*shared_memory_multiplier];
  uint32_t n_updated = 0;
  sem_t results_mutex;
};

int memoryID;
struct wrapper *memory;
int rc;
pid_t pid = -1;

void cleanup() {
  cout << "In cleanup" << endl;
  if (pid != 0) {
    cout << "Actually cleaning up" << endl;
    rc = shmctl(memoryID, IPC_RMID, NULL);
    rc = shmdt(memory);
    sem_destroy(&memory->results_mutex);
  }
}

void experiment (torch::optim::Optimizer* optimizer, torch::optim::Optimizer* optimizer_reward, string queue_disc_type="RLQueueDisc", double stopTime=5)
{
  if (queue_disc_type.compare("RLQueueDisc") == 0) {

    if (iteration == 0) {
      string dirToSave = "mkdir -p " + dir + queue_disc_type;
      if (system ((dirToSave + "/logs/").c_str ()) == -1 || system ((dirToSave + "/weights/").c_str ()) == -1)
        {
          exit (1);
        }

      chrono::system_clock::time_point now = chrono::system_clock::now();
      time_t tt = chrono::system_clock::to_time_t(now);
      tm local_tm = *localtime(&tt);

      date = to_string(local_tm.tm_year+1900)+"-"+to_string(local_tm.tm_mon+1)+"-"+to_string(local_tm.tm_mday)+"-"+to_string(local_tm.tm_hour)+"-"+to_string(local_tm.tm_min)+"-"+to_string(local_tm.tm_sec);
      file_name = dir+queue_disc_type+"/logs/"+date+".log";

      log_file.open(file_name, fstream::out);

      log_file << "average_throughput" << ";" << "average_queue_length" << ";" << "average_throughput_after" << ";" << "average_queue_length_after" << ";" << "decision" << ";" << "reward_double" << ";" << "reward_total" << ";" << "prediction";
      if (ACTOR_CRITIC) {
        // log_file << ";" << "throughput_prediction";
        // log_file << ";" << "queue_prediction";
        log_file << ";" << "reward_prediction";
      }
      log_file << endl;
      log_file.flush();
    }
  }

  // cout << "before forking" << endl;
  uint32_t n_already_forked = 0;
  memory->n_updated = 0;
  // cout << "assigning crashed" << endl;
  pid = -1;
  while (pid != 0 && n_already_forked < nproc-1) {
    pid = fork();
    if (pid < 0) {
      perror("Fork failure");
    }
    n_already_forked += 1;
  }

  string family_status;
  if (pid==0) {
    family_status = "child";
  } else {
    family_status = "parent";
    n_already_forked = 0;
  }
  cout << "n_already_forked " << n_already_forked << endl;

  srand((((uint32_t) iteration) << 16) + n_already_forked);

  // std::random_device rd;  //Will be used to obtain a seed for the random number engine
  // cout << pid << " rand " << rand() << endl;
  std::mt19937 gen(rand()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> bw_dis(5.0, 25.0);
  // std::uniform_real_distribution<> delay_dis(5.0, 20.0);
  std::uniform_real_distribution<> delay_dis(5.0, 25.0);
  std::uniform_real_distribution<> stop_time_end_dis(-stopTime/4, stopTime/4);
  std::uniform_int_distribution<> cc_dis(CC_LOWER, CC_UPPER);
  auto delay = delay_dis(gen);
  auto bw = bw_dis(gen);
  auto stop_time_increment = stop_time_end_dis(gen);
  auto cc_int = cc_dis(gen);
  cout << pid << " bw " << bw << endl;
  cout << pid << " delay " << delay << endl;
  cout << pid << " stop_time_increment " << stop_time_increment << endl;
  cout << pid << " cc_int " << cc_int << endl;

  stopTime += stop_time_increment;

  string congestion_control = cc_int==0 ? "NewReno" : "Bic";

  std::uniform_real_distribution<> action_time_dis(0.0, stopTime/2);
  auto proposed_time_for_action = action_time_dis(gen);

  string queue_disc = string ("ns3::") + queue_disc_type;

  pid_t child_pid;
  string family_status_inner;
  if (!ACTOR_CRITIC) {
    child_pid = fork();
    if (child_pid==0) {
      family_status_inner = "child";
    } else {
      family_status_inner = "parent";
    }
  }

  string bottleneckBandwidth = to_string(bw)+"Mbps";
  double actual_delay = delay/2;
  string bottleneckDelay = to_string(actual_delay)+"ms";

  // Create sender
  NodeContainer tcpSender;
  tcpSender.Create (1);

  // Create sink
  NodeContainer sink;
  sink.Create (1);

  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::DelAckTimeout", TimeValue (Seconds (0)));
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (1));
  Config::SetDefault ("ns3::TcpSocketBase::LimitedTransmit", BooleanValue (false));
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (mtu));
  Config::SetDefault ("ns3::TcpSocketBase::WindowScaling", BooleanValue (true));
  Config::SetDefault ("ns3::FifoQueueDisc::MaxSize", QueueSizeValue (QueueSize ("100p")));
  if (queue_disc_type.compare("RLQueueDisc") != 0) {
    Config::SetDefault (queue_disc + "::MaxSize", QueueSizeValue (QueueSize ("100p")));
  } else {
    Config::SetDefault ("ns3::RlInternalQueueDisc::MaxSize", QueueSizeValue (QueueSize ("1p")));
    Config::SetDefault ("ns3::RlInternalQueueDisc::EveryNth", UintegerValue (every_nth));
  }
  // cout << "congestion_control " << congestion_control << endl;
  Config::SetDefault ("ns3::TcpL4Protocol::SocketType", StringValue ("ns3::Tcp" + congestion_control));

  InternetStackHelper internet;
  internet.InstallAll ();

  TrafficControlHelper tchFifo;
  tchFifo.SetRootQueueDisc ("ns3::FifoQueueDisc");

  TrafficControlHelper tch;
  tch.SetRootQueueDisc (queue_disc);

  PointToPointHelper bottleneckLink;
  bottleneckLink.SetDeviceAttribute ("DataRate", StringValue (bottleneckBandwidth));
  bottleneckLink.SetChannelAttribute ("Delay", StringValue (bottleneckDelay));

  // Configure the senders and sinks net devices
  // and the channels between the senders/sinks and the gateways
  NetDeviceContainer devices;

  devices = bottleneckLink.Install (tcpSender.Get (0), sink.Get (0));

  Ptr<PointToPointChannel> channel1 = DynamicCast<PointToPointChannel>(devices.Get(0)->GetChannel());
  Ptr<PointToPointChannel> channel2 = DynamicCast<PointToPointChannel>(devices.Get(1)->GetChannel());

  cout << n_already_forked << "_" << (uint32_t) (child_pid==0) << "_" << channel1->m_delay.GetSeconds() << "_" << channel2->m_delay.GetSeconds() << endl;

  QueueDiscContainer queueDiscs = tch.Install (devices.Get(0));
  // bottleneckLink.EnablePcapAll("rl");
  tchFifo.Install (devices.Get(1));

  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces;

  address.NewNetwork ();
  interfaces = address.Assign (devices);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);

  AddressValue remoteAddress (InetSocketAddress (interfaces.GetAddress (1), port));

  Ptr<PointToPointNetDevice> actual_channel = DynamicCast<PointToPointNetDevice> (devices.Get(0));
  actual_channel->GetQueue()->SetMaxSize(QueueSize ("1p"));
  // cout << "actual_channel_queue " << actual_channel->GetQueue()->GetMaxSize() << endl;

  Ptr<QueueDisc> queue = queueDiscs.Get (0);

  Ptr<RLQueueDisc> actual_qdisc = nullptr;
  if (queue_disc_type.compare("RLQueueDisc") == 0) {
    actual_qdisc = DynamicCast<RLQueueDisc> (queue);
    actual_qdisc->m_time_for_action = proposed_time_for_action;
    if (!ACTOR_CRITIC) {
      actual_qdisc->m_action_to_perform = child_pid != 0 ? 0 : 1;
    } else {
      std::uniform_int_distribution<> action_dis(0, 1);
      auto action_int = action_dis(gen);

      actual_qdisc->m_action_to_perform = action_int;
    }
    actual_qdisc->SetRlNet(net);
    actual_qdisc->SetBw(bw);
    actual_qdisc->SetDelay(delay);
  }

  BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
  ftp.SetAttribute ("Remote", remoteAddress);
  ftp.SetAttribute ("SendSize", UintegerValue (1000));

  ApplicationContainer sourceApp = ftp.Install (tcpSender);
  sourceApp.Start (Seconds (0));
  sourceApp.Stop (Seconds (stopTime));

  sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
  ApplicationContainer sinkApp = sinkHelper.Install (sink);
  sinkApp.Start (Seconds (0));
  sinkApp.Stop (Seconds (stopTime));



  string logging_path = date+"/"+to_string(iteration)+"/"+to_string(n_already_forked)+"/"+to_string(child_pid==0)+"_"+to_string(getpid());

  // Simulator::ScheduleNow (&CheckQueueSize, queue, queue_disc_type, logging_path);
  // string dirToSave = "mkdir -p " + dir + queue_disc_type;
  // if (system ((dirToSave + "/cwndTraces/"+date+"/"+to_string(iteration)+"/"+to_string(n_already_forked)+"/").c_str ()) == -1
  //     || system ((dirToSave + "/queueTraces/"+date+"/"+to_string(iteration)+"/"+to_string(n_already_forked)+"/").c_str ()) == -1)
  //   {
  //     exit (1);
  //   }
  // Simulator::Schedule (Seconds (queue_sampling_interval), &TraceCwnd, queue_disc_type, logging_path);
  // bottleneckLink.EnablePcapAll("rl_"+to_string(iteration)+"_"+to_string(child_pid==0));



  Simulator::Stop (Seconds (stopTime));
  auto t1 = chrono::high_resolution_clock::now();
  Simulator::Run ();
  auto t2 = chrono::high_resolution_clock::now();
  auto diff = ((double) (chrono::duration_cast<chrono::microseconds>(t2-t1).count()))/1000000;
  cout << n_already_forked << "_" << (uint32_t) (child_pid==0) << " took " << diff << " seconds" << endl;

  Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApp.Get(0));
	cout << "Total Bytes Received: " << sink1->GetTotalRx () << endl;

  // cout << "family_status " << family_status << endl;

  if (queue_disc_type.compare("RLQueueDisc") == 0) {

    cout << "action time " << actual_qdisc->m_time_for_action << endl;
    int8_t decision = actual_qdisc->GetActualDecision();
    if (decision != -1) {
      torch::Tensor input = actual_qdisc->GetActualInput();
      double actual_time_for_action = actual_qdisc->GetActualTime();
      cout << "actual action time " << actual_time_for_action << endl;
      double summed_length = actual_qdisc->GetSummedQueueLength();
      double sent_bytes = actual_qdisc->GetSentBytes();
      double summed_length_until = actual_qdisc->GetSummedQueueLengthUntilDecision();
      double sent_bytes_until = actual_qdisc->GetSentBytesUntilDecision();
      assert (summed_length_until <= summed_length);
      assert (sent_bytes_until <= sent_bytes);
      int chosen_queue_size = actual_qdisc->GetChosenQueueSize();

      double average_queue_length = summed_length/stopTime;
      double average_throughput = sent_bytes/stopTime;
      double average_queue_length_after = (summed_length-summed_length_until)/(stopTime-actual_time_for_action);
      double average_throughput_after = (sent_bytes-sent_bytes_until)/(stopTime-actual_time_for_action);

      double reward_total =  (average_throughput - alpha*average_queue_length*mtu)/reward_normalizer;
      double reward_double = (average_throughput_after - alpha*average_queue_length_after*mtu)/reward_normalizer;

      // assert (decision != -1);
      torch::Tensor prediction = actual_qdisc->GetPrediction().squeeze();
      torch::Tensor throughput_prediction;
      torch::Tensor queue_prediction;
      torch::Tensor reward_prediction;
      if (ACTOR_CRITIC) {
        // throughput_prediction = net_throughput->forward(input).squeeze();
        // queue_prediction = net_queue->forward(input).squeeze();
        reward_prediction = net_reward->forward(input).squeeze();
      }
      uint32_t inspected = actual_qdisc->GetAlreadyInspected();
      double time_for_action = actual_qdisc->m_time_for_action;

      // cout << "pid " << pid << endl;
      cout << "decision " << (int32_t) decision << endl;
      cout << "chosen_queue_size " << chosen_queue_size << endl;
      cout << "iteration " << iteration << endl;
      cout << "sent_bytes " << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << sent_bytes << endl;
      cout << "average_throughput "<< fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_throughput << endl;
      cout << "average_queue_length " << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_queue_length << endl;
      cout << "average_throughput_after " << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_throughput_after << endl;
      cout << "average_queue_length_after " << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_queue_length_after << endl;
      cout << "reward_double " << reward_double << endl;
      cout << "reward_total " << reward_total << endl;
      cout << "input " << input << endl;
      cout << "prediction " << prediction << endl;
      if (ACTOR_CRITIC) {
        // cout << "throughput_prediction " << throughput_prediction << endl;
        // cout << "queue_prediction " << queue_prediction << endl;
        cout << "reward_prediction " << reward_prediction << endl;
      }
      cout << "inspected " << inspected << endl;
      // cout << "time_for_action " << time_for_action << endl;

      log_file << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_throughput << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_queue_length << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_throughput_after << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_queue_length_after << ";" << (int32_t) decision << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << reward_double << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << reward_total << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << prediction.item<double>();
      if (ACTOR_CRITIC) {
        // log_file << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << throughput_prediction.item<double>();
        // log_file << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << queue_prediction.item<double>();
        log_file << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << reward_prediction.item<double>();
      }
      log_file << endl;
      log_file.flush();

      sem_wait(&memory->results_mutex);
        int offset;
        if (!ACTOR_CRITIC) {
          offset = n_already_forked*2+(child_pid==0);
        } else {
          offset = n_already_forked;
        }
        for (size_t i = 0; i < input_len; i++) {
          memory->inputs[offset*input_len+i] = input[i].item<double>();
        }
        memory->rewards[offset] = reward_double;
        // memory->rewards[offset] = average_throughput_after/reward_normalizer;
        memory->avg_throughputs_after[offset] = average_throughput_after;
        memory->avg_queues_after[offset] = average_queue_length_after;
        memory->decisions[offset] = (int64_t) decision;
        memory->chosen_queue_sizes[offset] = chosen_queue_size;
        memory->n_updated += 1;
      sem_post(&memory->results_mutex);
    } else {
      cout << "No decision taken :/" << endl;
    }

    if (!ACTOR_CRITIC) {
      if (child_pid != 0) {
        cout << pid << " waiting for child" << endl;
          wait(NULL);
        cout << pid << " finished waiting for child" << endl;
      } else {
        cout << "I'm a child, exiting " << pid << endl;
        exit(EXIT_SUCCESS);
      }
    }

    if (pid != 0) {
      cout << pid << " waiting" << endl;
      for (size_t i = 0; i < nproc-1; i++) {
        wait(NULL);
      }
      cout << pid << " finished waiting" << endl;
    } else {
      cout << "Terminating " << pid << endl;
      exit(EXIT_SUCCESS);
    }

    assert (pid != 0);
    bool condition = !ACTOR_CRITIC ? memory->n_updated == nproc*2 : memory->n_updated == nproc;
    if (condition) {
    // if (decision != -1) {
      cout << "Got " << memory->n_updated << " results" << endl;
      optimizer->zero_grad();
      optimizer_reward->zero_grad();

      auto options_double = torch::TensorOptions().dtype(torch::kFloat64);
      auto options_long = torch::TensorOptions().dtype(torch::kInt64);



      torch::Tensor output;
      torch::Tensor throughput_output;
      torch::Tensor queue_output;
      torch::Tensor predicted_reward;
      if (!ACTOR_CRITIC) {
        torch::Tensor collated_inputs = torch::from_blob(memory->inputs, {(nproc*2)*input_len}, options_double);
        // cout << "collated_inputs " << collated_inputs << endl;
        auto reshaped_inputs = collated_inputs.reshape({-1, 2, input_len});
        // cout << "reshaped_inputs " << reshaped_inputs << endl;
        auto parents = reshaped_inputs.narrow(1,0,1).squeeze(1);
        auto children = reshaped_inputs.narrow(1,1,1).squeeze(1);
        auto parents_equal_children = (parents == children).all();
        assert(*(parents_equal_children.data_ptr<bool>()));
        cout << "parents " << parents << endl;
        output = net->forward(parents).squeeze();
        cout << "output " << output << endl;
      } else {
        torch::Tensor collated_inputs = torch::from_blob(memory->inputs, {nproc*input_len}, options_double);
        auto reshaped_inputs = collated_inputs.reshape({-1, input_len});
        cout << "reshaped_inputs " << reshaped_inputs << endl;
        output = net->forward(reshaped_inputs).squeeze();
        // throughput_output = net_throughput->forward(reshaped_inputs).squeeze();
        // queue_output = net_queue->forward(reshaped_inputs).squeeze();
        predicted_reward = net_reward->forward(reshaped_inputs).squeeze();

        cout << "output " << output << endl;
        // cout << "throughput_output " << throughput_output << endl;
        // cout << "queue_output " << queue_output << endl;
        // cout << "predicted_reward " << predicted_reward << endl;
      }



      assert (!(*torch::isnan(output).any().data_ptr<bool>()));

      torch::Tensor decisions = torch::from_blob(memory->decisions, {memory->n_updated}, options_long);
      // cout << "decisions " << decisions << endl;


      vector<double> reward_labels;
      vector<double> reward_queue_sizes;

      torch::Tensor reward_tensor = torch::from_blob(memory->rewards, {memory->n_updated}, options_double);
      torch::Tensor queue_size_tensor = torch::from_blob(memory->chosen_queue_sizes, {memory->n_updated}, options_long);
      torch::Tensor original_queue_sizes = queue_size_tensor+(2*decisions-1);

      if (!ACTOR_CRITIC) {
        auto reward_tensor_reshaped = reward_tensor.reshape({-1, 2});
        cout << "reward_tensor_reshaped " << reward_tensor_reshaped << endl;

        auto queue_size_tensor_reshaped = queue_size_tensor.reshape({-1, 2});
        cout << "queue_size_tensor_reshaped " << queue_size_tensor_reshaped << endl;

        for (int32_t i = 0; i < nproc; i++) {
          auto reward_loss = reward_tensor_reshaped.narrow(0, i, 1).squeeze().index(torch::tensor({1}));
          auto reward_normal = reward_tensor_reshaped.narrow(0, i, 1).squeeze().index(torch::tensor({0}));

          bool loss_was_better = *(reward_loss > reward_normal).data_ptr<bool>();
          reward_labels.push_back((double) loss_was_better);
          auto queue_size_loss = *(queue_size_tensor_reshaped.narrow(0, i, 1).squeeze().index(torch::tensor({1})).data_ptr<int64_t>());
          auto queue_size_normal = *(queue_size_tensor_reshaped.narrow(0, i, 1).squeeze().index(torch::tensor({0})).data_ptr<int64_t>());
          auto diff = queue_size_normal-queue_size_loss;
          assert (diff == EXPERIMENT_DELTA);
          auto better_queue_size = loss_was_better ? queue_size_loss : queue_size_normal;
          reward_queue_sizes.push_back((double) better_queue_size);
        }
      } else {
        // cout << "reward_tensor " << reward_tensor << endl;
        cout << "queue_size_tensor " << queue_size_tensor << endl;
        cout << "original_queue_sizes " << original_queue_sizes << endl;
        cout << "reward_tensor " << reward_tensor << endl;

        // torch::Tensor reward_tensor_with_queue = (reward_tensor*reward_normalizer - alpha*queue_size_tensor*((double) mtu))/reward_normalizer;
        // cout << "reward_tensor_with_queue " << reward_tensor_with_queue << endl;
        // torch::Tensor predicted_reward_with_queue = (predicted_reward*reward_normalizer - alpha*original_queue_sizes*((double) mtu))/reward_normalizer;

        // auto was_better_tensor = reward_tensor_with_queue > predicted_reward_with_queue;
        auto was_better_tensor = reward_tensor > predicted_reward;
        cout << "was_better_tensor " << was_better_tensor << endl;
        cout << "decisions " << decisions << endl;

        cout << "predicted_reward " << predicted_reward << endl;
        // cout << "predicted_reward_with_queue " << predicted_reward_with_queue << endl;
        auto loss_was_better_tensor = torch::logical_not(torch::logical_xor(was_better_tensor, decisions));

        for (int32_t i = 0; i < nproc; i++) {

          bool loss_was_better = *(loss_was_better_tensor[i].data_ptr<bool>());
          reward_labels.push_back((double) loss_was_better);

          auto current_action = (bool) (*(decisions[i].data_ptr<int64_t>()));
          auto current_was_better = *(was_better_tensor[i].data_ptr<bool>());


          auto better_queue_size = *(queue_size_tensor[i].data_ptr<int64_t>());
          if (!current_action && !current_was_better) {
            better_queue_size -= EXPERIMENT_DELTA;
          } else if (current_action && !current_was_better) {
            better_queue_size += EXPERIMENT_DELTA;
          }
          reward_queue_sizes.push_back((double) better_queue_size);
        }
      }
      cout << "reward_labels " << reward_labels << endl;
      cout << "reward_queue_sizes " << reward_queue_sizes << endl;
      // cout << "after vector building" << endl;
      auto labels_tensor = torch::tensor(reward_labels);
      auto queue_sizes_labels = torch::tensor(reward_queue_sizes);
      // cout << "survived labels" << endl;
      torch::Tensor loss = torch::l1_loss(output, queue_sizes_labels);
      torch::Tensor loss_reward;

      cout << "loss " << loss << endl;
      assert (!(*torch::isnan(loss).any().data_ptr<bool>()));
      loss.backward();
      optimizer->step();

      if (ACTOR_CRITIC) {
        loss_reward = torch::mse_loss(predicted_reward, reward_tensor);
        cout << "loss_reward " << loss_reward << endl;
        assert (!(*torch::isnan(loss_reward).any().data_ptr<bool>()));
        loss_reward.backward();
        optimizer_reward->step();
      }

      if (iteration%1000 == 0 && iteration > 0) {
        string file_name = dir+queue_disc_type+"/weights/"+date+"_"+to_string(iteration)+".weights";
        torch::save(net, file_name);
        if (ACTOR_CRITIC) {
          string file_name_critic = dir+queue_disc_type+"/weights/"+date+"_critic_"+to_string(iteration)+".weights";
          torch::save(net_reward, file_name_critic);
        }
      }
    } else {
      cout << "No decision" << endl;
    }
  }

  cout << endl;

  Simulator::Destroy ();
}

void evaluate (string file_name, string extracted_path, double bw, double delay, int32_t cc_int, double stopTime=5)
{
  string queue_disc_type = "RLQueueDisc";
  cout << "delay " << delay << endl;
  cout << "bw " << bw << endl;
  cout << "cc_int " << cc_int << endl;

  string congestion_control = cc_int==0 ? "NewReno" : "Bic";
  // string congestion_control = "NewReno";

  auto proposed_time_for_action = stopTime+1;

  string queue_disc = string ("ns3::") + queue_disc_type;

  string bottleneckBandwidth = to_string(bw)+"Mbps";
  double actual_delay = delay/2;
  string bottleneckDelay = to_string(actual_delay)+"ms";

  // Create sender
  NodeContainer tcpSender;
  tcpSender.Create (numSenders);

  // Create sink
  NodeContainer sink;
  sink.Create (1);

  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::DelAckTimeout", TimeValue (Seconds (0)));
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (1));
  Config::SetDefault ("ns3::TcpSocketBase::LimitedTransmit", BooleanValue (false));
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (mtu));
  Config::SetDefault ("ns3::TcpSocketBase::WindowScaling", BooleanValue (true));
  Config::SetDefault ("ns3::FifoQueueDisc::MaxSize", QueueSizeValue (QueueSize ("100p")));
  Config::SetDefault ("ns3::RlInternalQueueDisc::MaxSize", QueueSizeValue (QueueSize ("1p")));
  Config::SetDefault ("ns3::RlInternalQueueDisc::EveryNth", UintegerValue (every_nth));
  // cout << "congestion_control " << congestion_control << endl;
  Config::SetDefault ("ns3::TcpL4Protocol::SocketType", StringValue ("ns3::Tcp" + congestion_control));

  InternetStackHelper internet;
  internet.InstallAll ();

  TrafficControlHelper tchFifo;
  tchFifo.SetRootQueueDisc ("ns3::FifoQueueDisc");

  TrafficControlHelper tch;
  tch.SetRootQueueDisc (queue_disc);

  PointToPointHelper bottleneckLink;
  bottleneckLink.SetDeviceAttribute ("DataRate", StringValue (bottleneckBandwidth));
  bottleneckLink.SetChannelAttribute ("Delay", StringValue (bottleneckDelay));

  // Configure the senders and sinks net devices
  // and the channels between the senders/sinks and the gateways
  NetDeviceContainer devices;

  devices = bottleneckLink.Install (tcpSender.Get (0), sink.Get (0));

  QueueDiscContainer queueDiscs = tch.Install (devices.Get(0));
  // bottleneckLink.EnablePcapAll("rl");
  tchFifo.Install (devices.Get(1));

  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces;

  address.NewNetwork ();
  interfaces = address.Assign (devices);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);

  AddressValue remoteAddress (InetSocketAddress (interfaces.GetAddress (1), port));

  Ptr<PointToPointNetDevice> actual_channel = DynamicCast<PointToPointNetDevice> (devices.Get(0));
  actual_channel->GetQueue()->SetMaxSize(QueueSize ("1p"));
  // cout << "actual_channel_queue " << actual_channel->GetQueue()->GetMaxSize() << endl;

  Ptr<QueueDisc> queue = queueDiscs.Get (0);

  Ptr<RLQueueDisc> actual_qdisc = nullptr;
  if (queue_disc_type.compare("RLQueueDisc") == 0) {
    actual_qdisc = DynamicCast<RLQueueDisc> (queue);
    actual_qdisc->m_time_for_action = proposed_time_for_action;
    actual_qdisc->SetRlNet(net);
    actual_qdisc->SetBw(bw);
    actual_qdisc->SetDelay(delay);
  }

  BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
  ftp.SetAttribute ("Remote", remoteAddress);
  ftp.SetAttribute ("SendSize", UintegerValue (1000));

  ApplicationContainer sourceApp = ftp.Install (tcpSender);
  sourceApp.Start (Seconds (0));
  sourceApp.Stop (Seconds (stopTime));

  sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
  ApplicationContainer sinkApp = sinkHelper.Install (sink);
  sinkApp.Start (Seconds (0));
  sinkApp.Stop (Seconds (stopTime));



  Simulator::ScheduleNow (&CheckQueueSize, queue, queue_disc_type, extracted_path+"/bw_"+to_string(bw)+"_delay_"+to_string(delay));

  string dirToSave = "mkdir -p " + dir + queue_disc_type;
  if (system ((dirToSave + "/cwndTraces/"+extracted_path+"/").c_str ()) == -1
      || system ((dirToSave + "/queueTraces/"+extracted_path+"/").c_str ()) == -1)
    {
      exit (1);
    }

  Simulator::Schedule (Seconds (queue_sampling_interval), &TraceCwnd, queue_disc_type, extracted_path+"/bw_"+to_string(bw)+"_delay_"+to_string(delay));



  Simulator::Stop (Seconds (stopTime));
  Simulator::Run ();

  Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApp.Get(0));
	cout << "Total Bytes Received: " << sink1->GetTotalRx () << endl;

  // cout << "family_status " << family_status << endl;

  double summed_length = actual_qdisc->GetSummedQueueLength();
  double summed_max_length = actual_qdisc->GetSummedMaxQueueLength();
  double sent_bytes = actual_qdisc->GetSentBytes();
  double summed_length_until = actual_qdisc->GetSummedQueueLengthUntilDecision();
  double sent_bytes_until = actual_qdisc->GetSentBytesUntilDecision();
  assert (summed_length_until <= summed_length);
  assert (sent_bytes_until <= sent_bytes);

  double average_queue_length = summed_length/stopTime;
  double average_max_queue_length = summed_max_length/stopTime;
  double average_throughput = sent_bytes/stopTime;

  double reward_total = (average_throughput - alpha*average_queue_length*mtu)/reward_normalizer;

  // assert (decision != -1);
  uint32_t inspected = actual_qdisc->GetAlreadyInspected();

  // cout << "bw " << bw << endl;
  cout << "average_throughput " << average_throughput << endl;
  cout << "average_queue_length " << average_queue_length << endl;
  cout << "average_max_queue_length " << average_max_queue_length << endl;
  // cout << "reward_total " << reward_total << endl;
  // cout << "inspected " << inspected << endl;

  log_file.open(file_name, fstream::app);
  log_file << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << bw << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << delay << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_throughput << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_queue_length << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << average_max_queue_length << ";" << fixed << setprecision(numeric_limits<long double>::digits10 + 1) << reward_total << endl;
  log_file.flush();
  log_file.close();

  cout << endl;

  Simulator::Destroy ();
}

int main (int argc, char **argv)
{
  net->to(torch::kFloat64);
  // net_throughput->to(torch::kFloat64);
  // net_queue->to(torch::kFloat64);
  net_reward->to(torch::kFloat64);

  torch::autograd::variable_list all_params = net->parameters();
  torch::optim::SGD optimizer = torch::optim::SGD(all_params, /*lr=*/LR);

  torch::autograd::variable_list reward_params = net_reward->parameters();
  torch::optim::SGD optimizer_reward = torch::optim::SGD(reward_params, /*lr=*/LR);

  if (argc > 1) {
    string queue_disc_type = "RLQueueDisc";
    size_t steps = 100;
    string path = argv[1];
    torch::load(net, path);

    string extracted_path = path.substr(path.find_last_of("/"), path.length());
    extracted_path = extracted_path.substr(0, path.find_last_of("."));

    for (size_t i = CC_LOWER; i < CC_UPPER+1; i++) {
      string file_name = dir+queue_disc_type+"/logs/"+extracted_path+"_cc_"+to_string(i)+"_delay"+".out";

      double delay_max = 25;
      double delay_min = 5;
      double bw_max = 25;
      double bw_min = 5;

      double bw_mean = (bw_max+bw_min)/2;
      double delay_mean = (delay_max+delay_min)/2;

      log_file.open(file_name, fstream::out);

      log_file << "bw" << ";" << "delay" << ";" << "average_throughput" << ";" << "average_queue_length" << ";" << "average_max_queue_length" << ";" << "reward" << endl;
      log_file.close();

      for (size_t j = 0; j < steps; j++) {
        double delay = delay_min+(delay_max-delay_min)/(steps-1)*j;
        cout << "delay " << delay << endl;
        evaluate(file_name, extracted_path, bw_mean, delay, i);
      }

      file_name = dir+queue_disc_type+"/logs/"+extracted_path+"_cc_"+to_string(i)+"_bw"+".out";

      log_file.open(file_name, fstream::out);

      log_file << "bw" << ";" << "delay" << ";" << "average_throughput" << ";" << "average_queue_length" << ";" << "average_max_queue_length" << ";" << "reward" << endl;
      log_file.close();

      for (size_t j = 0; j < steps; j++) {
        double bw = bw_min+(bw_max-bw_min)/(steps-1)*j;
        cout << "bw " << bw << endl;
        evaluate(file_name, extracted_path, bw, delay_mean, i);
      }
    }
    exit(0);
  }



  cout << "training, ACTOR_CRITIC: " << ACTOR_CRITIC << endl;
  shm_key = rand();
  atexit(cleanup);
  key_t sharedMemoryKey = ftok(".", shm_key);
  if(sharedMemoryKey==-1)
  {
      perror("ftok()");
      exit(1);
  }
  memoryID=shmget(sharedMemoryKey,sizeof(struct wrapper),IPC_CREAT | 0660);
  if(memoryID==-1) {
    perror("shmget()");
    exit(1);
  }
  // cout << "survived shmget" << endl;
  memory = (wrapper*) shmat(memoryID,NULL,0);
  // cout << "survived shmat" << endl;
  if(memory== (void*)-1) {
    perror("shmat()");
    exit(1);
  }
  sem_init(&memory->results_mutex, 1, 1);
  // cout << "survived sem_init" << endl;

  while (true) {
    cout << "Simulation with RL QueueDisc: Start\n" << flush;

    experiment (&optimizer, &optimizer_reward);
    cout << "Simulation with RL QueueDisc: End\n" << flush;
    iteration += 1;
  }
  // cout << "------------------------------------------------\n";
  // cout << "Simulation with CoDel QueueDisc: Start\n";
  // experiment ("CoDelQueueDisc");
  // cout << "Simulation with CoDel QueueDisc: End\n";
  // cout << "Simulation with Fifo QueueDisc: Start\n";
  // experiment ("FifoQueueDisc");
  // cout << "Simulation with Fifo QueueDisc: End\n";
  return 0;
}
