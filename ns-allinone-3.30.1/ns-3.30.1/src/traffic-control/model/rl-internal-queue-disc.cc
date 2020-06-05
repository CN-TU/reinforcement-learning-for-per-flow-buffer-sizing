/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
 * Authors:  Stefano Avallone <stavallo@unina.it>
 */

#include <cassert>
#include "ns3/core-module.h"
#include "ns3/log.h"
#include "rl-internal-queue-disc.h"
#include "ns3/object-factory.h"
#include "ns3/drop-tail-queue.h"
#include <algorithm>
#include <string>
// #include "torch/torch.h"

#define MAX_QUEUE_SIZE 1000
#define MIN_QUEUE_SIZE 1

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RlInternalQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (RlInternalQueueDisc);

TypeId RlInternalQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RlInternalQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<RlInternalQueueDisc> ()
    .AddAttribute ("MaxSize",
                   "The max queue size",
                   QueueSizeValue (QueueSize ("1p")),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("EveryNth",
                   "Apply logic on every nth packet",
                   UintegerValue (1),
                   MakeUintegerAccessor (&RlInternalQueueDisc::m_every_nth),
                   MakeUintegerChecker<uint32_t>(1))
  ;
  return tid;
}

RlInternalQueueDisc::RlInternalQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE)
{
  NS_LOG_FUNCTION (this);
  // fv.updateQueue(GetMaxSize().GetValue());
}

RlInternalQueueDisc::~RlInternalQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void RlInternalQueueDisc::do_logic(bool should_drop) {

  int lower = GetMaxSize().GetValue() - EXPERIMENT_DELTA/2;
  int upper = GetMaxSize().GetValue() + EXPERIMENT_DELTA/2;
  auto too_little = std::max(MIN_QUEUE_SIZE-lower, 0);
  auto too_much = std::max(upper-MAX_QUEUE_SIZE, 0);
  lower = lower + too_little - too_much;
  upper = upper + too_little - too_much;

  assert(upper-lower == EXPERIMENT_DELTA);
  if (should_drop) {
    SetMaxSize(std::to_string(lower)+"p");
  } else {
    SetMaxSize(std::to_string(upper)+"p");
  }
  // std::cout << "max_size " << GetMaxSize().GetValue() << std::endl;
  assert(MIN_QUEUE_SIZE <= GetMaxSize().GetValue() && GetMaxSize().GetValue() <= MAX_QUEUE_SIZE);
  fv.updateQueue(GetMaxSize().GetValue());
  m_chosen_queue_size = GetMaxSize().GetValue();
}

void RlInternalQueueDisc::do_regular_logic() {
  auto current_vector = fv.getCurrentVector();
  // std::cout << "current_vector " << current_vector << std::endl;
  torch::Tensor result = m_rl_net->forward(current_vector);
  double result_double = *(result.data_ptr<double>());
  // std::cout << "RL output " << *result_double << std::endl;
  SetMaxSize(std::to_string(std::min(std::max((int) (std::round(result_double)), MIN_QUEUE_SIZE), MAX_QUEUE_SIZE))+"p");
  fv.updateQueue(GetMaxSize().GetValue());
  // std::cout << "regular" << std::endl;
}

bool
RlInternalQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  double current_time = Simulator::Now().GetSeconds();

  double diff = current_time - lastArrivalTime;
  // Apparently it is possible that packets arrive back-to-back...
  // assert (diff > 0 || sentBytes==0);
  fv.updateNewPacket(diff, GetCurrentSize().GetValue());
  summedQueueLength += GetCurrentSize().GetValue() * (current_time-std::max(lastDepartureTime, lastArrivalTime));

  summedMaxQueueLength += GetMaxSize().GetValue() * (current_time-lastArrivalTime);

  if (m_already_inspected == 0) {
    do_regular_logic();
  }

  // std::cout << "enqueue: " << lastArrivalTime << " diff " << diff << " feature_vector " << fv.getCurrentVector() << std::endl;

  if (!decided && m_already_inspected % m_every_nth == 0) {
    if (m_time_for_action != -1 && current_time > m_time_for_action) {
      decided = true;
      m_actual_input = fv.getCurrentVector();
      prediction = m_rl_net->forward(m_actual_input);
      summedQueueLengthUntilDecision = summedQueueLength;
      sentBytesUntilDecision = sentBytes;
      decision = m_action_to_perform;
      m_time_for_actual_action = current_time;
      do_logic(decision==1);
      // std::cout << "max_size " << GetMaxSize().GetValue() << std::endl;
      // std::cout << "decision" << std::endl;
    } else {
      do_regular_logic();
    }
  }

  m_already_inspected += 1;

  if (GetCurrentSize () + item > GetMaxSize ()) {
    // std::cout << "queue full" <<< std::endl;
    NS_LOG_LOGIC ("Queue full -- dropping pkt");
    DropBeforeEnqueue (item, LIMIT_EXCEEDED_DROP);
    fv.updateNewLoss(1);
    lastArrivalTime = current_time;
    return false;
  }

  fv.updateNewLoss(0);
  lastArrivalTime = current_time;

  bool retval = GetInternalQueue (0)->Enqueue (item);

  // If Queue::Enqueue fails, QueueDisc::DropBeforeEnqueue is called by the
  // internal queue because QueueDisc::AddInternalQueue sets the trace callback

  NS_LOG_LOGIC ("Number packets " << GetInternalQueue (0)->GetNPackets ());
  NS_LOG_LOGIC ("Number bytes " << GetInternalQueue (0)->GetNBytes ());

  return retval;
}

Ptr<QueueDiscItem>
RlInternalQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  uint32_t old_queue_len = GetCurrentSize().GetValue();
  Ptr<QueueDiscItem> item = GetInternalQueue (0)->Dequeue ();

  if (!item)
    {
      NS_LOG_LOGIC ("Queue empty");
      return 0;
    }

  double current_time = Simulator::Now().GetSeconds();
  double diff = current_time - lastDepartureTime;
  // std::cout << "dequeue: current_time " << current_time << " diff " << diff << std::endl;
  // assert (diff > 0 || sentBytes==0);
  summedQueueLength += ((double) old_queue_len) * (current_time-std::max(lastDepartureTime, lastArrivalTime));

  sentBytes += item->GetSize ();

  lastDepartureTime = current_time;

  fv.updateNewDeparture(diff);

  return item;
}

Ptr<const QueueDiscItem>
RlInternalQueueDisc::DoPeek (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<const QueueDiscItem> item = GetInternalQueue (0)->Peek ();

  if (!item)
    {
      NS_LOG_LOGIC ("Queue empty");
      return 0;
    }

  return item;
}

bool
RlInternalQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("RlInternalQueueDisc cannot have classes");
      return false;
    }

  if (GetNPacketFilters () > 0)
    {
      NS_LOG_ERROR ("RlInternalQueueDisc needs no packet filter");
      return false;
    }

  if (GetNInternalQueues () == 0)
    {
      // add a DropTail queue
      AddInternalQueue (CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> >
                          ("MaxSize", QueueSizeValue (GetMaxSize ())));
    }

  if (GetNInternalQueues () != 1)
    {
      NS_LOG_ERROR ("RlInternalQueueDisc needs 1 internal queue");
      return false;
    }

  return true;
}

void
RlInternalQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);
}

void RlInternalQueueDisc::SetRlNet (std::shared_ptr<ns3::RlNet>  rl_net) {
  // std::cout << "internal " << rl_net << std::endl;
  m_rl_net = rl_net;
}

void RlInternalQueueDisc::SetBw(double bw) {
  m_bw = bw;
  fv.setBw(m_bw);
}

void RlInternalQueueDisc::SetDelay(double delay) {
  m_delay = delay;
  fv.setDelay(m_delay);
}

std::shared_ptr<ns3::RlNet>  RlInternalQueueDisc::GetRlNet (void) const {
  return m_rl_net;
}

} // namespace ns3
