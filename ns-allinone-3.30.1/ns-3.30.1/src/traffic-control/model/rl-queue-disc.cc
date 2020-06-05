/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
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
 * Authors: Pasquale Imputato <p.imputato@gmail.com>
 *          Stefano Avallone <stefano.avallone@unina.it>
*/

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/queue.h"
#include "rl-queue-disc.h"
#include "rl-internal-queue-disc.h"
#include "codel-queue-disc.h"
#include "ns3/net-device-queue-interface.h"
#include <cassert>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RLQueueDisc");

NS_OBJECT_ENSURE_REGISTERED (RLFlow);

TypeId RLFlow::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RLFlow")
    .SetParent<QueueDiscClass> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<RLFlow> ()
  ;
  return tid;
}

RLFlow::RLFlow ()
  : m_deficit (0),
    m_status (INACTIVE)
{
  NS_LOG_FUNCTION (this);
}

RLFlow::~RLFlow ()
{
  NS_LOG_FUNCTION (this);
}

void
RLFlow::SetDeficit (uint32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit = deficit;
}

int32_t
RLFlow::GetDeficit (void) const
{
  NS_LOG_FUNCTION (this);
  return m_deficit;
}

void
RLFlow::IncreaseDeficit (int32_t deficit)
{
  NS_LOG_FUNCTION (this << deficit);
  m_deficit += deficit;
}

void
RLFlow::SetStatus (FlowStatus status)
{
  NS_LOG_FUNCTION (this);
  m_status = status;
}

RLFlow::FlowStatus
RLFlow::GetStatus (void) const
{
  NS_LOG_FUNCTION (this);
  return m_status;
}


NS_OBJECT_ENSURE_REGISTERED (RLQueueDisc);

TypeId RLQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RLQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<RLQueueDisc> ()
    .AddAttribute ("MaxSize",
                   "The maximum number of packets accepted by this queue disc",
                   QueueSizeValue (QueueSize ("10240p")),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("Flows",
                   "The number of queues into which the incoming packets are classified",
                   UintegerValue (1024),
                   MakeUintegerAccessor (&RLQueueDisc::m_flows),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("DropBatchSize",
                   "The maximum number of packets dropped from the fat flow",
                   UintegerValue (64),
                   MakeUintegerAccessor (&RLQueueDisc::m_dropBatchSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Perturbation",
                   "The salt used as an additional input to the hash function used to classify packets",
                   UintegerValue (0),
                   MakeUintegerAccessor (&RLQueueDisc::m_perturbation),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

RLQueueDisc::RLQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::MULTIPLE_QUEUES, QueueSizeUnit::PACKETS),
    m_quantum (0)
{
  NS_LOG_FUNCTION (this);
}

RLQueueDisc::~RLQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

void
RLQueueDisc::SetQuantum (uint32_t quantum)
{
  NS_LOG_FUNCTION (this << quantum);
  m_quantum = quantum;
}

uint32_t
RLQueueDisc::GetQuantum (void) const
{
  return m_quantum;
}

void RLQueueDisc::SetRlNet (std::shared_ptr<ns3::RlNet> rl_net) {
  m_rl_net = rl_net;
}
void RLQueueDisc::SetBw(double bw) {
  m_bw = bw;
}
void RLQueueDisc::SetDelay(double delay) {
  m_delay = delay;
}

std::shared_ptr<ns3::RlNet> RLQueueDisc::GetRlNet (void) const {
  return m_rl_net;
}

bool
RLQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  uint32_t h = 0;

  if (GetNPacketFilters () == 0)
    {
      h = item->Hash (m_perturbation) % m_flows;
    }
  else
    {
      int32_t ret = Classify (item);

      if (ret != PacketFilter::PF_NO_MATCH)
        {
          h = ret % m_flows;
        }
      else
        {
          NS_LOG_ERROR ("No filter has been able to classify this packet, drop it.");
          DropBeforeEnqueue (item, UNCLASSIFIED_DROP);
          return false;
        }
    }

  Ptr<RLFlow> flow;
  if (m_flowsIndices.find (h) == m_flowsIndices.end ())
    {
      flow = m_flowFactory.Create<RLFlow> ();
      Ptr<QueueDisc> qd = m_queueDiscFactory.Create<QueueDisc> ();
      qd->Initialize ();
      flow->SetQueueDisc (qd);
      AddQueueDiscClass (flow);

      Ptr<RlInternalQueueDisc> actual_qdisc = DynamicCast<RlInternalQueueDisc> (qd);
      actual_qdisc->SetRlNet(m_rl_net);
      actual_qdisc->SetBw(m_bw);
      actual_qdisc->SetDelay(m_delay);
      actual_qdisc->m_time_for_action = m_time_for_action;
      actual_qdisc->m_action_to_perform = m_action_to_perform;
      // std::cout << "Set time for action to " << m_time_for_action << std::endl;

      m_flowsIndices[h] = GetNQueueDiscClasses () - 1;
      NS_LOG_DEBUG ("Creating a new flow queue with index " << h << " with number " << m_flowsIndices[h]);
      // std::cout << "Creating a new flow queue with index " << h << " with number " << m_flowsIndices[h] << std::endl;
      assert(m_flowsIndices[h] < 1);
    }
  else
    {
      flow = StaticCast<RLFlow> (GetQueueDiscClass (m_flowsIndices[h]));
    }

  if (flow->GetStatus () == RLFlow::INACTIVE)
    {
      flow->SetStatus (RLFlow::NEW_FLOW);
      flow->SetDeficit (m_quantum);
      m_newFlows.push_back (flow);
    }

  flow->GetQueueDisc ()->Enqueue (item);

  NS_LOG_DEBUG ("Packet enqueued into flow " << h << "; flow index " << m_flowsIndices[h]);

  // if (GetCurrentSize () > GetMaxSize ())
  //   {
  //     RLDrop ();
  //   }

  return true;
}

Ptr<QueueDiscItem>
RLQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<RLFlow> flow;
  Ptr<QueueDiscItem> item;

  do
    {
      bool found = false;

      while (!found && !m_newFlows.empty ())
        {
          flow = m_newFlows.front ();

          if (flow->GetDeficit () <= 0)
            {
              flow->IncreaseDeficit (m_quantum);
              flow->SetStatus (RLFlow::OLD_FLOW);
              m_oldFlows.push_back (flow);
              m_newFlows.pop_front ();
            }
          else
            {
              NS_LOG_DEBUG ("Found a new flow with positive deficit");
              found = true;
            }
        }

      while (!found && !m_oldFlows.empty ())
        {
          flow = m_oldFlows.front ();

          if (flow->GetDeficit () <= 0)
            {
              flow->IncreaseDeficit (m_quantum);
              m_oldFlows.push_back (flow);
              m_oldFlows.pop_front ();
            }
          else
            {
              NS_LOG_DEBUG ("Found an old flow with positive deficit");
              found = true;
            }
        }

      if (!found)
        {
          NS_LOG_DEBUG ("No flow found to dequeue a packet");
          return 0;
        }

      item = flow->GetQueueDisc ()->Dequeue ();

      if (!item)
        {
          NS_LOG_DEBUG ("Could not get a packet from the selected flow queue");
          if (!m_newFlows.empty ())
            {
              flow->SetStatus (RLFlow::OLD_FLOW);
              m_oldFlows.push_back (flow);
              m_newFlows.pop_front ();
            }
          else
            {
              flow->SetStatus (RLFlow::INACTIVE);
              m_oldFlows.pop_front ();
            }
        }
      else
        {
          NS_LOG_DEBUG ("Dequeued packet " << item->GetPacket ());
        }
    } while (item == 0);

  flow->IncreaseDeficit (item->GetSize () * -1);

  return item;
}

bool
RLQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("RLQueueDisc cannot have classes");
      return false;
    }

  if (GetNInternalQueues () > 0)
    {
      NS_LOG_ERROR ("RLQueueDisc cannot have internal queues");
      return false;
    }

  // we are at initialization time. If the user has not set a quantum value,
  // set the quantum to the MTU of the device (if any)
  if (!m_quantum)
    {
      Ptr<NetDeviceQueueInterface> ndqi = GetNetDeviceQueueInterface ();
      Ptr<NetDevice> dev;
      // if the NetDeviceQueueInterface object is aggregated to a
      // NetDevice, get the MTU of such NetDevice
      if (ndqi && (dev = ndqi->GetObject<NetDevice> ()))
        {
          m_quantum = dev->GetMtu ();
          NS_LOG_DEBUG ("Setting the quantum to the MTU of the device: " << m_quantum);
        }

      if (!m_quantum)
        {
          NS_LOG_ERROR ("The quantum parameter cannot be null");
          return false;
        }
    }

  return true;
}

void
RLQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);

  m_flowFactory.SetTypeId ("ns3::RLFlow");

  m_queueDiscFactory.SetTypeId ("ns3::RlInternalQueueDisc");
  // m_queueDiscFactory.Set ("MaxSize", QueueSizeValue (GetMaxSize ()));
}

uint32_t
RLQueueDisc::RLDrop (void)
{
  NS_LOG_FUNCTION (this);

  uint32_t maxBacklog = 0, index = 0;
  Ptr<QueueDisc> qd;

  /* Queue is full! Find the fat flow and drop packet(s) from it */
  for (uint32_t i = 0; i < GetNQueueDiscClasses (); i++)
    {
      qd = GetQueueDiscClass (i)->GetQueueDisc ();
      uint32_t bytes = qd->GetNBytes ();
      if (bytes > maxBacklog)
        {
          maxBacklog = bytes;
          index = i;
        }
    }

  /* Our goal is to drop half of this fat flow backlog */
  uint32_t len = 0, count = 0, threshold = maxBacklog >> 1;
  qd = GetQueueDiscClass (index)->GetQueueDisc ();
  Ptr<QueueDiscItem> item;

  do
    {
      item = qd->GetInternalQueue (0)->Dequeue ();
      DropAfterDequeue (item, OVERLIMIT_DROP);
      len += item->GetSize ();
    } while (++count < m_dropBatchSize && len < threshold);

  return index;
}

double RLQueueDisc::GetSummedQueueLength() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  return actual_qd->summedQueueLength;
}
double RLQueueDisc::GetSummedMaxQueueLength() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  return actual_qd->summedMaxQueueLength;
}
double RLQueueDisc::GetSentBytes() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  return actual_qd->sentBytes;
}
double RLQueueDisc::GetSummedQueueLengthUntilDecision() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->summedQueueLengthUntilDecision;
}
double RLQueueDisc::GetSentBytesUntilDecision() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->sentBytesUntilDecision;
}

int8_t RLQueueDisc::GetActualDecision() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->decision;
}

uint32_t RLQueueDisc::GetAlreadyInspected() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  return actual_qd->m_already_inspected;
}

torch::Tensor RLQueueDisc::GetActualInput() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->m_actual_input;
}

torch::Tensor RLQueueDisc::GetPrediction() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->prediction;
}

double RLQueueDisc::GetActualTime() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->m_time_for_actual_action;
}

int RLQueueDisc::GetChosenQueueSize() const {
  Ptr<RLFlow> flow = StaticCast<RLFlow> (GetQueueDiscClass (0));
  Ptr<RlInternalQueueDisc> actual_qd = StaticCast<RlInternalQueueDisc> (flow->GetQueueDisc());
  // assert (actual_qd->decided);
  return actual_qd->m_chosen_queue_size;
}


} // namespace ns3
