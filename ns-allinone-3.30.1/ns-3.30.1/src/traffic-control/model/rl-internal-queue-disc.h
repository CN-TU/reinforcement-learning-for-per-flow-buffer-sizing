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

#ifndef RL_INTERNAL_QUEUE_DISC_H
#define RL_INTERNAL_QUEUE_DISC_H

#define EXPERIMENT_DELTA 2

#include "ns3/queue-disc.h"
#include "ns3/rl-net.h"

namespace ns3 {

/**
 * \ingroup traffic-control
 *
 * Simple queue disc implementing the RL_INTERNAL (First-In First-Out) policy.
 *
 */
class RlInternalQueueDisc : public QueueDisc {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief RlInternalQueueDisc constructor
   *
   * Creates a queue with a depth of 1000 packets by default
   */
  RlInternalQueueDisc ();

  virtual ~RlInternalQueueDisc();

  // Reasons for dropping packets
  static constexpr const char* LIMIT_EXCEEDED_DROP = "Queue disc limit exceeded";  //!< Packet dropped due to queue disc limit exceeded

  void SetRlNet (std::shared_ptr<ns3::RlNet> rl_net);
  void SetBw(double bw);
  void SetDelay(double delay);
  std::shared_ptr<ns3::RlNet> GetRlNet (void) const;
  FeatureVector fv;
  double lastDepartureTime = 0.0;
  double lastArrivalTime = 0.0;
  std::shared_ptr<ns3::RlNet> m_rl_net;
  double m_bw;
  double m_delay;
  double summedQueueLength = 0.0;
  double summedMaxQueueLength = 0.0;
  double sentBytes = 0.0;
  double summedQueueLengthUntilDecision = 0.0;
  double sentBytesUntilDecision = 0.0;
  bool decided = false;
  double m_time_for_action = -1;
  double m_time_for_actual_action = -1;
  torch::Tensor m_actual_input;
  torch::Tensor prediction;
  int8_t decision = -1;
  uint32_t m_every_nth = 1;
  uint64_t m_already_inspected = 0;
  int32_t m_action_to_perform = -1;
  int m_chosen_queue_size = -1;
  // std::vector<torch::Tensor> inputs;
  // std::vector<double> sentBytesUntil;
  // std::vector<double> summedQueueLengthUntil;
  // std::vector<double> times;

private:
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual Ptr<const QueueDiscItem> DoPeek (void);
  virtual bool CheckConfig (void);
  virtual void InitializeParams (void);
  void do_logic(bool should_drop);
  void do_regular_logic();
};

} // namespace ns3

#endif /* RL_INTERNAL_QUEUE_DISC_H */
