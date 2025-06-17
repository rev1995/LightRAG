---
title Scheduling, Preemption and Eviction
weight 95
content_type concept
no_list true
---

In Kubernetes, scheduling refers to making sure that
are matched to  so that the
 can run them. Preemption
is the process of terminating Pods with lower
so that Pods with higher Priority can schedule on Nodes. Eviction is the process
of terminating one or more Pods on Nodes.

# # Scheduling

* [Kubernetes Scheduler](docsconceptsscheduling-evictionkube-scheduler)
* [Assigning Pods to Nodes](docsconceptsscheduling-evictionassign-pod-node)
* [Pod Overhead](docsconceptsscheduling-evictionpod-overhead)
* [Pod Topology Spread Constraints](docsconceptsscheduling-evictiontopology-spread-constraints)
* [Taints and Tolerations](docsconceptsscheduling-evictiontaint-and-toleration)
* [Scheduling Framework](docsconceptsscheduling-evictionscheduling-framework)
* [Dynamic Resource Allocation](docsconceptsscheduling-evictiondynamic-resource-allocation)
* [Scheduler Performance Tuning](docsconceptsscheduling-evictionscheduler-perf-tuning)
* [Resource Bin Packing for Extended Resources](docsconceptsscheduling-evictionresource-bin-packing)
* [Pod Scheduling Readiness](docsconceptsscheduling-evictionpod-scheduling-readiness)
* [Descheduler](httpsgithub.comkubernetes-sigsdescheduler#descheduler-for-kubernetes)

# # Pod Disruption

* [Pod Priority and Preemption](docsconceptsscheduling-evictionpod-priority-preemption)
* [Node-pressure Eviction](docsconceptsscheduling-evictionnode-pressure-eviction)
* [API-initiated Eviction](docsconceptsscheduling-evictionapi-eviction)
