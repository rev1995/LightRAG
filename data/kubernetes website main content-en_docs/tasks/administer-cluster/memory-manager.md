---
title Utilizing the NUMA-aware Memory Manager

reviewers
- klueska
- derekwaynecarr

content_type task
min-kubernetes-server-version v1.32
weight 410
---

The Kubernetes *Memory Manager* enables the feature of guaranteed memory (and hugepages)
allocation for pods in the `Guaranteed` .

The Memory Manager employs hint generation protocol to yield the most suitable NUMA affinity for a pod.
The Memory Manager feeds the central manager (*Topology Manager*) with these affinity hints.
Based on both the hints and Topology Manager policy, the pod is rejected or admitted to the node.

Moreover, the Memory Manager ensures that the memory which a pod requests
is allocated from a minimum number of NUMA nodes.

The Memory Manager is only pertinent to Linux based hosts.

# #  heading prerequisites

To align memory resources with other requested resources in a Pod spec

- the CPU Manager should be enabled and proper CPU Manager policy should be configured on a Node.
  See [control CPU Management Policies](docstasksadminister-clustercpu-management-policies)
- the Topology Manager should be enabled and proper Topology Manager policy should be configured on a Node.
  See [control Topology Management Policies](docstasksadminister-clustertopology-manager).

Starting from v1.22, the Memory Manager is enabled by default through `MemoryManager`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates).

Preceding v1.22, the `kubelet` must be started with the following flag

`--feature-gatesMemoryManagertrue`

in order to enable the Memory Manager feature.

# # How does the Memory Manager Operate

The Memory Manager currently offers the guaranteed memory (and hugepages) allocation
for Pods in Guaranteed QoS class.
To immediately put the Memory Manager into operation follow the guidelines in the section
[Memory Manager configuration](#memory-manager-configuration), and subsequently,
prepare and deploy a `Guaranteed` pod as illustrated in the section
[Placing a Pod in the Guaranteed QoS class](#placing-a-pod-in-the-guaranteed-qos-class).

The Memory Manager is a Hint Provider, and it provides topology hints for
the Topology Manager which then aligns the requested resources according to these topology hints.
On Linux, it also enforces `cgroups` (i.e. `cpuset.mems`) for pods.
The complete flow diagram concerning pod admission and deployment process is illustrated in
[Memory Manager KEP Design Overview][4] and below

![Memory Manager in the pod admission and deployment process](imagesdocsmemory-manager-diagram.svg)

During this process, the Memory Manager updates its internal counters stored in
[Node Map and Memory Maps][2] to manage guaranteed memory allocation.

The Memory Manager updates the Node Map during the startup and runtime as follows.

# # # Startup

This occurs once a node administrator employs `--reserved-memory` (section
[Reserved memory flag](#reserved-memory-flag)).
In this case, the Node Map becomes updated to reflect this reservation as illustrated in
[Memory Manager KEP Memory Maps at start-up (with examples)][5].

The administrator must provide `--reserved-memory` flag when `Static` policy is configured.

# # # Runtime

Reference [Memory Manager KEP Memory Maps at runtime (with examples)][6] illustrates
how a successful pod deployment affects the Node Map, and it also relates to
how potential Out-of-Memory (OOM) situations are handled further by Kubernetes or operating system.

Important topic in the context of Memory Manager operation is the management of NUMA groups.
Each time pods memory request is in excess of single NUMA node capacity, the Memory Manager
attempts to create a group that comprises several NUMA nodes and features extend memory capacity.
The problem has been solved as elaborated in
[Memory Manager KEP How to enable the guaranteed memory allocation over many NUMA nodes][3].
Also, reference [Memory Manager KEP Simulation - how the Memory Manager works (by examples)][1]
illustrates how the management of groups occurs.

# # # Windows Support

Windows support can be enabled via the `WindowsCPUAndMemoryAffinity` feature gate
and it requires support in the container runtime.
Only the [BestEffort Policy](#policy-best-effort) is supported on Windows.

# # Memory Manager configuration

Other Managers should be first pre-configured. Next, the Memory Manager feature should be enabled
and be run with `Static` policy (section [Static policy](#policy-static)).
Optionally, some amount of memory can be reserved for system or kubelet processes to increase
node stability (section [Reserved memory flag](#reserved-memory-flag)).

# # # Policies

Memory Manager supports two policies. You can select a policy via a `kubelet` flag `--memory-manager-policy`

* `None` (default)
* `Static` (Linux only)
* `BestEffort` (Windows Only)

# # # # None policy #policy-none

This is the default policy and does not affect the memory allocation in any way.
It acts the same as if the Memory Manager is not present at all.

The `None` policy returns default topology hint. This special hint denotes that Hint Provider
(Memory Manager in this case) has no preference for NUMA affinity with any resource.

# # # # Static policy #policy-static

In the case of the `Guaranteed` pod, the `Static` Memory Manager policy returns topology hints
relating to the set of NUMA nodes where the memory can be guaranteed,
and reserves the memory through updating the internal [NodeMap][2] object.

In the case of the `BestEffort` or `Burstable` pod, the `Static` Memory Manager policy sends back
the default topology hint as there is no request for the guaranteed memory,
and does not reserve the memory in the internal [NodeMap][2] object.

This policy is only supported on Linux.

# # # # BestEffort policy #policy-best-effort

This policy is only supported on Windows.

On Windows, NUMA node assignment works differently than Linux.
There is no mechanism to ensure that Memory access only comes from a specific NUMA node.
Instead the Windows scheduler will select the most optimal NUMA node based on the CPU(s) assignments.
It is possible that Windows might use other NUMA nodes if deemed optimal by the Windows scheduler.

The policy does track the amount of memory available and requested through the internal [NodeMap][2].
The memory manager will make a best effort at ensuring that enough memory is available on
a NUMA node before making the assignment.
This means that in most cases memory assignment should function as expected.

# # # Reserved memory flag

The [Node Allocatable](docstasksadminister-clusterreserve-compute-resources) mechanism
is commonly used by node administrators to reserve K8S node system resources for the kubelet
or operating system processes in order to enhance the node stability.
A dedicated set of flags can be used for this purpose to set the total amount of reserved memory
for a node. This pre-configured value is subsequently utilized to calculate
the real amount of nodes allocatable memory available to pods.

The Kubernetes scheduler incorporates allocatable to optimise pod scheduling process.
The foregoing flags include `--kube-reserved`, `--system-reserved` and `--eviction-threshold`.
The sum of their values will account for the total amount of reserved memory.

A new `--reserved-memory` flag was added to Memory Manager to allow for this total reserved memory
to be split (by a node administrator) and accordingly reserved across many NUMA nodes.

The flag specifies a comma-separated list of memory reservations of different memory types per NUMA node.
Memory reservations across multiple NUMA nodes can be specified using semicolon as separator.
This parameter is only useful in the context of the Memory Manager feature.
The Memory Manager will not use this reserved memory for the allocation of container workloads.

For example, if you have a NUMA node NUMA0 with `10Gi` of memory available, and
the `--reserved-memory` was specified to reserve `1Gi` of memory at NUMA0,
the Memory Manager assumes that only `9Gi` is available for containers.

You can omit this parameter, however, you should be aware that the quantity of reserved memory
from all NUMA nodes should be equal to the quantity of memory specified by the
[Node Allocatable feature](docstasksadminister-clusterreserve-compute-resources).
If at least one node allocatable parameter is non-zero, you will need to specify
`--reserved-memory` for at least one NUMA node.
In fact, `eviction-hard` threshold value is equal to `100Mi` by default, so
if `Static` policy is used, `--reserved-memory` is obligatory.

Also, avoid the following configurations

1. duplicates, i.e. the same NUMA node or memory type, but with a different value
1. setting zero limit for any of memory types
1. NUMA node IDs that do not exist in the machine hardware
1. memory type names different than `memory` or `hugepages-`
   (hugepages of particular `` should also exist).

Syntax

`--reserved-memory Nmemory-type1value1,memory-type2value2,...`

* `N` (integer) - NUMA node index, e.g. `0`
* `memory-type` (string) - represents memory type
  * `memory` - conventional memory
  * `hugepages-2Mi` or `hugepages-1Gi` - hugepages
* `value` (string) - the quantity of reserved memory, e.g. `1Gi`

Example usage

`--reserved-memory 0memory1Gi,hugepages-1Gi2Gi`

or

`--reserved-memory 0memory1Gi --reserved-memory 1memory2Gi`

or

`--reserved-memory 0memory1Gi1memory2Gi`

When you specify values for `--reserved-memory` flag, you must comply with the setting that
you prior provided via Node Allocatable Feature flags.
That is, the following rule must be obeyed for each memory type

`sum(reserved-memory(i))  kube-reserved  system-reserved  eviction-threshold`,

where `i` is an index of a NUMA node.

If you do not follow the formula above, the Memory Manager will show an error on startup.

In other words, the example above illustrates that for the conventional memory (`typememory`),
we reserve `3Gi` in total, i.e.

`sum(reserved-memory(i))  reserved-memory(0)  reserved-memory(1)  1Gi  2Gi  3Gi`

An example of kubelet command-line arguments relevant to the node Allocatable configuration

* `--kube-reservedcpu500m,memory50Mi`
* `--system-reservedcpu123m,memory333Mi`
* `--eviction-hardmemory.available
The default hard eviction threshold is 100MiB, and **not** zero.
Remember to increase the quantity of memory that you reserve by setting `--reserved-memory`
by that hard eviction threshold. Otherwise, the kubelet will not start Memory Manager and
display an error.

Here is an example of a correct configuration

```shell
--kube-reservedcpu4,memory4Gi
--system-reservedcpu1,memory1Gi
--memory-manager-policyStatic
--reserved-memory 0memory3Gi1memory2148Mi
```

Prior to Kubernetes 1.32, you also need to add
```shell
--feature-gatesMemoryManagertrue
```

Let us validate the configuration above

1. `kube-reserved  system-reserved  eviction-hard(default)  reserved-memory(0)  reserved-memory(1)`
1. `4GiB  1GiB  100MiB  3GiB  2148MiB`
1. `5120MiB  100MiB  3072MiB  2148MiB`
1. `5220MiB  5220MiB` (which is correct)

# # Placing a Pod in the Guaranteed QoS class

If the selected policy is anything other than `None`, the Memory Manager identifies pods
that are in the `Guaranteed` QoS class.
The Memory Manager provides specific topology hints to the Topology Manager for each `Guaranteed` pod.
For pods in a QoS class other than `Guaranteed`, the Memory Manager provides default topology hints
to the Topology Manager.

The following excerpts from pod manifests assign a pod to the `Guaranteed` QoS class.

Pod with integer CPU(s) runs in the `Guaranteed` QoS class, when `requests` are equal to `limits`

```yaml
spec
  containers
  - name nginx
    image nginx
    resources
      limits
        memory 200Mi
        cpu 2
        example.comdevice 1
      requests
        memory 200Mi
        cpu 2
        example.comdevice 1
```

Also, a pod sharing CPU(s) runs in the `Guaranteed` QoS class, when `requests` are equal to `limits`.

```yaml
spec
  containers
  - name nginx
    image nginx
    resources
      limits
        memory 200Mi
        cpu 300m
        example.comdevice 1
      requests
        memory 200Mi
        cpu 300m
        example.comdevice 1
```

Notice that both CPU and memory requests must be specified for a Pod to lend it to Guaranteed QoS class.

# # Troubleshooting

The following means can be used to troubleshoot the reason why a pod could not be deployed or
became rejected at a node

- pod status - indicates topology affinity errors
- system logs - include valuable information for debugging, e.g., about generated hints
- state file - the dump of internal state of the Memory Manager
  (includes [Node Map and Memory Maps][2])
- starting from v1.22, the [device plugin resource API](#device-plugin-resource-api) can be used
  to retrieve information about the memory reserved for containers

# # # Pod status (TopologyAffinityError) #TopologyAffinityError

This error typically occurs in the following situations

* a node has not enough resources available to satisfy the pods request
* the pods request is rejected due to particular Topology Manager policy constraints

The error appears in the status of a pod

```shell
kubectl get pods
```

```none
NAME         READY   STATUS                  RESTARTS   AGE
guaranteed   01     TopologyAffinityError   0          113s
```

Use `kubectl describe pod ` or `kubectl get events` to obtain detailed error message

```none
Warning  TopologyAffinityError  10m   kubelet, dell8  Resources cannot be allocated with Topology locality
```

# # # System logs

Search system logs with respect to a particular pod.

The set of hints that Memory Manager generated for the pod can be found in the logs.
Also, the set of hints generated by CPU Manager should be present in the logs.

Topology Manager merges these hints to calculate a single best hint.
The best hint should be also present in the logs.

The best hint indicates where to allocate all the resources.
Topology Manager tests this hint against its current policy, and based on the verdict,
it either admits the pod to the node or rejects it.

Also, search the logs for occurrences associated with the Memory Manager,
e.g. to find out information about `cgroups` and `cpuset.mems` updates.

# # # Examine the memory manager state on a node

Let us first deploy a sample `Guaranteed` pod whose specification is as follows

```yaml
apiVersion v1
kind Pod
metadata
  name guaranteed
spec
  containers
  - name guaranteed
    image consumer
    imagePullPolicy Never
    resources
      limits
        cpu 2
        memory 150Gi
      requests
        cpu 2
        memory 150Gi
    command [sleep,infinity]
```

Next, let us log into the node where it was deployed and examine the state file in
`varlibkubeletmemory_manager_state`

```json

   policyNameStatic,
   machineState
      0
         numberOfAssignments1,
         memoryMap
            hugepages-1Gi
               total0,
               systemReserved0,
               allocatable0,
               reserved0,
               free0
            ,
            memory
               total134987354112,
               systemReserved3221225472,
               allocatable131766128640,
               reserved131766128640,
               free0

         ,
         nodes[
            0,
            1
         ]
      ,
      1
         numberOfAssignments1,
         memoryMap
            hugepages-1Gi
               total0,
               systemReserved0,
               allocatable0,
               reserved0,
               free0
            ,
            memory
               total135286722560,
               systemReserved2252341248,
               allocatable133034381312,
               reserved29295144960,
               free103739236352

         ,
         nodes[
            0,
            1
         ]

   ,
   entries
      fa9bdd38-6df9-4cf9-aa67-8c4814da37a8
         guaranteed[

               numaAffinity[
                  0,
                  1
               ],
               typememory,
               size161061273600

         ]

   ,
   checksum4142013182

```

It can be deduced from the state file that the pod was pinned to both NUMA nodes, i.e.

```json
numaAffinity[
   0,
   1
],
```

Pinned term means that pods memory consumption is constrained (through `cgroups` configuration)
to these NUMA nodes.

This automatically implies that Memory Manager instantiated a new group that
comprises these two NUMA nodes, i.e. `0` and `1` indexed NUMA nodes.

Notice that the management of groups is handled in a relatively complex manner, and
further elaboration is provided in Memory Manager KEP in [this][1] and [this][3] sections.

In order to analyse memory resources available in a group,the corresponding entries from
NUMA nodes belonging to the group must be added up.

For example, the total amount of free conventional memory in the group can be computed
by adding up the free memory available at every NUMA node in the group,
i.e., in the `memory` section of NUMA node `0` (`free0`) and NUMA node `1` (`free103739236352`).
So, the total amount of free conventional memory in this group is equal to `0  103739236352` bytes.

The line `systemReserved3221225472` indicates that the administrator of this node reserved
`3221225472` bytes (i.e. `3Gi`) to serve kubelet and system processes at NUMA node `0`,
by using `--reserved-memory` flag.

# # # Device plugin resource API

The kubelet provides a `PodResourceLister` gRPC service to enable discovery of resources and associated metadata.
By using its [List gRPC endpoint](docsconceptsextend-kubernetescompute-storage-netdevice-plugins#grpc-endpoint-list),
information about reserved memory for each container can be retrieved, which is contained
in protobuf `ContainerMemory` message.
This information can be retrieved solely for pods in Guaranteed QoS class.

# #  heading whatsnext

- [Memory Manager KEP Design Overview][4]
- [Memory Manager KEP Memory Maps at start-up (with examples)][5]
- [Memory Manager KEP Memory Maps at runtime (with examples)][6]
- [Memory Manager KEP Simulation - how the Memory Manager works (by examples)][1]
- [Memory Manager KEP The Concept of Node Map and Memory Maps][2]
- [Memory Manager KEP How to enable the guaranteed memory allocation over many NUMA nodes][3]

[1] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#simulation---how-the-memory-manager-works-by-examples
[2] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#the-concept-of-node-map-and-memory-maps
[3] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#how-to-enable-the-guaranteed-memory-allocation-over-many-numa-nodes
[4] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#design-overview
[5] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#memory-maps-at-start-up-with-examples
[6] httpsgithub.comkubernetesenhancementstreemasterkepssig-node1769-memory-manager#memory-maps-at-runtime-with-examples
