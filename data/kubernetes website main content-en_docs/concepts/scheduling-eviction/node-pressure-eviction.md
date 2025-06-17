---
title Node-pressure Eviction
content_type concept
weight 100
---

The _split image filesystem_ feature, which enables support for the `containerfs`
filesystem, adds several new eviction signals, thresholds and metrics. To use
`containerfs`, the Kubernetes release v requires the
`KubeletSeparateDiskGC` [feature gate](docsreferencecommand-line-tools-referencefeature-gates)
to be enabled. Currently, only CRI-O (v1.29 or higher) offers the `containerfs`
filesystem support.

The  monitors resources
like memory, disk space, and filesystem inodes on your clusters nodes.
When one or more of these resources reach specific consumption levels, the
kubelet can proactively fail one or more pods on the node to reclaim resources
and prevent starvation.

During a node-pressure eviction, the kubelet sets the [phase](docsconceptsworkloadspodspod-lifecycle#pod-phase) for the
selected pods to `Failed`, and terminates the Pod.

Node-pressure eviction is not the same as
[API-initiated eviction](docsconceptsscheduling-evictionapi-eviction).

The kubelet does not respect your configured
or the pods
`terminationGracePeriodSeconds`. If you use [soft eviction thresholds](#soft-eviction-thresholds),
the kubelet respects your configured `eviction-max-pod-grace-period`. If you use
[hard eviction thresholds](#hard-eviction-thresholds), the kubelet uses a `0s` grace period (immediate shutdown) for termination.

# # Self healing behavior

The kubelet attempts to [reclaim node-level resources](#reclaim-node-resources)
before it terminates end-user pods. For example, it removes unused container
images when disk resources are starved.

If the pods are managed by a
management object (such as
or ) that
replaces failed pods, the control plane (`kube-controller-manager`) creates new
pods in place of the evicted pods.

# # # Self healing for static pods

If you are running a [static pod](docsconceptsworkloadspods#static-pods)
on a node that is under resource pressure, the kubelet may evict that static
Pod. The kubelet then tries to create a replacement, because static Pods always
represent an intent to run a Pod on that node.

The kubelet takes the _priority_ of the static pod into account when creating
a replacement. If the static pod manifest specifies a low priority, and there
are higher-priority Pods defined within the clusters control plane, and the
node is under resource pressure, the kubelet may not be able to make room for
that static pod. The kubelet continues to attempt to run all static pods even
when there is resource pressure on a node.

# # Eviction signals and thresholds

The kubelet uses various parameters to make eviction decisions, like the following

- Eviction signals
- Eviction thresholds
- Monitoring intervals

# # # Eviction signals #eviction-signals

Eviction signals are the current state of a particular resource at a specific
point in time. The kubelet uses eviction signals to make eviction decisions by
comparing the signals to eviction thresholds, which are the minimum amount of
the resource that should be available on the node.

The kubelet uses the following eviction signals

 Eviction Signal           Description                                                                            Linux Only
-----------------------------------------------------------------------------------------------------------------------------
 `memory.available`        `memory.available`  `node.status.capacity[memory]` - `node.stats.memory.workingSet`
 `nodefs.available`        `nodefs.available`  `node.stats.fs.available`
 `nodefs.inodesFree`       `nodefs.inodesFree`  `node.stats.fs.inodesFree`
 `imagefs.available`       `imagefs.available`  `node.stats.runtime.imagefs.available`
 `imagefs.inodesFree`      `imagefs.inodesFree`  `node.stats.runtime.imagefs.inodesFree`
 `containerfs.available`   `containerfs.available`  `node.stats.runtime.containerfs.available`
 `containerfs.inodesFree`  `containerfs.inodesFree`  `node.stats.runtime.containerfs.inodesFree`
 `pid.available`           `pid.available`  `node.stats.rlimit.maxpid` - `node.stats.rlimit.curproc`

In this table, the **Description** column shows how kubelet gets the value of the
signal. Each signal supports either a percentage or a literal value. The kubelet
calculates the percentage value relative to the total capacity associated with
the signal.

# # # # Memory signals

On Linux nodes, the value for `memory.available` is derived from the cgroupfs instead of tools
like `free -m`. This is important because `free -m` does not work in a
container, and if users use the [node allocatable](docstasksadminister-clusterreserve-compute-resources#node-allocatable)
feature, out of resource decisions
are made local to the end user Pod part of the cgroup hierarchy as well as the
root node. This [script](examplesadminresourcememory-available.sh) or
[cgroupv2 script](examplesadminresourcememory-available-cgroupv2.sh)
reproduces the same set of steps that the kubelet performs to calculate
`memory.available`. The kubelet excludes inactive_file (the number of bytes of
file-backed memory on the inactive LRU list) from its calculation, as it assumes that
memory is reclaimable under pressure.

On Windows nodes, the value for `memory.available` is derived from the nodes global
memory commit levels (queried through the [`GetPerformanceInfo()`](httpslearn.microsoft.comwindowswin32apipsapinf-psapi-getperformanceinfo)
system call) by subtracting the nodes global [`CommitTotal`](httpslearn.microsoft.comwindowswin32apipsapins-psapi-performance_information) from the nodes [`CommitLimit`](httpslearn.microsoft.comwindowswin32apipsapins-psapi-performance_information). Please note that `CommitLimit` can change if the nodes page-file size changes!

# # # # Filesystem signals

The kubelet recognizes three specific filesystem identifiers that can be used with
eviction signals (`.inodesFree` or `.available`)

1. `nodefs` The nodes main filesystem, used for local disk volumes,
    emptyDir volumes not backed by memory, log storage, ephemeral storage,
    and more. For example, `nodefs` contains `varlibkubelet`.

1. `imagefs` An optional filesystem that container runtimes can use to store
   container images (which are the read-only layers) and container writable
   layers.

1. `containerfs` An optional filesystem that container runtime can use to
   store the writeable layers. Similar to the main filesystem (see `nodefs`),
   its used to store local disk volumes, emptyDir volumes not backed by memory,
   log storage, and ephemeral storage, except for the container images. When
   `containerfs` is used, the `imagefs` filesystem can be split to only store
   images (read-only layers) and nothing else.

As such, kubelet generally allows three options for container filesystems

- Everything is on the single `nodefs`, also referred to as rootfs or
  simply root, and there is no dedicated image filesystem.

- Container storage (see `nodefs`) is on a dedicated disk, and `imagefs`
  (writable and read-only layers) is separate from the root filesystem.
  This is often referred to as split disk (or separate disk) filesystem.

- Container filesystem `containerfs` (same as `nodefs` plus writable
  layers) is on root and the container images (read-only layers) are
  stored on separate `imagefs`. This is often referred to as split image
  filesystem.

The kubelet will attempt to auto-discover these filesystems with their current
configuration directly from the underlying container runtime and will ignore
other local node filesystems.

The kubelet does not support other container filesystems or storage configurations,
and it does not currently support multiple filesystems for images and containers.

# # # Deprecated kubelet garbage collection features

Some kubelet garbage collection features are deprecated in favor of eviction

 Existing Flag  Rationale
 -------------  ---------
 `--maximum-dead-containers`  deprecated once old logs are stored outside of containers context
 `--maximum-dead-containers-per-container`  deprecated once old logs are stored outside of containers context
 `--minimum-container-ttl-duration`  deprecated once old logs are stored outside of containers context

# # # Eviction thresholds

You can specify custom eviction thresholds for the kubelet to use when it makes
eviction decisions. You can configure [soft](#soft-eviction-thresholds) and
[hard](#hard-eviction-thresholds) eviction thresholds.

Eviction thresholds have the form `[eviction-signal][operator][quantity]`, where

- `eviction-signal` is the [eviction signal](#eviction-signals) to use.
- `operator` is the [relational operator](httpsen.wikipedia.orgwikiRelational_operator#Standard_relational_operators)
  you want, such as `
The kubelet does not use the pods [QoS class](docsconceptsworkloadspodspod-qos) to determine the eviction order.
You can use the QoS class to estimate the most likely pod eviction order when
reclaiming resources like memory. QoS classification does not apply to EphemeralStorage requests,
so the above scenario will not apply if the node is, for example, under `DiskPressure`.

`Guaranteed` pods are guaranteed only when requests and limits are specified for
all the containers and they are equal. These pods will never be evicted because
of another pods resource consumption. If a system daemon (such as `kubelet`
and `journald`) is consuming more resources than were reserved via
`system-reserved` or `kube-reserved` allocations, and the node only has
`Guaranteed` or `Burstable` pods using less resources than requests left on it,
then the kubelet must choose to evict one of these pods to preserve node stability
and to limit the impact of resource starvation on other pods. In this case, it
will choose to evict pods of lowest Priority first.

If you are running a [static pod](docsconceptsworkloadspods#static-pods)
and want to avoid having it evicted under resource pressure, set the
`priority` field for that Pod directly. Static pods do not support the
`priorityClassName` field.

When the kubelet evicts pods in response to inode or process ID starvation, it uses
the Pods relative priority to determine the eviction order, because inodes and PIDs have no
requests.

The kubelet sorts pods differently based on whether the node has a dedicated
`imagefs` or `containerfs` filesystem

# # # # Without `imagefs` or `containerfs` (`nodefs` and `imagefs` use the same filesystem) #without-imagefs

- If `nodefs` triggers evictions, the kubelet sorts pods based on their
  total disk usage (`local volumes  logs and a writable layer of all containers`).

# # # # With `imagefs` (`nodefs` and `imagefs` filesystems are separate) #with-imagefs

- If `nodefs` triggers evictions, the kubelet sorts pods based on `nodefs`
  usage (`local volumes  logs of all containers`).

- If `imagefs` triggers evictions, the kubelet sorts pods based on the
  writable layer usage of all containers.

# # # # With `imagesfs` and `containerfs` (`imagefs` and `containerfs` have been split) #with-containersfs

- If `containerfs` triggers evictions, the kubelet sorts pods based on
  `containerfs` usage (`local volumes  logs and a writable layer of all containers`).

- If `imagefs` triggers evictions, the kubelet sorts pods based on the
  `storage of images` rank, which represents the disk usage of a given image.

# # # Minimum eviction reclaim

As of Kubernetes v, you cannot set a custom value
for the `containerfs.available` metric. The configuration for this specific
metric will be set automatically to reflect values set for either the `nodefs`
or `imagefs`, depending on the configuration.

In some cases, pod eviction only reclaims a small amount of the starved resource.
This can lead to the kubelet repeatedly hitting the configured eviction thresholds
and triggering multiple evictions.

You can use the `--eviction-minimum-reclaim` flag or a [kubelet config file](docstasksadminister-clusterkubelet-config-file)
to configure a minimum reclaim amount for each resource. When the kubelet notices
that a resource is starved, it continues to reclaim that resource until it
reclaims the quantity you specify.

For example, the following configuration sets minimum reclaim amounts

```yaml
apiVersion kubelet.config.k8s.iov1beta1
kind KubeletConfiguration
evictionHard
  memory.available 500Mi
  nodefs.available 1Gi
  imagefs.available 100Gi
evictionMinimumReclaim
  memory.available 0Mi
  nodefs.available 500Mi
  imagefs.available 2Gi
```

In this example, if the `nodefs.available` signal meets the eviction threshold,
the kubelet reclaims the resource until the signal reaches the threshold of 1GiB,
and then continues to reclaim the minimum amount of 500MiB, until the available
nodefs storage value reaches 1.5GiB.

Similarly, the kubelet tries to reclaim the `imagefs` resource until the `imagefs.available`
value reaches `102Gi`, representing 102 GiB of available container image storage. If the amount
of storage that the kubelet could reclaim is less than 2GiB, the kubelet doesnt reclaim anything.

The default `eviction-minimum-reclaim` is `0` for all resources.

# # Node out of memory behavior

If the node experiences an _out of memory_ (OOM) event prior to the kubelet
being able to reclaim memory, the node depends on the [oom_killer](httpslwn.netArticles391222)
to respond.

The kubelet sets an `oom_score_adj` value for each container based on the QoS for the pod.

 Quality of Service  `oom_score_adj`
-------------------------------------------------------------------------------------------------------
 `Guaranteed`        -997
 `BestEffort`        1000
 `Burstable`         _min(max(2, 1000 - (1000  memoryRequestBytes)  machineMemoryCapacityBytes), 999)_

The kubelet also sets an `oom_score_adj` value of `-997` for any containers in Pods that have
`system-node-critical` .

If the kubelet cant reclaim memory before a node experiences OOM, the
`oom_killer` calculates an `oom_score` based on the percentage of memory its
using on the node, and then adds the `oom_score_adj` to get an effective `oom_score`
for each container. It then kills the container with the highest score.

This means that containers in low QoS pods that consume a large amount of memory
relative to their scheduling requests are killed first.

Unlike pod eviction, if a container is OOM killed, the kubelet can restart it
based on its `restartPolicy`.

# # Good practices #node-pressure-eviction-good-practices

The following sections describe good practice for eviction configuration.

# # # Schedulable resources and eviction policies

When you configure the kubelet with an eviction policy, you should make sure that
the scheduler will not schedule pods if they will trigger eviction because they
immediately induce memory pressure.

Consider the following scenario

- Node memory capacity 10GiB
- Operator wants to reserve 10 of memory capacity for system daemons (kernel, `kubelet`, etc.)
- Operator wants to evict Pods at 95 memory utilization to reduce incidence of system OOM.

For this to work, the kubelet is launched as follows

```none
--eviction-hardmemory.available#create-eviction-pod-v1-core)
