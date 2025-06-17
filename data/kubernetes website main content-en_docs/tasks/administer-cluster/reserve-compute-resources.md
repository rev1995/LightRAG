---
reviewers
- vishh
- derekwaynecarr
- dashpole
title Reserve Compute Resources for System Daemons
content_type task
weight 290
---

Kubernetes nodes can be scheduled to `Capacity`. Pods can consume all the
available capacity on a node by default. This is an issue because nodes
typically run quite a few system daemons that power the OS and Kubernetes
itself. Unless resources are set aside for these system daemons, pods and system
daemons compete for resources and lead to resource starvation issues on the
node.

The `kubelet` exposes a feature named Node Allocatable that helps to reserve
compute resources for system daemons. Kubernetes recommends cluster
administrators to configure Node Allocatable based on their workload density
on each node.

# #  heading prerequisites

You can configure below kubelet [configuration settings](docsreferenceconfig-apikubelet-config.v1beta1)
using the [kubelet configuration file](docstasksadminister-clusterkubelet-config-file).

# # Node Allocatable

![node capacity](imagesdocsnode-capacity.svg)

Allocatable on a Kubernetes node is defined as the amount of compute resources
that are available for pods. The scheduler does not over-subscribe
Allocatable. CPU, memory and ephemeral-storage are supported as of now.

Node Allocatable is exposed as part of `v1.Node` object in the API and as part
of `kubectl describe node` in the CLI.

Resources can be reserved for two categories of system daemons in the `kubelet`.

# # # Enabling QoS and Pod level cgroups

To properly enforce node allocatable constraints on the node, you must
enable the new cgroup hierarchy via the `cgroupsPerQOS` setting. This setting is
enabled by default. When enabled, the `kubelet` will parent all end-user pods
under a cgroup hierarchy managed by the `kubelet`.

# # # Configuring a cgroup driver

The `kubelet` supports manipulation of the cgroup hierarchy on
the host using a cgroup driver. The driver is configured via the `cgroupDriver` setting.

The supported values are the following

* `cgroupfs` is the default driver that performs direct manipulation of the
cgroup filesystem on the host in order to manage cgroup sandboxes.
* `systemd` is an alternative driver that manages cgroup sandboxes using
transient slices for resources that are supported by that init system.

Depending on the configuration of the associated container runtime,
operators may have to choose a particular cgroup driver to ensure
proper system behavior. For example, if operators use the `systemd`
cgroup driver provided by the `containerd` runtime, the `kubelet` must
be configured to use the `systemd` cgroup driver.

# # # Kube Reserved

- **KubeletConfiguration Setting** `kubeReserved `. Example value `cpu 100m, memory 100Mi, ephemeral-storage 1Gi, pid1000`
- **KubeletConfiguration Setting** `kubeReservedCgroup `

`kubeReserved` is meant to capture resource reservation for kubernetes system
daemons like the `kubelet`, `container runtime`, etc.
It is not meant to reserve resources for system daemons that are run as pods.
`kubeReserved` is typically a function of `pod density` on the nodes.

In addition to `cpu`, `memory`, and `ephemeral-storage`, `pid` may be
specified to reserve the specified number of process IDs for
kubernetes system daemons.

To optionally enforce `kubeReserved` on kubernetes system daemons, specify the parent
control group for kube daemons as the value for `kubeReservedCgroup` setting,
and [add `kube-reserved` to `enforceNodeAllocatable`](#enforcing-node-allocatable).

It is recommended that the kubernetes system daemons are placed under a top
level control group (`runtime.slice` on systemd machines for example). Each
system daemon should ideally run within its own child control group. Refer to
[the design proposal](httpsgit.k8s.iodesign-proposals-archivenodenode-allocatable.md#recommended-cgroups-setup)
for more details on recommended control group hierarchy.

Note that Kubelet **does not** create `kubeReservedCgroup` if it doesnt
exist. The kubelet will fail to start if an invalid cgroup is specified. With `systemd`
cgroup driver, you should follow a specific pattern for the name of the cgroup you
define the name should be the value you set for `kubeReservedCgroup`,
with `.slice` appended.

# # # System Reserved

- **KubeletConfiguration Setting** `systemReserved `. Example value `cpu 100m, memory 100Mi, ephemeral-storage 1Gi, pid1000`
- **KubeletConfiguration Setting** `systemReservedCgroup `

`systemReserved` is meant to capture resource reservation for OS system daemons
like `sshd`, `udev`, etc. `systemReserved` should reserve `memory` for the
`kernel` too since `kernel` memory is not accounted to pods in Kubernetes at this time.
Reserving resources for user login sessions is also recommended (`user.slice` in
systemd world).

In addition to `cpu`, `memory`, and `ephemeral-storage`, `pid` may be
specified to reserve the specified number of process IDs for OS system
daemons.

To optionally enforce `systemReserved` on system daemons, specify the parent
control group for OS system daemons as the value for `systemReservedCgroup` setting,
and [add `system-reserved` to `enforceNodeAllocatable`](#enforcing-node-allocatable).

It is recommended that the OS system daemons are placed under a top level
control group (`system.slice` on systemd machines for example).

Note that `kubelet` **does not** create `systemReservedCgroup` if it doesnt
exist. `kubelet` will fail if an invalid cgroup is specified.  With `systemd`
cgroup driver, you should follow a specific pattern for the name of the cgroup you
define the name should be the value you set for `systemReservedCgroup`,
with `.slice` appended.

# # # Explicitly Reserved CPU List

**KubeletConfiguration Setting** `reservedSystemCPUs`. Example value `0-3`

`reservedSystemCPUs` is meant to define an explicit CPU set for OS system daemons and
kubernetes system daemons. `reservedSystemCPUs` is for systems that do not intend to
define separate top level cgroups for OS system daemons and kubernetes system daemons
with regard to cpuset resource.
If the Kubelet **does not** have `kubeReservedCgroup` and `systemReservedCgroup`,
the explicit cpuset provided by `reservedSystemCPUs` will take precedence over the CPUs
defined by `kubeReservedCgroup` and `systemReservedCgroup` options.

This option is specifically designed for TelcoNFV use cases where uncontrolled
interruptstimers may impact the workload performance. you can use this option
to define the explicit cpuset for the systemkubernetes daemons as well as the
interruptstimers, so the rest CPUs on the system can be used exclusively for
workloads, with less impact from uncontrolled interruptstimers. To move the
system daemon, kubernetes daemons and interruptstimers to the explicit cpuset
defined by this option, other mechanism outside Kubernetes should be used.
For example in Centos, you can do this using the tuned toolset.

# # # Eviction Thresholds

**KubeletConfiguration Setting** `evictionHard memory.available 100Mi, nodefs.available 10, nodefs.inodesFree 5, imagefs.available 15`. Example value `memory.available

# # Example Scenario

Here is an example to illustrate Node Allocatable computation

* Node has `32Gi` of `memory`, `16 CPUs` and `100Gi` of `Storage`
* `kubeReserved` is set to `cpu 1000m, memory 2Gi, ephemeral-storage 1Gi`
* `systemReserved` is set to `cpu 500m, memory 1Gi, ephemeral-storage 1Gi`
* `evictionHard` is set to `memory.available 500Mi, nodefs.available 10`

Under this scenario, Allocatable will be 14.5 CPUs, 28.5Gi of memory and
`88Gi` of local storage.
Scheduler ensures that the total memory `requests` across all pods on this node does
not exceed 28.5Gi and storage doesnt exceed 88Gi.
Kubelet evicts pods whenever the overall memory usage across pods exceeds 28.5Gi,
or if overall disk usage exceeds 88Gi. If all processes on the node consume as
much CPU as they can, pods together cannot consume more than 14.5 CPUs.

If `kubeReserved` andor `systemReserved` is not enforced and system daemons
exceed their reservation, `kubelet` evicts pods whenever the overall node memory
usage is higher than 31.5Gi or `storage` is greater than 90Gi.
