---
reviewers
- vincepri
- bart0sh
title Container Runtimes
content_type concept
weight 20
---

 dockershim-removal

You need to install a

into each node in the cluster so that Pods can run there. This page outlines
what is involved and describes related tasks for setting up nodes.

Kubernetes  requires that you use a runtime that
conforms with the
 (CRI).

See [CRI version support](#cri-versions) for more information.

This page provides an outline of how to use several common container runtimes with
Kubernetes.

- [containerd](#containerd)
- [CRI-O](#cri-o)
- [Docker Engine](#docker)
- [Mirantis Container Runtime](#mcr)

Kubernetes releases before v1.24 included a direct integration with Docker Engine,
using a component named _dockershim_. That special direct integration is no longer
part of Kubernetes (this removal was
[announced](blog20201208kubernetes-1-20-release-announcement#dockershim-deprecation)
as part of the v1.20 release).
You can read
[Check whether Dockershim removal affects you](docstasksadminister-clustermigrating-from-dockershimcheck-if-dockershim-removal-affects-you)
to understand how this removal might affect you. To learn about migrating from using dockershim, see
[Migrating from dockershim](docstasksadminister-clustermigrating-from-dockershim).

If you are running a version of Kubernetes other than v,
check the documentation for that version.

# # Install and configure prerequisites

# # # Network configuration

By default, the Linux kernel does not allow IPv4 packets to be routed
between interfaces. Most Kubernetes cluster networking implementations
will change this setting (if needed), but some might expect the
administrator to do it for them. (Some might also expect other sysctl
parameters to be set, kernel modules to be loaded, etc consult the
documentation for your specific network implementation.)

# # # Enable IPv4 packet forwarding #prerequisite-ipv4-forwarding-optional

To manually enable IPv4 packet forwarding

```bash
# sysctl params required by setup, params persist across reboots
cat
are used to constrain resources that are allocated to processes.

Both the  and the
underlying container runtime need to interface with control groups to enforce
[resource management for pods and containers](docsconceptsconfigurationmanage-resources-containers)
and set resources such as cpumemory requests and limits. To interface with control
groups, the kubelet and the container runtime need to use a *cgroup driver*.
Its critical that the kubelet and the container runtime use the same cgroup
driver and are configured the same.

There are two cgroup drivers available

* [`cgroupfs`](#cgroupfs-cgroup-driver)
* [`systemd`](#systemd-cgroup-driver)

# # # cgroupfs driver #cgroupfs-cgroup-driver

The `cgroupfs` driver is the [default cgroup driver in the kubelet](docsreferenceconfig-apikubelet-config.v1beta1).
 When the `cgroupfs` driver is used, the kubelet and the container runtime directly interface with
 the cgroup filesystem to configure cgroups.

The `cgroupfs` driver is **not** recommended when
[systemd](httpswww.freedesktop.orgwikiSoftwaresystemd) is the
init system because systemd expects a single cgroup manager on
the system. Additionally, if you use [cgroup v2](docsconceptsarchitecturecgroups), use the `systemd`
cgroup driver instead of `cgroupfs`.

# # # systemd cgroup driver #systemd-cgroup-driver

When [systemd](httpswww.freedesktop.orgwikiSoftwaresystemd) is chosen as the init
system for a Linux distribution, the init process generates and consumes a root control group
(`cgroup`) and acts as a cgroup manager.

systemd has a tight integration with cgroups and allocates a cgroup per systemd
unit. As a result, if you use `systemd` as the init system with the `cgroupfs`
driver, the system gets two different cgroup managers.

Two cgroup managers result in two views of the available and in-use resources in
the system. In some cases, nodes that are configured to use `cgroupfs` for the
kubelet and container runtime, but use `systemd` for the rest of the processes become
unstable under resource pressure.

The approach to mitigate this instability is to use `systemd` as the cgroup driver for
the kubelet and the container runtime when systemd is the selected init system.

To set `systemd` as the cgroup driver, edit the
[`KubeletConfiguration`](docstasksadminister-clusterkubelet-config-file)
option of `cgroupDriver` and set it to `systemd`. For example

```yaml
apiVersion kubelet.config.k8s.iov1beta1
kind KubeletConfiguration
...
cgroupDriver systemd
```

Starting with v1.22 and later, when creating a cluster with kubeadm, if the user does not set
the `cgroupDriver` field under `KubeletConfiguration`, kubeadm defaults it to `systemd`.

If you configure `systemd` as the cgroup driver for the kubelet, you must also
configure `systemd` as the cgroup driver for the container runtime. Refer to
the documentation for your container runtime for instructions. For example

*  [containerd](#containerd-systemd)
*  [CRI-O](#cri-o)

In Kubernetes , with the `KubeletCgroupDriverFromCRI`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates)
enabled and a container runtime that supports the `RuntimeConfig` CRI RPC,
the kubelet automatically detects the appropriate cgroup driver from the runtime,
and ignores the `cgroupDriver` setting within the kubelet configuration.

Changing the cgroup driver of a Node that has joined a cluster is a sensitive operation.
If the kubelet has created Pods using the semantics of one cgroup driver, changing the container
runtime to another cgroup driver can cause errors when trying to re-create the Pod sandbox
for such existing Pods. Restarting the kubelet may not solve such errors.

If you have automation that makes it feasible, replace the node with another using the updated
configuration, or reinstall it using automation.

# # # Migrating to the `systemd` driver in kubeadm managed clusters

If you wish to migrate to the `systemd` cgroup driver in existing kubeadm managed clusters,
follow [configuring a cgroup driver](docstasksadminister-clusterkubeadmconfigure-cgroup-driver).

# # CRI version support #cri-versions

Your container runtime must support at least v1alpha2 of the container runtime interface.

Kubernetes [starting v1.26](blog20221118upcoming-changes-in-kubernetes-1-26#cri-api-removal)
_only works_ with v1 of the CRI API. Earlier versions default
to v1 version, however if a container runtime does not support the v1 API, the kubelet falls back to
using the (deprecated) v1alpha2 API instead.

# # Container runtimes

 thirdparty-content

# # # containerd

This section outlines the necessary steps to use containerd as CRI runtime.

To install containerd on your system, follow the instructions on
[getting started with containerd](httpsgithub.comcontainerdcontainerdblobmaindocsgetting-started.md).
Return to this step once youve created a valid `config.toml` configuration file.

 tab nameLinux
You can find this file under the path `etccontainerdconfig.toml`.
 tab
 tab nameWindows
You can find this file under the path `CProgram Filescontainerdconfig.toml`.
 tab

On Linux the default CRI socket for containerd is `runcontainerdcontainerd.sock`.
On Windows the default CRI endpoint is `npipe.pipecontainerd-containerd`.

# # # # Configuring the `systemd` cgroup driver #containerd-systemd

To use the `systemd` cgroup driver in `etccontainerdconfig.toml` with `runc`, set

```
[plugins.io.containerd.grpc.v1.cri.containerd.runtimes.runc]
  ...
  [plugins.io.containerd.grpc.v1.cri.containerd.runtimes.runc.options]
    SystemdCgroup  true
```

The `systemd` cgroup driver is recommended if you use [cgroup v2](docsconceptsarchitecturecgroups).

If you installed containerd from a package (for example, RPM or `.deb`), you may find
that the CRI integration plugin is disabled by default.

You need CRI support enabled to use containerd with Kubernetes. Make sure that `cri`
is not included in the`disabled_plugins` list within `etccontainerdconfig.toml`
if you made changes to that file, also restart `containerd`.

If you experience container crash loops after the initial cluster installation or after
installing a CNI, the containerd configuration provided with the package might contain
incompatible configuration parameters. Consider resetting the containerd configuration
with `containerd config default  etccontainerdconfig.toml` as specified in
[getting-started.md](httpsgithub.comcontainerdcontainerdblobmaindocsgetting-started.md#advanced-topics)
and then set the configuration parameters specified above accordingly.

If you apply this change, make sure to restart containerd

```shell
sudo systemctl restart containerd
```

When using kubeadm, manually configure the
[cgroup driver for kubelet](docstasksadminister-clusterkubeadmconfigure-cgroup-driver#configuring-the-kubelet-cgroup-driver).

In Kubernetes v1.28, you can enable automatic detection of the
cgroup driver as an alpha feature. See [systemd cgroup driver](#systemd-cgroup-driver)
for more details.

# # # # Overriding the sandbox (pause) image #override-pause-image-containerd

In your [containerd config](httpsgithub.comcontainerdcontainerdblobmaindocscriconfig.md) you can overwrite the
sandbox image by setting the following config

```toml
[plugins.io.containerd.grpc.v1.cri]
  sandbox_image  registry.k8s.iopause3.10
```

You might need to restart `containerd` as well once youve updated the config file `systemctl restart containerd`.

# # # CRI-O

This section contains the necessary steps to install CRI-O as a container runtime.

To install CRI-O, follow [CRI-O Install Instructions](httpsgithub.comcri-opackagingblobmainREADME.md#usage).

# # # # cgroup driver

CRI-O uses the systemd cgroup driver per default, which is likely to work fine
for you. To switch to the `cgroupfs` cgroup driver, either edit
`etccriocrio.conf` or place a drop-in configuration in
`etccriocrio.conf.d02-cgroup-manager.conf`, for example

```toml
[crio.runtime]
conmon_cgroup  pod
cgroup_manager  cgroupfs
```

You should also note the changed `conmon_cgroup`, which has to be set to the value
`pod` when using CRI-O with `cgroupfs`. It is generally necessary to keep the
cgroup driver configuration of the kubelet (usually done via kubeadm) and CRI-O
in sync.

In Kubernetes v1.28, you can enable automatic detection of the
cgroup driver as an alpha feature. See [systemd cgroup driver](#systemd-cgroup-driver)
for more details.

For CRI-O, the CRI socket is `varruncriocrio.sock` by default.

# # # # Overriding the sandbox (pause) image #override-pause-image-cri-o

In your [CRI-O config](httpsgithub.comcri-ocri-oblobmaindocscrio.conf.5.md) you can set the following
config value

```toml
[crio.image]
pause_imageregistry.k8s.iopause3.10
```

This config option supports live configuration reload to apply this change `systemctl reload crio` or by sending
`SIGHUP` to the `crio` process.

# # # Docker Engine #docker

These instructions assume that you are using the
[`cri-dockerd`](httpsmirantis.github.iocri-dockerd) adapter to integrate
Docker Engine with Kubernetes.

1. On each of your nodes, install Docker for your Linux distribution as per
  [Install Docker Engine](httpsdocs.docker.comengineinstall#server).

2. Install [`cri-dockerd`](httpsmirantis.github.iocri-dockerdusageinstall), following the directions in the install section of the documentation.

For `cri-dockerd`, the CRI socket is `runcri-dockerd.sock` by default.

# # # Mirantis Container Runtime #mcr

[Mirantis Container Runtime](httpsdocs.mirantis.commcr20.10overview.html) (MCR) is a commercially
available container runtime that was formerly known as Docker Enterprise Edition.

You can use Mirantis Container Runtime with Kubernetes using the open source
[`cri-dockerd`](httpsmirantis.github.iocri-dockerd) component, included with MCR.

To learn more about how to install Mirantis Container Runtime,
visit [MCR Deployment Guide](httpsdocs.mirantis.commcr20.10install.html).

Check the systemd unit named `cri-docker.socket` to find out the path to the CRI
socket.

# # # # Overriding the sandbox (pause) image #override-pause-image-cri-dockerd-mcr

The `cri-dockerd` adapter accepts a command line argument for
specifying which container image to use as the Pod infrastructure container (pause image).
The command line argument to use is `--pod-infra-container-image`.

# #  heading whatsnext

As well as a container runtime, your cluster will need a working
[network plugin](docsconceptscluster-administrationnetworking#how-to-implement-the-kubernetes-network-model).
