---
title Running Kubernetes Node Components as a Non-root User
content_type task
min-kubernetes-server-version 1.22
weight 300
---

This document describes how to run Kubernetes Node components such as kubelet, CRI, OCI, and CNI
without root privileges, by using a .

This technique is also known as _rootless mode_.

This document describes how to run Kubernetes Node components (and hence pods) as a non-root user.

If you are just looking for how to run a pod as a non-root user, see [SecurityContext](docstasksconfigure-pod-containersecurity-context).

# #  heading prerequisites

 version-check

* [Enable Cgroup v2](httpsrootlesscontaine.rsgetting-startedcommoncgroup2)
* [Enable systemd with user session](httpsrootlesscontaine.rsgetting-startedcommonlogin)
* [Configure several sysctl values, depending on host Linux distribution](httpsrootlesscontaine.rsgetting-startedcommonsysctl)
* [Ensure that your unprivileged user is listed in `etcsubuid` and `etcsubgid`](httpsrootlesscontaine.rsgetting-startedcommonsubuid)
* Enable the `KubeletInUserNamespace` [feature gate](docsreferencecommand-line-tools-referencefeature-gates)

# # Running Kubernetes inside Rootless DockerPodman

# # # kind

[kind](httpskind.sigs.k8s.io) supports running Kubernetes inside Rootless Docker or Rootless Podman.

See [Running kind with Rootless Docker](httpskind.sigs.k8s.iodocsuserrootless).

# # # minikube

[minikube](httpsminikube.sigs.k8s.io) also supports running Kubernetes inside Rootless Docker or Rootless Podman.

See the Minikube documentation

* [Rootless Docker](httpsminikube.sigs.k8s.iodocsdriversdocker)
* [Rootless Podman](httpsminikube.sigs.k8s.iodocsdriverspodman)

# # Running Kubernetes inside Unprivileged Containers

 thirdparty-content

# # # sysbox

[Sysbox](httpsgithub.comnestyboxsysbox) is an open-source container runtime
(similar to runc) that supports running system-level workloads such as Docker
and Kubernetes inside unprivileged containers isolated with the Linux user
namespace.

See [Sysbox Quick Start Guide Kubernetes-in-Docker](httpsgithub.comnestyboxsysboxblobmasterdocsquickstartkind.md) for more info.

Sysbox supports running Kubernetes inside unprivileged containers without
requiring Cgroup v2 and without the `KubeletInUserNamespace` feature gate. It
does this by exposing specially crafted `proc` and `sys` filesystems inside
the container plus several other advanced OS virtualization techniques.

# # Running Rootless Kubernetes directly on a host

 thirdparty-content

# # # K3s

[K3s](httpsk3s.io) experimentally supports rootless mode.

See [Running K3s with Rootless mode](httpsrancher.comdocsk3slatestenadvanced#running-k3s-with-rootless-mode-experimental) for the usage.

# # # Usernetes
[Usernetes](httpsgithub.comrootless-containersusernetes) is a reference distribution of Kubernetes that can be installed under `HOME` directory without the root privilege.

Usernetes supports both containerd and CRI-O as CRI runtimes.
Usernetes supports multi-node clusters using Flannel (VXLAN).

See [the Usernetes repo](httpsgithub.comrootless-containersusernetes) for the usage.

# # Manually deploy a node that runs the kubelet in a user namespace #userns-the-hard-way

This section provides hints for running Kubernetes in a user namespace manually.

This section is intended to be read by developers of Kubernetes distributions, not by end users.

# # # Creating a user namespace

The first step is to create a .

If you are trying to run Kubernetes in a user-namespaced container such as
Rootless DockerPodman or LXCLXD, you are all set, and you can go to the next subsection.

Otherwise you have to create a user namespace by yourself, by calling `unshare(2)` with `CLONE_NEWUSER`.

A user namespace can be also unshared by using command line tools such as

- [`unshare(1)`](httpsman7.orglinuxman-pagesman1unshare.1.html)
- [RootlessKit](httpsgithub.comrootless-containersrootlesskit)
- [become-root](httpsgithub.comgiuseppebecome-root)

After unsharing the user namespace, you will also have to unshare other namespaces such as mount namespace.

You do *not* need to call `chroot()` nor `pivot_root()` after unsharing the mount namespace,
however, you have to mount writable filesystems on several directories *in* the namespace.

At least, the following directories need to be writable *in* the namespace (not *outside* the namespace)

- `etc`
- `run`
- `varlogs`
- `varlibkubelet`
- `varlibcni`
- `varlibcontainerd` (for containerd)
- `varlibcontainers` (for CRI-O)

# # # Creating a delegated cgroup tree

In addition to the user namespace, you also need to have a writable cgroup tree with cgroup v2.

Kubernetes support for running Node components in user namespaces requires cgroup v2.
Cgroup v1 is not supported.

If you are trying to run Kubernetes in Rootless DockerPodman or LXCLXD on a systemd-based host, you are all set.

Otherwise you have to create a systemd unit with `Delegateyes` property to delegate a cgroup tree with writable permission.

On your node, systemd must already be configured to allow delegation for more details, see
[cgroup v2](httpsrootlesscontaine.rsgetting-startedcommoncgroup2) in the Rootless
Containers documentation.

# # # Configuring network

 thirdparty-content

The network namespace of the Node components has to have a non-loopback interface, which can be for example configured with
[slirp4netns](httpsgithub.comrootless-containersslirp4netns),
[VPNKit](httpsgithub.commobyvpnkit), or
[lxc-user-nic(1)](httpswww.man7.orglinuxman-pagesman1lxc-user-nic.1.html).

The network namespaces of the Pods can be configured with regular CNI plugins.
For multi-node networking, Flannel (VXLAN, 8472UDP) is known to work.

Ports such as the kubelet port (10250TCP) and `NodePort` service ports have to be exposed from the Node network namespace to
the host with an external port forwarder, such as RootlessKit, slirp4netns, or
[socat(1)](httpslinux.die.netman1socat).

You can use the port forwarder from K3s.
See [Running K3s in Rootless Mode](httpsrancher.comdocsk3slatestenadvanced#known-issues-with-rootless-mode)
for more details.
The implementation can be found in [the `pkgrootlessports` package](httpsgithub.comk3s-iok3sblobv1.22.3k3s1pkgrootlessportscontroller.go) of k3s.

# # # Configuring CRI

The kubelet relies on a container runtime. You should deploy a container runtime such as
containerd or CRI-O and ensure that it is running within the user namespace before the kubelet starts.

 tab namecontainerd

Running CRI plugin of containerd in a user namespace is supported since containerd 1.4.

Running containerd within a user namespace requires the following configurations.

```toml
version  2

[plugins.io.containerd.grpc.v1.cri]
# Disable AppArmor
  disable_apparmor  true
# Ignore an error during setting oom_score_adj
  restrict_oom_score_adj  true
# Disable hugetlb cgroup v2 controller (because systemd does not support delegating hugetlb controller)
  disable_hugetlb_controller  true

[plugins.io.containerd.grpc.v1.cri.containerd]
# Using non-fuse overlayfs is also possible for kernel  5.11, but requires SELinux to be disabled
  snapshotter  fuse-overlayfs

[plugins.io.containerd.grpc.v1.cri.containerd.runtimes.runc.options]
# We use cgroupfs that is delegated by systemd, so we do not use SystemdCgroup driver
# (unless you run another systemd in the namespace)
  SystemdCgroup  false
```

The default path of the configuration file is `etccontainerdconfig.toml`.
The path can be specified with `containerd -c pathtocontainerdconfig.toml`.

 tab
 tab nameCRI-O

Running CRI-O in a user namespace is supported since CRI-O 1.22.

CRI-O requires an environment variable `_CRIO_ROOTLESS1` to be set.

The following configurations are also recommended

```toml
[crio]
  storage_driver  overlay
# Using non-fuse overlayfs is also possible for kernel  5.11, but requires SELinux to be disabled
  storage_option  [overlay.mount_programusrlocalbinfuse-overlayfs]

[crio.runtime]
# We use cgroupfs that is delegated by systemd, so we do not use systemd driver
# (unless you run another systemd in the namespace)
  cgroup_manager  cgroupfs
```

The default path of the configuration file is `etccriocrio.conf`.
The path can be specified with `crio --config pathtocriocrio.conf`.
 tab

# # # Configuring kubelet

Running kubelet in a user namespace requires the following configuration

```yaml
apiVersion kubelet.config.k8s.iov1beta1
kind KubeletConfiguration
featureGates
  KubeletInUserNamespace true
# We use cgroupfs that is delegated by systemd, so we do not use systemd driver
# (unless you run another systemd in the namespace)
cgroupDriver cgroupfs
```

When the `KubeletInUserNamespace` feature gate is enabled, the kubelet ignores errors
that may happen during setting the following sysctl values on the node.

- `vm.overcommit_memory`
- `vm.panic_on_oom`
- `kernel.panic`
- `kernel.panic_on_oops`
- `kernel.keys.root_maxkeys`
- `kernel.keys.root_maxbytes`.

Within a user namespace, the kubelet also ignores any error raised from trying to open `devkmsg`.
This feature gate also allows kube-proxy to ignore an error during setting `RLIMIT_NOFILE`.

The `KubeletInUserNamespace` feature gate was introduced in Kubernetes v1.22 with alpha status.

Running kubelet in a user namespace without using this feature gate is also possible
by mounting a specially crafted proc filesystem (as done by [Sysbox](httpsgithub.comnestyboxsysbox)), but not officially supported.

# # # Configuring kube-proxy

Running kube-proxy in a user namespace requires the following configuration

```yaml
apiVersion kubeproxy.config.k8s.iov1alpha1
kind KubeProxyConfiguration
mode iptables # or userspace
conntrack
# Skip setting sysctl value net.netfilter.nf_conntrack_max
  maxPerCore 0
# Skip setting net.netfilter.nf_conntrack_tcp_timeout_established
  tcpEstablishedTimeout 0s
# Skip setting net.netfilter.nf_conntrack_tcp_timeout_close
  tcpCloseWaitTimeout 0s
```

# # Caveats

- Most of non-local volume drivers such as `nfs` and `iscsi` do not work.
  Local volumes like `local`, `hostPath`, `emptyDir`, `configMap`, `secret`, and `downwardAPI` are known to work.

- Some CNI plugins may not work. Flannel (VXLAN) is known to work.

For more on this, see the [Caveats and Future work](httpsrootlesscontaine.rscaveats) page
on the rootlesscontaine.rs website.

# #  heading seealso

- [rootlesscontaine.rs](httpsrootlesscontaine.rs)
- [Rootless Containers 2020 (KubeCon NA 2020)](httpswww.slideshare.netAkihiroSudakubecon-na-2020-containerd-rootless-containers-2020)
- [Running kind with Rootless Docker](httpskind.sigs.k8s.iodocsuserrootless)
- [Usernetes](httpsgithub.comrootless-containersusernetes)
- [Running K3s with rootless mode](httpsrancher.comdocsk3slatestenadvanced#running-k3s-with-rootless-mode-experimental)
- [KEP-2033 Kubelet-in-UserNS (aka Rootless mode)](httpsgithub.comkubernetesenhancementstreemasterkepssig-node2033-kubelet-in-userns-aka-rootless)
