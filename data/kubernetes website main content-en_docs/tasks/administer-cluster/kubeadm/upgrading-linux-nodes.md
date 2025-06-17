---
title Upgrading Linux nodes
content_type task
weight 40
---

This page explains how to upgrade a Linux Worker Nodes created with kubeadm.

# #  heading prerequisites

* Familiarize yourself with [the process for upgrading the rest of your kubeadm
cluster](docstasksadminister-clusterkubeadmkubeadm-upgrade). You will want to
upgrade the control plane nodes before upgrading your Linux Worker nodes.

# # Changing the package repository

If youre using the community-owned package repositories (`pkgs.k8s.io`), you need to
enable the package repository for the desired Kubernetes minor release. This is explained in
[Changing the Kubernetes package repository](docstasksadminister-clusterkubeadmchange-package-repository)
document.

 legacy-repos-deprecation

# # Upgrading worker nodes

# # # Upgrade kubeadm

Upgrade kubeadm

 tab nameUbuntu, Debian or HypriotOS
```shell
# replace x in .x-* with the latest patch version
sudo apt-mark unhold kubeadm
sudo apt-get update  sudo apt-get install -y kubeadm.x-*
sudo apt-mark hold kubeadm
```
 tab
 tab nameCentOS, RHEL or Fedora
```shell
# replace x in .x-* with the latest patch version
sudo yum install -y kubeadm-.x-* --disableexcludeskubernetes
```
 tab

# # # Call kubeadm upgrade

For worker nodes this upgrades the local kubelet configuration

```shell
sudo kubeadm upgrade node
```

# # # Drain the node

Prepare the node for maintenance by marking it unschedulable and evicting the workloads

```shell
# execute this command on a control plane node
# replace  with the name of your node you are draining
kubectl drain  --ignore-daemonsets
```

# # # Upgrade kubelet and kubectl

1. Upgrade the kubelet and kubectl

    tab nameUbuntu, Debian or HypriotOS
   ```shell
   # replace x in .x-* with the latest patch version
   sudo apt-mark unhold kubelet kubectl
   sudo apt-get update  sudo apt-get install -y kubelet.x-* kubectl.x-*
   sudo apt-mark hold kubelet kubectl
   ```
    tab
    tab nameCentOS, RHEL or Fedora
   ```shell
   # replace x in .x-* with the latest patch version
   sudo yum install -y kubelet-.x-* kubectl-.x-* --disableexcludeskubernetes
   ```
    tab

1. Restart the kubelet

   ```shell
   sudo systemctl daemon-reload
   sudo systemctl restart kubelet
   ```

# # # Uncordon the node

Bring the node back online by marking it schedulable

```shell
# execute this command on a control plane node
# replace  with the name of your node
kubectl uncordon
```

# #  heading whatsnext

* See how to [Upgrade Windows nodes](docstasksadminister-clusterkubeadmupgrading-windows-nodes).