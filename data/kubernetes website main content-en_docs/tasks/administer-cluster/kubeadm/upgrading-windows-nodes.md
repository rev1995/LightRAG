---
title Upgrading Windows nodes
min-kubernetes-server-version 1.17
content_type task
weight 41
---

This page explains how to upgrade a Windows node created with kubeadm.

# #  heading prerequisites

* Familiarize yourself with [the process for upgrading the rest of your kubeadm
cluster](docstasksadminister-clusterkubeadmkubeadm-upgrade). You will want to
upgrade the control plane nodes before upgrading your Windows nodes.

# # Upgrading worker nodes

# # # Upgrade kubeadm

1.  From the Windows node, upgrade kubeadm

    ```powershell
    # replace  with your desired version
    curl.exe -Lo   httpsdl.k8s.iovbinwindowsamd64kubeadm.exe
    ```

# # # Drain the node

1.  From a machine with access to the Kubernetes API,
    prepare the node for maintenance by marking it unschedulable and evicting the workloads

    ```shell
    # replace  with the name of your node you are draining
    kubectl drain  --ignore-daemonsets
    ```

    You should see output similar to this

    ```
    nodeip-172-31-85-18 cordoned
    nodeip-172-31-85-18 drained
    ```

# # # Upgrade the kubelet configuration

1.  From the Windows node, call the following command to sync new kubelet configuration

    ```powershell
    kubeadm upgrade node
    ```

# # # Upgrade kubelet and kube-proxy

1.  From the Windows node, upgrade and restart the kubelet

    ```powershell
    stop-service kubelet
    curl.exe -Lo  httpsdl.k8s.iovbinwindowsamd64kubelet.exe
    restart-service kubelet
    ```

2. From the Windows node, upgrade and restart the kube-proxy.

    ```powershell
    stop-service kube-proxy
    curl.exe -Lo  httpsdl.k8s.iovbinwindowsamd64kube-proxy.exe
    restart-service kube-proxy
    ```

If you are running kube-proxy in a HostProcess container within a Pod, and not as a Windows Service,
you can upgrade kube-proxy by applying a newer version of your kube-proxy manifests.

# # # Uncordon the node

1.  From a machine with access to the Kubernetes API,
bring the node back online by marking it schedulable

    ```shell
    # replace  with the name of your node
    kubectl uncordon
    ```
 ##  heading whatsnext

* See how to [Upgrade Linux nodes](docstasksadminister-clusterkubeadmupgrading-linux-nodes).
