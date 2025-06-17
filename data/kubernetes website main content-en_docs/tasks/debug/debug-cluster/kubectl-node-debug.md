---
title Debugging Kubernetes Nodes With Kubectl
content_type task
min-kubernetes-server-version 1.20
---

This page shows how to debug a [node](docsconceptsarchitecturenodes)
running on the Kubernetes cluster using `kubectl debug` command.

# #  heading prerequisites

You need to have permission to create Pods and to assign those new Pods to arbitrary nodes.
You also need to be authorized to create Pods that access filesystems from the host.

# # Debugging a Node using `kubectl debug node`

Use the `kubectl debug node` command to deploy a Pod to a Node that you want to troubleshoot.
This command is helpful in scenarios where you cant access your Node by using an SSH connection.
When the Pod is created, the Pod opens an interactive shell on the Node.
To create an interactive shell on a Node named mynode, run

```shell
kubectl debug nodemynode -it --imageubuntu
```

```console
Creating debugging pod node-debugger-mynode-pdx84 with container debugger on node mynode.
If you dont see a command prompt, try pressing enter.
rootmynode#
```

The debug command helps to gather information and troubleshoot issues. Commands
that you might use include `ip`, `ifconfig`, `nc`, `ping`, and `ps` and so on. You can also
install other tools, such as `mtr`, `tcpdump`, and `curl`, from the respective package manager.

The debug commands may differ based on the image the debugging pod is using and
these commands might need to be installed.

The debugging Pod can access the root filesystem of the Node, mounted at `host` in the Pod.
If you run your kubelet in a filesystem namespace,
the debugging Pod sees the root for that namespace, not for the entire node. For a typical Linux node,
you can look at the following paths to find relevant logs

`hostvarlogkubelet.log`
 Logs from the `kubelet`, responsible for running containers on the node.

`hostvarlogkube-proxy.log`
 Logs from `kube-proxy`, which is responsible for directing traffic to Service endpoints.

`hostvarlogcontainerd.log`
 Logs from the `containerd` process running on the node.

`hostvarlogsyslog`
 Shows general messages and information regarding the system.

`hostvarlogkern.log`
 Shows kernel logs.

When creating a debugging session on a Node, keep in mind that

* `kubectl debug` automatically generates the name of the new pod, based on
  the name of the node.
* The root filesystem of the Node will be mounted at `host`.
* Although the container runs in the host IPC, Network, and PID namespaces,
  the pod isnt privileged. This means that reading some process information might fail
  because access to that information is restricted to superusers. For example, `chroot host` will fail.
  If you need a privileged pod, create it manually or use the `--profilesysadmin` flag.
* By applying [Debugging Profiles](docstasksdebugdebug-applicationdebug-running-pod#debugging-profiles), you can set specific properties such as [securityContext](docstasksconfigure-pod-containersecurity-context) to a debugging Pod.

# #  heading cleanup

When you finish using the debugging Pod, delete it

```shell
kubectl get pods
```

```none
NAME                          READY   STATUS       RESTARTS   AGE
node-debugger-mynode-pdx84    01     Completed    0          8m1s
```

```shell
# Change the pod name accordingly
kubectl delete pod node-debugger-mynode-pdx84 --now
```

```none
pod node-debugger-mynode-pdx84 deleted
```

The `kubectl debug node` command wont work if the Node is down (disconnected
from the network, or kubelet dies and wont restart, etc.).
Check [debugging a downunreachable node ](docstasksdebugdebug-cluster#example-debugging-a-down-unreachable-node)
in that case.
