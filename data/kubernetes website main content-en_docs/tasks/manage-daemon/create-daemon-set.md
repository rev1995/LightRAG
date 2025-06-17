---
title Building a Basic DaemonSet
content_type task
weight 5
---

This page demonstrates how to build a basic
that runs a Pod on every node in a Kubernetes cluster.
It covers a simple use case of mounting a file from the host, logging its contents using
an [init container](docsconceptsworkloadspodsinit-containers), and utilizing a pause container.

# #  heading prerequisites

A Kubernetes cluster with at least two nodes (one control plane node and one worker node)
to demonstrate the behavior of DaemonSets.

# # Define the DaemonSet

In this task, a basic DaemonSet is created which ensures that the copy of a Pod is scheduled on every node.
The Pod will use an init container to read and log the contents of `etcmachine-id` from the host,
while the main container will be a `pause` container, which keeps the Pod running.

 code_sample fileapplicationbasic-daemonset.yaml

1. Create a DaemonSet based on the (YAML) manifest

   ```shell
   kubectl apply -f httpsk8s.ioexamplesapplicationbasic-daemonset.yaml
   ```

1. Once applied, you can verify that the DaemonSet is running a Pod on every node in the cluster

   ```shell
   kubectl get pods -o wide
   ```

   The output will list one Pod per node, similar to

   ```
   NAME                                READY   STATUS    RESTARTS   AGE    IP       NODE
   example-daemonset-xxxxx             11     Running   0          5m     x.x.x.x  node-1
   example-daemonset-yyyyy             11     Running   0          5m     x.x.x.x  node-2
   ```

1. You can inspect the contents of the logged `etcmachine-id` file by checking
   the log directory mounted from the host

   ```shell
   kubectl exec  -- cat varlogmachine-id.log
   ```

   Where `` is the name of one of your Pods.

# #  heading cleanup

To delete the DaemonSet, run this command

```shell
kubectl delete --cascadeforeground --ignore-not-found --now daemonsetsexample-daemonset
```

This simple DaemonSet example introduces key components like init containers and host path volumes,
which can be expanded upon for more advanced use cases. For more details refer to
[DaemonSet](docsconceptsworkloadscontrollersdaemonset).

# #  heading whatsnext

* See [Performing a rolling update on a DaemonSet](docstasksmanage-daemonupdate-daemon-set)
* See [Creating a DaemonSet to adopt existing DaemonSet pods](docsconceptsworkloadscontrollersdaemonset)
