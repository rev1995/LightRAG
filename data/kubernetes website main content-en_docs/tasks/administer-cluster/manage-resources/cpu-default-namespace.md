---
title Configure Default CPU Requests and Limits for a Namespace
content_type task
weight 20
description -
  Define a default CPU resource limits for a namespace, so that every new Pod
  in that namespace has a CPU resource limit configured.
---

This page shows how to configure default CPU requests and limits for a
.

A Kubernetes cluster can be divided into namespaces. If you create a Pod within a
namespace that has a default CPU
[limit](docsconceptsconfigurationmanage-resources-containers#requests-and-limits), and any container in that Pod does not specify
its own CPU limit, then the
 assigns the default
CPU limit to that container.

Kubernetes assigns a default CPU
[request](docsconceptsconfigurationmanage-resources-containers#requests-and-limits),
but only under certain conditions that are explained later in this page.

# #  heading prerequisites

You must have access to create namespaces in your cluster.

If youre not already familiar with what Kubernetes means by 1.0 CPU,
read [meaning of CPU](docsconceptsconfigurationmanage-resources-containers#meaning-of-cpu).

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace default-cpu-example
```

# # Create a LimitRange and a Pod

Heres a manifest for an example .
The manifest specifies a default CPU request and a default CPU limit.

 code_sample fileadminresourcecpu-defaults.yaml

Create the LimitRange in the default-cpu-example namespace

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcecpu-defaults.yaml --namespacedefault-cpu-example
```

Now if you create a Pod in the default-cpu-example namespace, and any container
in that Pod does not specify its own values for CPU request and CPU limit,
then the control plane applies default values a CPU request of 0.5 and a default
CPU limit of 1.

Heres a manifest for a Pod that has one container. The container
does not specify a CPU request and limit.

 code_sample fileadminresourcecpu-defaults-pod.yaml

Create the Pod.

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcecpu-defaults-pod.yaml --namespacedefault-cpu-example
```

View the Pods specification

```shell
kubectl get pod default-cpu-demo --outputyaml --namespacedefault-cpu-example
```

The output shows that the Pods only container has a CPU request of 500m `cpu`
(which you can read as 500 millicpu), and a CPU limit of 1 `cpu`.
These are the default values specified by the LimitRange.

```shell
containers
- image nginx
  imagePullPolicy Always
  name default-cpu-demo-ctr
  resources
    limits
      cpu 1
    requests
      cpu 500m
```

# # What if you specify a containers limit, but not its request

Heres a manifest for a Pod that has one container. The container
specifies a CPU limit, but not a request

 code_sample fileadminresourcecpu-defaults-pod-2.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcecpu-defaults-pod-2.yaml --namespacedefault-cpu-example
```

View the [specification](docsconceptsoverviewworking-with-objects#object-spec-and-status)
of the Pod that you created

```
kubectl get pod default-cpu-demo-2 --outputyaml --namespacedefault-cpu-example
```

The output shows that the containers CPU request is set to match its CPU limit.
Notice that the container was not assigned the default CPU request value of 0.5 `cpu`

```
resources
  limits
    cpu 1
  requests
    cpu 1
```

# # What if you specify a containers request, but not its limit

Heres an example manifest for a Pod that has one container. The container
specifies a CPU request, but not a limit

 code_sample fileadminresourcecpu-defaults-pod-3.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcecpu-defaults-pod-3.yaml --namespacedefault-cpu-example
```

View the specification of the Pod that you created

```
kubectl get pod default-cpu-demo-3 --outputyaml --namespacedefault-cpu-example
```

The output shows that the containers CPU request is set to the value you specified at
the time you created the Pod (in other words it matches the manifest).
However, the same containers CPU limit is set to 1 `cpu`, which is the default CPU limit
for that namespace.

```
resources
  limits
    cpu 1
  requests
    cpu 750m
```

# # Motivation for default CPU limits and requests

If your namespace has a CPU
configured,
it is helpful to have a default value in place for CPU limit.
Here are two of the restrictions that a CPU resource quota imposes on a namespace

* For every Pod that runs in the namespace, each of its containers must have a CPU limit.
* CPU limits apply a resource reservation on the node where the Pod in question is scheduled.
  The total amount of CPU that is reserved for use by all Pods in the namespace must not
  exceed a specified limit.

When you add a LimitRange

If any Pod in that namespace that includes a container does not specify its own CPU limit,
the control plane applies the default CPU limit to that container, and the Pod can be
allowed to run in a namespace that is restricted by a CPU ResourceQuota.

# # Clean up

Delete your namespace

```shell
kubectl delete namespace default-cpu-example
```

# #  heading whatsnext

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure Memory and CPU Quotas for a Namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace)

* [Configure a Pod Quota for a Namespace](docstasksadminister-clustermanage-resourcesquota-pod-namespace)

* [Configure Quotas for API Objects](docstasksadminister-clusterquota-api-object)

# # # For app developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

* [Assign Pod-level CPU and memory resources](docstasksconfigure-pod-containerassign-pod-level-resources)

* [Configure Quality of Service for Pods](docstasksconfigure-pod-containerquality-service-pod)
