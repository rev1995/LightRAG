---
title Configure Default Memory Requests and Limits for a Namespace
content_type task
weight 10
description -
  Define a default memory resource limit for a namespace, so that every new Pod
  in that namespace has a memory resource limit configured.
---

This page shows how to configure default memory requests and limits for a
.

A Kubernetes cluster can be divided into namespaces. Once you have a namespace that
has a default memory
[limit](docsconceptsconfigurationmanage-resources-containers#requests-and-limits),
and you then try to create a Pod with a container that does not specify its own memory
limit, then the
 assigns the default
memory limit to that container.

Kubernetes assigns a default memory request under certain conditions that are explained later in this topic.

# #  heading prerequisites

You must have access to create namespaces in your cluster.

Each node in your cluster must have at least 2 GiB of memory.

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace default-mem-example
```

# # Create a LimitRange and a Pod

Heres a manifest for an example .
The manifest specifies a default memory
request and a default memory limit.

 code_sample fileadminresourcememory-defaults.yaml

Create the LimitRange in the default-mem-example namespace

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcememory-defaults.yaml --namespacedefault-mem-example
```

Now if you create a Pod in the default-mem-example namespace, and any container
within that Pod does not specify its own values for memory request and memory limit,
then the
applies default values a memory request of 256MiB and a memory limit of 512MiB.

Heres an example manifest for a Pod that has one container. The container
does not specify a memory request and limit.

 code_sample fileadminresourcememory-defaults-pod.yaml

Create the Pod.

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcememory-defaults-pod.yaml --namespacedefault-mem-example
```

View detailed information about the Pod

```shell
kubectl get pod default-mem-demo --outputyaml --namespacedefault-mem-example
```

The output shows that the Pods container has a memory request of 256 MiB and
a memory limit of 512 MiB. These are the default values specified by the LimitRange.

```shell
containers
- image nginx
  imagePullPolicy Always
  name default-mem-demo-ctr
  resources
    limits
      memory 512Mi
    requests
      memory 256Mi
```

Delete your Pod

```shell
kubectl delete pod default-mem-demo --namespacedefault-mem-example
```

# # What if you specify a containers limit, but not its request

Heres a manifest for a Pod that has one container. The container
specifies a memory limit, but not a request

 code_sample fileadminresourcememory-defaults-pod-2.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcememory-defaults-pod-2.yaml --namespacedefault-mem-example
```

View detailed information about the Pod

```shell
kubectl get pod default-mem-demo-2 --outputyaml --namespacedefault-mem-example
```

The output shows that the containers memory request is set to match its memory limit.
Notice that the container was not assigned the default memory request value of 256Mi.

```
resources
  limits
    memory 1Gi
  requests
    memory 1Gi
```

# # What if you specify a containers request, but not its limit

Heres a manifest for a Pod that has one container. The container
specifies a memory request, but not a limit

 code_sample fileadminresourcememory-defaults-pod-3.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcememory-defaults-pod-3.yaml --namespacedefault-mem-example
```

View the Pods specification

```shell
kubectl get pod default-mem-demo-3 --outputyaml --namespacedefault-mem-example
```

The output shows that the containers memory request is set to the value specified in the
containers manifest. The container is limited to use no more than 512MiB of
memory, which matches the default memory limit for the namespace.

```
resources
  limits
    memory 512Mi
  requests
    memory 128Mi
```

A `LimitRange` does **not** check the consistency of the default values it applies. This means that a default value for the _limit_ that is set by `LimitRange` may be less than the _request_ value specified for the container in the spec that a client submits to the API server. If that happens, the final Pod will not be scheduleable.
See [Constraints on resource limits and requests](docsconceptspolicylimit-range#constraints-on-resource-limits-and-requests) for more details.

# # Motivation for default memory limits and requests

If your namespace has a memory
configured,
it is helpful to have a default value in place for memory limit.
Here are three of the restrictions that a resource quota imposes on a namespace

* For every Pod that runs in the namespace, the Pod and each of its containers must have a memory limit.
  (If you specify a memory limit for every container in a Pod, Kubernetes can infer the Pod-level memory
  limit by adding up the limits for its containers).
* Memory limits apply a resource reservation on the node where the Pod in question is scheduled.
  The total amount of memory reserved for all Pods in the namespace must not exceed a specified limit.
* The total amount of memory actually used by all Pods in the namespace must also not exceed a specified limit.

When you add a LimitRange

If any Pod in that namespace that includes a container does not specify its own memory limit,
the control plane applies the default memory limit to that container, and the Pod can be
allowed to run in a namespace that is restricted by a memory ResourceQuota.

# # Clean up

Delete your namespace

```shell
kubectl delete namespace default-mem-example
```

# #  heading whatsnext

# # # For cluster administrators

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

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
