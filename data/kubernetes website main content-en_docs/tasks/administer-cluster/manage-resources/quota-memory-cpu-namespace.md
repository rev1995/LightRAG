---
title Configure Memory and CPU Quotas for a Namespace
content_type task
weight 50
description -
  Define overall memory and CPU resource limits for a namespace.
---

This page shows how to set quotas for the total amount memory and CPU that
can be used by all Pods running in a .
You specify quotas in a
[ResourceQuota](docsreferencekubernetes-apipolicy-resourcesresource-quota-v1)
object.

# #  heading prerequisites

You must have access to create namespaces in your cluster.

Each node in your cluster must have at least 1 GiB of memory.

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace quota-mem-cpu-example
```

# # Create a ResourceQuota

Here is a manifest for an example ResourceQuota

 code_sample fileadminresourcequota-mem-cpu.yaml

Create the ResourceQuota

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-mem-cpu.yaml --namespacequota-mem-cpu-example
```

View detailed information about the ResourceQuota

```shell
kubectl get resourcequota mem-cpu-demo --namespacequota-mem-cpu-example --outputyaml
```

The ResourceQuota places these requirements on the quota-mem-cpu-example namespace

* For every Pod in the namespace, each container must have a memory request, memory limit, cpu request, and cpu limit.
* The memory request total for all Pods in that namespace must not exceed 1 GiB.
* The memory limit total for all Pods in that namespace must not exceed 2 GiB.
* The CPU request total for all Pods in that namespace must not exceed 1 cpu.
* The CPU limit total for all Pods in that namespace must not exceed 2 cpu.

See [meaning of CPU](docsconceptsconfigurationmanage-resources-containers#meaning-of-cpu)
to learn what Kubernetes means by 1 CPU.

# # Create a Pod

Here is a manifest for an example Pod

 code_sample fileadminresourcequota-mem-cpu-pod.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-mem-cpu-pod.yaml --namespacequota-mem-cpu-example
```

Verify that the Pod is running and that its (only) container is healthy

```shell
kubectl get pod quota-mem-cpu-demo --namespacequota-mem-cpu-example
```

Once again, view detailed information about the ResourceQuota

```shell
kubectl get resourcequota mem-cpu-demo --namespacequota-mem-cpu-example --outputyaml
```

The output shows the quota along with how much of the quota has been used.
You can see that the memory and CPU requests and limits for your Pod do not
exceed the quota.

```
status
  hard
    limits.cpu 2
    limits.memory 2Gi
    requests.cpu 1
    requests.memory 1Gi
  used
    limits.cpu 800m
    limits.memory 800Mi
    requests.cpu 400m
    requests.memory 600Mi
```

If you have the `jq` tool, you can also query (using [JSONPath](docsreferencekubectljsonpath))
for just the `used` values, **and** pretty-print that that of the output. For example

```shell
kubectl get resourcequota mem-cpu-demo --namespacequota-mem-cpu-example -o jsonpath .status.used   jq .
```

# # Attempt to create a second Pod

Here is a manifest for a second Pod

 code_sample fileadminresourcequota-mem-cpu-pod-2.yaml

In the manifest, you can see that the Pod has a memory request of 700 MiB.
Notice that the sum of the used memory request and this new memory
request exceeds the memory request quota 600 MiB  700 MiB  1 GiB.

Attempt to create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-mem-cpu-pod-2.yaml --namespacequota-mem-cpu-example
```

The second Pod does not get created. The output shows that creating the second Pod
would cause the memory request total to exceed the memory request quota.

```
Error from server (Forbidden) error when creating examplesadminresourcequota-mem-cpu-pod-2.yaml
pods quota-mem-cpu-demo-2 is forbidden exceeded quota mem-cpu-demo,
requested requests.memory700Mi,used requests.memory600Mi, limited requests.memory1Gi
```

# # Discussion

As you have seen in this exercise, you can use a ResourceQuota to restrict
the memory request total for all Pods running in a namespace.
You can also restrict the totals for memory limit, cpu request, and cpu limit.

Instead of managing total resource use within a namespace, you might want to restrict
individual Pods, or the containers in those Pods. To achieve that kind of limiting, use a
[LimitRange](docsconceptspolicylimit-range).

# # Clean up

Delete your namespace

```shell
kubectl delete namespace quota-mem-cpu-example
```

# #  heading whatsnext

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure a Pod Quota for a Namespace](docstasksadminister-clustermanage-resourcesquota-pod-namespace)

* [Configure Quotas for API Objects](docstasksadminister-clusterquota-api-object)

# # # For app developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

* [Assign Pod-level CPU and memory resources](docstasksconfigure-pod-containerassign-pod-level-resources)

* [Configure Quality of Service for Pods](docstasksconfigure-pod-containerquality-service-pod)
