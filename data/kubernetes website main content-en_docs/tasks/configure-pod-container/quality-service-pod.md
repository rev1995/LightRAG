---
title Configure Quality of Service for Pods
content_type task
weight 60
---

This page shows how to configure Pods so that they will be assigned particular
.
Kubernetes uses QoS classes to make decisions about evicting Pods when Node resources are exceeded.

When Kubernetes creates a Pod it assigns one of these QoS classes to the Pod

* [Guaranteed](docsconceptsworkloadspodspod-qos#guaranteed)
* [Burstable](docsconceptsworkloadspodspod-qos#burstable)
* [BestEffort](docsconceptsworkloadspodspod-qos#besteffort)

# #  heading prerequisites

You also need to be able to create and delete namespaces.

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace qos-example
```

# # Create a Pod that gets assigned a QoS class of Guaranteed

For a Pod to be given a QoS class of `Guaranteed`

* Every Container in the Pod must have a memory limit and a memory request.
* For every Container in the Pod, the memory limit must equal the memory request.
* Every Container in the Pod must have a CPU limit and a CPU request.
* For every Container in the Pod, the CPU limit must equal the CPU request.

These restrictions apply to init containers and app containers equally.
[Ephemeral containers](docsconceptsworkloadspodsephemeral-containers)
cannot define resources so these restrictions do not apply.

Here is a manifest for a Pod that has one Container. The Container has a memory limit and a
memory request, both equal to 200 MiB. The Container has a CPU limit and a CPU request, both equal to 700 milliCPU

 code_sample filepodsqosqos-pod.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsqosqos-pod.yaml --namespaceqos-example
```

View detailed information about the Pod

```shell
kubectl get pod qos-demo --namespaceqos-example --outputyaml
```

The output shows that Kubernetes gave the Pod a QoS class of `Guaranteed`. The output also
verifies that the Pod Container has a memory request that matches its memory limit, and it has
a CPU request that matches its CPU limit.

```yaml
spec
  containers
    ...
    resources
      limits
        cpu 700m
        memory 200Mi
      requests
        cpu 700m
        memory 200Mi
    ...
status
  qosClass Guaranteed
```

If a Container specifies its own memory limit, but does not specify a memory request, Kubernetes
automatically assigns a memory request that matches the limit. Similarly, if a Container specifies its own
CPU limit, but does not specify a CPU request, Kubernetes automatically assigns a CPU request that matches
the limit.

# # # # Clean up #clean-up-guaranteed

Delete your Pod

```shell
kubectl delete pod qos-demo --namespaceqos-example
```

# # Create a Pod that gets assigned a QoS class of Burstable

A Pod is given a QoS class of `Burstable` if

* The Pod does not meet the criteria for QoS class `Guaranteed`.
* At least one Container in the Pod has a memory or CPU request or limit.

Here is a manifest for a Pod that has one Container. The Container has a memory limit of 200 MiB
and a memory request of 100 MiB.

 code_sample filepodsqosqos-pod-2.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsqosqos-pod-2.yaml --namespaceqos-example
```

View detailed information about the Pod

```shell
kubectl get pod qos-demo-2 --namespaceqos-example --outputyaml
```

The output shows that Kubernetes gave the Pod a QoS class of `Burstable`

```yaml
spec
  containers
  - image nginx
    imagePullPolicy Always
    name qos-demo-2-ctr
    resources
      limits
        memory 200Mi
      requests
        memory 100Mi
  ...
status
  qosClass Burstable
```

# # # # Clean up #clean-up-burstable

Delete your Pod

```shell
kubectl delete pod qos-demo-2 --namespaceqos-example
```

# # Create a Pod that gets assigned a QoS class of BestEffort

For a Pod to be given a QoS class of `BestEffort`, the Containers in the Pod must not
have any memory or CPU limits or requests.

Here is a manifest for a Pod that has one Container. The Container has no memory or CPU
limits or requests

 code_sample filepodsqosqos-pod-3.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsqosqos-pod-3.yaml --namespaceqos-example
```

View detailed information about the Pod

```shell
kubectl get pod qos-demo-3 --namespaceqos-example --outputyaml
```

The output shows that Kubernetes gave the Pod a QoS class of `BestEffort`

```yaml
spec
  containers
    ...
    resources
  ...
status
  qosClass BestEffort
```

# # # # Clean up #clean-up-besteffort

Delete your Pod

```shell
kubectl delete pod qos-demo-3 --namespaceqos-example
```

# # Create a Pod that has two Containers

Here is a manifest for a Pod that has two Containers. One container specifies a memory
request of 200 MiB. The other Container does not specify any requests or limits.

 code_sample filepodsqosqos-pod-4.yaml

Notice that this Pod meets the criteria for QoS class `Burstable`. That is, it does not meet the
criteria for QoS class `Guaranteed`, and one of its Containers has a memory request.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsqosqos-pod-4.yaml --namespaceqos-example
```

View detailed information about the Pod

```shell
kubectl get pod qos-demo-4 --namespaceqos-example --outputyaml
```

The output shows that Kubernetes gave the Pod a QoS class of `Burstable`

```yaml
spec
  containers
    ...
    name qos-demo-4-ctr-1
    resources
      requests
        memory 200Mi
    ...
    name qos-demo-4-ctr-2
    resources
    ...
status
  qosClass Burstable
```

# # Retrieve the QoS class for a Pod

Rather than see all the fields, you can view just the field you need

```bash
kubectl --namespaceqos-example get pod qos-demo-4 -o jsonpath .status.qosClassn
```

```none
Burstable
```

# # Clean up

Delete your namespace

```shell
kubectl delete namespace qos-example
```

# #  heading whatsnext

# # # For app developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure Memory and CPU Quotas for a Namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace)

* [Configure a Pod Quota for a Namespace](docstasksadminister-clustermanage-resourcesquota-pod-namespace)

* [Configure Quotas for API Objects](docstasksadminister-clusterquota-api-object)

* [Control Topology Management policies on a node](docstasksadminister-clustertopology-manager)
