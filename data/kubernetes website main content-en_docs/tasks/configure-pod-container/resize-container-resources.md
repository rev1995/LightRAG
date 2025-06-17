---
title Resize CPU and Memory Resources assigned to Containers
content_type task
weight 30
min-kubernetes-server-version 1.33
---

This page explains how to change the CPU and memory resource requests and limits
assigned to a container *without recreating the Pod*.

Traditionally, changing a Pods resource requirements necessitated deleting the existing Pod
and creating a replacement, often managed by a [workload controller](docsconceptsworkloadscontrollers).
In-place Pod Resize allows changing the CPUmemory allocation of container(s) within a running Pod
while potentially avoiding application disruption.

**Key Concepts**

* **Desired Resources** A containers `spec.containers[*].resources` represent
  the *desired* resources for the container, and are mutable for CPU and memory.
* **Actual Resources** The `status.containerStatuses[*].resources` field
  reflects the resources *currently configured* for a running container.
  For containers that havent started or were restarted,
  it reflects the resources allocated upon their next start.
* **Triggering a Resize** You can request a resize by updating the desired `requests`
  and `limits` in the Pods specification.
  This is typically done using `kubectl patch`, `kubectl apply`, or `kubectl edit`
  targeting the Pods `resize` subresource.
  When the desired resources dont match the allocated resources,
  the Kubelet will attempt to resize the container.
* **Allocated Resources (Advanced)**
  The `status.containerStatuses[*].allocatedResources` field tracks resource values
  confirmed by the Kubelet, primarily used for internal scheduling logic.
  For most monitoring and validation purposes, focus on `status.containerStatuses[*].resources`.

If a node has pods with a pending or incomplete resize (see [Pod Resize Status](#pod-resize-status) below),
the  uses
the *maximum* of a containers desired requests, allocated requests,
and actual requests from the status when making scheduling decisions.

# #  heading prerequisites

The `InPlacePodVerticalScaling` [feature gate](docsreferencecommand-line-tools-referencefeature-gates)
must be enabled
for your control plane and for all nodes in your cluster.

The `kubectl` client version must be at least v1.32 to use the `--subresourceresize` flag.

# # Pod resize status

The Kubelet updates the Pods status conditions to indicate the state of a resize request

* `type PodResizePending` The Kubelet cannot immediately grant the request.
  The `message` field provides an explanation of why.
    * `reason Infeasible` The requested resize is impossible on the current node
      (for example, requesting more resources than the node has).
    * `reason Deferred` The requested resize is currently not possible,
      but might become feasible later (for example if another pod is removed).
      The Kubelet will retry the resize.
* `type PodResizeInProgress` The Kubelet has accepted the resize and allocated resources,
  but the changes are still being applied.
  This is usually brief but might take longer depending on the resource type and runtime behavior.
  Any errors during actuation are reported in the `message` field (along with `reason Error`).

# # Container resize policies

You can control whether a container should be restarted when resizing
by setting `resizePolicy` in the container specification.
This allows fine-grained control based on resource type (CPU or memory).

```yaml
    resizePolicy
    - resourceName cpu
      restartPolicy NotRequired
    - resourceName memory
      restartPolicy RestartContainer
```

* `NotRequired` (Default) Apply the resource change to the running container without restarting it.
* `RestartContainer` Restart the container to apply the new resource values.
  This is often necessary for memory changes because many applications
  and runtimes cannot adjust their memory allocation dynamically.

If `resizePolicy[*].restartPolicy` is not specified for a resource, it defaults to `NotRequired`.

If a Pods overall `restartPolicy` is `Never`, then any container `resizePolicy` must be `NotRequired` for all resources.
You cannot configure a resize policy that would require a restart in such Pods.

**Example Scenario**

Consider a container configured with `restartPolicy NotRequired` for CPU and `restartPolicy RestartContainer` for memory.
* If only CPU resources are changed, the container is resized in-place.
* If only memory resources are changed, the container is restarted.
* If *both* CPU and memory resources are changed simultaneously, the container is restarted (due to the memory policy).

# # Limitations

For Kubernetes , resizing pod resources in-place has the following limitations

* **Resource Types** Only CPU and memory resources can be resized.
* **Memory Decrease** Memory limits _cannot be decreased_ unless the `resizePolicy` for memory is `RestartContainer`.
  Memory requests can generally be decreased.
* **QoS Class** The Pods original [Quality of Service (QoS) class](docsconceptsworkloadspodspod-qos)
  (Guaranteed, Burstable, or BestEffort) is determined at creation and **cannot** be changed by a resize.
  The resized resource values must still adhere to the rules of the original QoS class
    * *Guaranteed* Requests must continue to equal limits for both CPU and memory after resizing.
    * *Burstable* Requests and limits cannot become equal for *both* CPU and memory simultaneously
      (as this would change it to Guaranteed).
    * *BestEffort* Resource requirements (`requests` or `limits`) cannot be added
      (as this would change it to Burstable or Guaranteed).
* **Container Types** Non-restartable  and
   cannot be resized.
  [Sidecar containers](docsconceptsworkloadspodssidecar-containers) can be resized.
* **Resource Removal** Resource requests and limits cannot be entirely removed once set
  they can only be changed to different values.
* **Operating System** Windows pods do not support in-place resize.
* **Node Policies** Pods managed by [static CPU or Memory manager policies](docstasksadminister-clustercpu-management-policies)
  cannot be resized in-place.
* **Swap** Pods utilizing [swap memory](docsconceptsarchitecturenodes#swap-memory) cannot resize memory requests
  unless the `resizePolicy` for memory is `RestartContainer`.

These restrictions might be relaxed in future Kubernetes versions.

# # Example 1 Resizing CPU without restart

First, create a Pod designed for in-place CPU resize and restart-required memory resize.

 code_sample filepodsresourcepod-resize.yaml

Create the pod

```shell
kubectl create -f pod-resize.yaml
```

This pod starts in the Guaranteed QoS class. Verify its initial state

```shell
# Wait a moment for the pod to be running
kubectl get pod resize-demo --outputyaml
```

Observe the `spec.containers[0].resources` and `status.containerStatuses[0].resources`.
They should match the manifest (700m CPU, 200Mi memory). Note the `status.containerStatuses[0].restartCount` (should be 0).

Now, increase the CPU request and limit to `800m`. You use `kubectl patch` with the `--subresource resize` command line argument.

```shell
kubectl patch pod resize-demo --subresource resize --patch
  speccontainers[namepause, resourcesrequestscpu800m, limitscpu800m]

# Alternative methods
# kubectl -n qos-example edit pod resize-demo --subresource resize
# kubectl -n qos-example apply -f  --subresource resize
```

The `--subresource resize` command line argument requires `kubectl` client version v1.32.0 or later.
Older versions will report an `invalid subresource` error.

Check the pod status again after patching

```shell
kubectl get pod resize-demo --outputyaml --namespaceqos-example
```

You should see
* `spec.containers[0].resources` now shows `cpu 800m`.
* `status.containerStatuses[0].resources` also shows `cpu 800m`, indicating the resize was successful on the node.
* `status.containerStatuses[0].restartCount` remains `0`, because the CPU `resizePolicy` was `NotRequired`.

# # Example 2 Resizing memory with restart

Now, resize the memory for the *same* pod by increasing it to `300Mi`.
Since the memory `resizePolicy` is `RestartContainer`, the container is expected to restart.

```shell
kubectl patch pod resize-demo --subresource resize --patch
  speccontainers[namepause, resourcesrequestsmemory300Mi, limitsmemory300Mi]
```

Check the pod status shortly after patching

```shell
kubectl get pod resize-demo --outputyaml
```

You should now observe
* `spec.containers[0].resources` shows `memory 300Mi`.
* `status.containerStatuses[0].resources` also shows `memory 300Mi`.
* `status.containerStatuses[0].restartCount` has increased to `1` (or more, if restarts happened previously),
  indicating the container was restarted to apply the memory change.

# # Troubleshooting Infeasible resize request

Next, try requesting an unreasonable amount of CPU, such as 1000 full cores (written as `1000` instead of `1000m` for millicores), which likely exceeds node capacity.

```shell
# Attempt to patch with an excessively large CPU request
kubectl patch pod resize-demo --subresource resize --patch
  speccontainers[namepause, resourcesrequestscpu1000, limitscpu1000]
```

Query the Pods details

```shell
kubectl get pod resize-demo --outputyaml
```

Youll see changes indicating the problem

* The `spec.containers[0].resources` reflects the *desired* state (`cpu 1000`).
* A condition with `type PodResizePending` and `reason Infeasible` was added to the Pod.
* The conditions `message` will explain why (`Node didnt have enough capacity cpu, requested 800000, capacity ...`)
* Crucially, `status.containerStatuses[0].resources` will *still show the previous values* (`cpu 800m`, `memory 300Mi`),
  because the infeasible resize was not applied by the Kubelet.
* The `restartCount` will not have changed due to this failed attempt.

To fix this, you would need to patch the pod again with feasible resource values.

# # Clean up

Delete the pod

```shell
kubectl delete pod resize-demo
```

# #  heading whatsnext

# # # For application developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

* [Assign Pod-level CPU and memory resources](docstasksconfigure-pod-containerassign-pod-level-resources)

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure Memory and CPU Quotas for a Namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace)
