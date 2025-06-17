---
reviewers
- nelvadas
title Limit Ranges
api_metadata
- apiVersion v1
  kind LimitRange
content_type concept
weight 10
---

By default, containers run with unbounded
[compute resources](docsconceptsconfigurationmanage-resources-containers) on a Kubernetes cluster.
Using  Kubernetes [resource quotas](docsconceptspolicyresource-quotas),
administrators (also termed _cluster operators_) can restrict consumption and creation
of cluster resources (such as CPU time, memory, and persistent storage) within a specified
.
Within a namespace, a  can consume as much CPU and memory as is allowed by the ResourceQuotas that apply to that namespace.
As a cluster operator, or as a namespace-level administrator, you might also be concerned
about making sure that a single object cannot monopolize all available resources within a namespace.

A LimitRange is a policy to constrain the resource allocations (limits and requests) that you can specify for
each applicable object kind (such as Pod or ) in a namespace.

A _LimitRange_ provides constraints that can

- Enforce minimum and maximum compute resources usage per Pod or Container in a namespace.
- Enforce minimum and maximum storage request per
   in a namespace.
- Enforce a ratio between request and limit for a resource in a namespace.
- Set default requestlimit for compute resources in a namespace and automatically
  inject them to Containers at runtime.

Kubernetes constrains resource allocations to Pods in a particular namespace
whenever there is at least one LimitRange object in that namespace.

The name of a LimitRange object must be a valid
[DNS subdomain name](docsconceptsoverviewworking-with-objectsnames#dns-subdomain-names).

# # Constraints on resource limits and requests

- The administrator creates a LimitRange in a namespace.
- Users create (or try to create) objects in that namespace, such as Pods or
  PersistentVolumeClaims.
- First, the LimitRange admission controller applies default request and limit values
  for all Pods (and their containers) that do not set compute resource requirements.
- Second, the LimitRange tracks usage to ensure it does not exceed resource minimum,
  maximum and ratio defined in any LimitRange present in the namespace.
- If you attempt to create or update an object (Pod or PersistentVolumeClaim) that violates
  a LimitRange constraint, your request to the API server will fail with anHTTP status
  code `403 Forbidden` and a message explaining the constraint that has been violated.
- If you add a LimitRange in a namespace that applies to compute-related resources
  such as `cpu` and `memory`, you must specify requests or limits for those values.
  Otherwise, the system may reject Pod creation.
- LimitRange validations occur only at Pod admission stage, not on running Pods.
  If you add or modify a LimitRange, the Pods that already exist in that namespace
  continue unchanged.
- If two or more LimitRange objects exist in the namespace, it is not deterministic
  which default value will be applied.

# # LimitRange and admission checks for Pods

A LimitRange does **not** check the consistency of the default values it applies.
This means that a default value for the _limit_ that is set by LimitRange may be
less than the _request_ value specified for the container in the spec that a client
submits to the API server. If that happens, the final Pod will not be schedulable.

For example, you define a LimitRange with below manifest

The following examples operate within the default namespace of your cluster, as the namespace
parameter is undefined and the LimitRange scope is limited to the namespace level.
This implies that any references or operations within these examples will interact
with elements within the default namespace of your cluster. You can override the
operating namespace by configuring namespace in the `metadata.namespace` field.

 code_sample fileconceptspolicylimit-rangeproblematic-limit-range.yaml

along with a Pod that declares a CPU resource request of `700m`, but not a limit

 code_sample fileconceptspolicylimit-rangeexample-conflict-with-limitrange-cpu.yaml

then that Pod will not be scheduled, failing with an error similar to
```
Pod example-conflict-with-limitrange-cpu is invalid spec.containers[0].resources.requests Invalid value 700m must be less than or equal to cpu limit
```

If you set both `request` and `limit`, then that new Pod will be scheduled successfully
even with the same LimitRange in place

 code_sample fileconceptspolicylimit-rangeexample-no-conflict-with-limitrange-cpu.yaml

# # Example resource constraints

Examples of policies that could be created using LimitRange are

- In a 2 node cluster with a capacity of 8 GiB RAM and 16 cores, constrain Pods in a
  namespace to request 100m of CPU with a max limit of 500m for CPU and request 200Mi
  for Memory with a max limit of 600Mi for Memory.
- Define default CPU limit and request to 150m and memory default request to 300Mi for
  Containers started with no cpu and memory requests in their specs.

In the case where the total limits of the namespace is less than the sum of the limits
of the PodsContainers, there may be contention for resources. In this case, the
Containers or Pods will not be created.

Neither contention nor changes to a LimitRange will affect already created resources.

# #  heading whatsnext

For examples on using limits, see

- [how to configure minimum and maximum CPU constraints per namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace).
- [how to configure minimum and maximum Memory constraints per namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace).
- [how to configure default CPU Requests and Limits per namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace).
- [how to configure default Memory Requests and Limits per namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace).
- [how to configure minimum and maximum Storage consumption per namespace](docstasksadminister-clusterlimit-storage-consumption#limitrange-to-limit-requests-for-storage).
- a [detailed example on configuring quota per namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace).

Refer to the [LimitRanger design document](httpsgit.k8s.iodesign-proposals-archiveresource-managementadmission_control_limit_range.md)
for context and historical information.