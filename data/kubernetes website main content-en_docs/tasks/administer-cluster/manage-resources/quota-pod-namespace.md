---
title Configure a Pod Quota for a Namespace
content_type task
weight 60
description -
  Restrict how many Pods you can create within a namespace.
---

This page shows how to set a quota for the total number of Pods that can run
in a . You specify quotas in a
[ResourceQuota](docsreferencekubernetes-apipolicy-resourcesresource-quota-v1)
object.

# #  heading prerequisites

You must have access to create namespaces in your cluster.

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace quota-pod-example
```

# # Create a ResourceQuota

Here is an example manifest for a ResourceQuota

 code_sample fileadminresourcequota-pod.yaml

Create the ResourceQuota

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-pod.yaml --namespacequota-pod-example
```

View detailed information about the ResourceQuota

```shell
kubectl get resourcequota pod-demo --namespacequota-pod-example --outputyaml
```

The output shows that the namespace has a quota of two Pods, and that currently there are
no Pods that is, none of the quota is used.

```yaml
spec
  hard
    pods 2
status
  hard
    pods 2
  used
    pods 0
```

Here is an example manifest for a

 code_sample fileadminresourcequota-pod-deployment.yaml

In that manifest, `replicas 3` tells Kubernetes to attempt to create three new Pods, all
running the same application.

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-pod-deployment.yaml --namespacequota-pod-example
```

View detailed information about the Deployment

```shell
kubectl get deployment pod-quota-demo --namespacequota-pod-example --outputyaml
```

The output shows that even though the Deployment specifies three replicas, only two
Pods were created because of the quota you defined earlier

```yaml
spec
  ...
  replicas 3
...
status
  availableReplicas 2
...
lastUpdateTime 2021-04-02T205705Z
    message unable to create pods pods pod-quota-demo-1650323038- is forbidden
      exceeded quota pod-demo, requested pods1, used pods2, limited pods2
```

# # # Choice of resource

In this task you have defined a ResourceQuota that limited the total number of Pods, but
you could also limit the total number of other kinds of object. For example, you
might decide to limit how many
that can live in a single namespace.

# # Clean up

Delete your namespace

```shell
kubectl delete namespace quota-pod-example
```

# #  heading whatsnext

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure Memory and CPU Quotas for a Namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace)

* [Configure Quotas for API Objects](docstasksadminister-clusterquota-api-object)

# # # For app developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

* [Assign Pod-level CPU and memory resources](docstasksconfigure-pod-containerassign-pod-level-resources)

* [Configure Quality of Service for Pods](docstasksconfigure-pod-containerquality-service-pod)
