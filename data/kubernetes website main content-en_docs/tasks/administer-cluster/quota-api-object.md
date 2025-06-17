---
title Configure Quotas for API Objects
content_type task
weight 130
---

This page shows how to configure quotas for API objects, including
PersistentVolumeClaims and Services. A quota restricts the number of
objects, of a particular type, that can be created in a namespace.
You specify quotas in a
[ResourceQuota](docsreferencegeneratedkubernetes-api#resourcequota-v1-core)
object.

# #  heading prerequisites

# # Create a namespace

Create a namespace so that the resources you create in this exercise are
isolated from the rest of your cluster.

```shell
kubectl create namespace quota-object-example
```

# # Create a ResourceQuota

Here is the configuration file for a ResourceQuota object

 code_sample fileadminresourcequota-objects.yaml

Create the ResourceQuota

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-objects.yaml --namespacequota-object-example
```

View detailed information about the ResourceQuota

```shell
kubectl get resourcequota object-quota-demo --namespacequota-object-example --outputyaml
```

The output shows that in the quota-object-example namespace, there can be at most
one PersistentVolumeClaim, at most two Services of type LoadBalancer, and no Services
of type NodePort.

```yaml
status
  hard
    persistentvolumeclaims 1
    services.loadbalancers 2
    services.nodeports 0
  used
    persistentvolumeclaims 0
    services.loadbalancers 0
    services.nodeports 0
```

# # Create a PersistentVolumeClaim

Here is the configuration file for a PersistentVolumeClaim object

 code_sample fileadminresourcequota-objects-pvc.yaml

Create the PersistentVolumeClaim

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-objects-pvc.yaml --namespacequota-object-example
```

Verify that the PersistentVolumeClaim was created

```shell
kubectl get persistentvolumeclaims --namespacequota-object-example
```

The output shows that the PersistentVolumeClaim exists and has status Pending

```
NAME             STATUS
pvc-quota-demo   Pending
```

# # Attempt to create a second PersistentVolumeClaim

Here is the configuration file for a second PersistentVolumeClaim

 code_sample fileadminresourcequota-objects-pvc-2.yaml

Attempt to create the second PersistentVolumeClaim

```shell
kubectl apply -f httpsk8s.ioexamplesadminresourcequota-objects-pvc-2.yaml --namespacequota-object-example
```

The output shows that the second PersistentVolumeClaim was not created,
because it would have exceeded the quota for the namespace.

```
persistentvolumeclaims pvc-quota-demo-2 is forbidden
exceeded quota object-quota-demo, requested persistentvolumeclaims1,
used persistentvolumeclaims1, limited persistentvolumeclaims1
```

# # Notes

These are the strings used to identify API resources that can be constrained
by quotas

StringAPI Object
podsPod
servicesService
replicationcontrollersReplicationController
resourcequotasResourceQuota
secretsSecret
configmapsConfigMap
persistentvolumeclaimsPersistentVolumeClaim
services.nodeportsService of type NodePort
services.loadbalancersService of type LoadBalancer

# # Clean up

Delete your namespace

```shell
kubectl delete namespace quota-object-example
```

# #  heading whatsnext

# # # For cluster administrators

* [Configure Default Memory Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcesmemory-default-namespace)

* [Configure Default CPU Requests and Limits for a Namespace](docstasksadminister-clustermanage-resourcescpu-default-namespace)

* [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)

* [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)

* [Configure Memory and CPU Quotas for a Namespace](docstasksadminister-clustermanage-resourcesquota-memory-cpu-namespace)

* [Configure a Pod Quota for a Namespace](docstasksadminister-clustermanage-resourcesquota-pod-namespace)

# # # For app developers

* [Assign Memory Resources to Containers and Pods](docstasksconfigure-pod-containerassign-memory-resource)

* [Assign CPU Resources to Containers and Pods](docstasksconfigure-pod-containerassign-cpu-resource)

* [Assign Pod-level CPU and memory resources](docstasksconfigure-pod-containerassign-pod-level-resources)

* [Configure Quality of Service for Pods](docstasksconfigure-pod-containerquality-service-pod)
