---
title Use Cascading Deletion in a Cluster
content_type task
weight 360
---

This page shows you how to specify the type of
[cascading deletion](docsconceptsarchitecturegarbage-collection#cascading-deletion)
to use in your cluster during .

# #  heading prerequisites

You also need to [create a sample Deployment](docstasksrun-applicationrun-stateless-application-deployment#creating-and-exploring-an-nginx-deployment)
to experiment with the different types of cascading deletion. You will need to
recreate the Deployment for each type.

# # Check owner references on your pods

Check that the `ownerReferences` field is present on your pods

```shell
kubectl get pods -l appnginx --outputyaml
```

The output has an `ownerReferences` field similar to this

```yaml
apiVersion v1
    ...
    ownerReferences
    - apiVersion appsv1
      blockOwnerDeletion true
      controller true
      kind ReplicaSet
      name nginx-deployment-6b474476c4
      uid 4fdcd81c-bd5d-41f7-97af-3a3b759af9a7
    ...
```

# # Use foreground cascading deletion #use-foreground-cascading-deletion

By default, Kubernetes uses [background cascading deletion](docsconceptsarchitecturegarbage-collection#background-deletion)
to delete dependents of an object. You can switch to foreground cascading deletion
using either `kubectl` or the Kubernetes API, depending on the Kubernetes
version your cluster runs.

You can delete objects using foreground cascading deletion using `kubectl` or the
Kubernetes API.

**Using kubectl**

Run the following command

```shell
kubectl delete deployment nginx-deployment --cascadeforeground
```

**Using the Kubernetes API**

1. Start a local proxy session

   ```shell
   kubectl proxy --port8080
   ```

1. Use `curl` to trigger deletion

   ```shell
   curl -X DELETE localhost8080apisappsv1namespacesdefaultdeploymentsnginx-deployment
       -d kindDeleteOptions,apiVersionv1,propagationPolicyForeground
       -H Content-Type applicationjson
   ```

   The output contains a `foregroundDeletion`
   like this

   ```
   kind Deployment,
   apiVersion appsv1,
   metadata
       name nginx-deployment,
       namespace default,
       uid d1ce1b02-cae8-4288-8a53-30e84d8fa505,
       resourceVersion 1363097,
       creationTimestamp 2021-07-08T202437Z,
       deletionTimestamp 2021-07-08T202739Z,
       finalizers [
         foregroundDeletion
       ]
       ...
   ```

# # Use background cascading deletion #use-background-cascading-deletion

1. [Create a sample Deployment](docstasksrun-applicationrun-stateless-application-deployment#creating-and-exploring-an-nginx-deployment).
1. Use either `kubectl` or the Kubernetes API to delete the Deployment,
   depending on the Kubernetes version your cluster runs.

You can delete objects using background cascading deletion using `kubectl`
or the Kubernetes API.

Kubernetes uses background cascading deletion by default, and does so
even if you run the following commands without the `--cascade` flag or the
`propagationPolicy` argument.

**Using kubectl**

Run the following command

```shell
kubectl delete deployment nginx-deployment --cascadebackground
```

**Using the Kubernetes API**

1. Start a local proxy session

   ```shell
   kubectl proxy --port8080
   ```

1. Use `curl` to trigger deletion

   ```shell
   curl -X DELETE localhost8080apisappsv1namespacesdefaultdeploymentsnginx-deployment
       -d kindDeleteOptions,apiVersionv1,propagationPolicyBackground
       -H Content-Type applicationjson
   ```

   The output is similar to this

   ```
   kind Status,
   apiVersion v1,
   ...
   status Success,
   details
       name nginx-deployment,
       group apps,
       kind deployments,
       uid cc9eefb9-2d49-4445-b1c1-d261c9396456

   ```

# # Delete owner objects and orphan dependents #set-orphan-deletion-policy

By default, when you tell Kubernetes to delete an object, the
 also deletes
dependent objects. You can make Kubernetes *orphan* these dependents using
`kubectl` or the Kubernetes API, depending on the Kubernetes version your
cluster runs.

**Using kubectl**

Run the following command

```shell
kubectl delete deployment nginx-deployment --cascadeorphan
```

**Using the Kubernetes API**

1. Start a local proxy session

   ```shell
   kubectl proxy --port8080
   ```

1. Use `curl` to trigger deletion

   ```shell
   curl -X DELETE localhost8080apisappsv1namespacesdefaultdeploymentsnginx-deployment
       -d kindDeleteOptions,apiVersionv1,propagationPolicyOrphan
       -H Content-Type applicationjson
   ```

   The output contains `orphan` in the `finalizers` field, similar to this

   ```
   kind Deployment,
   apiVersion appsv1,
   namespace default,
   uid 6f577034-42a0-479d-be21-78018c466f1f,
   creationTimestamp 2021-07-09T164637Z,
   deletionTimestamp 2021-07-09T164708Z,
   deletionGracePeriodSeconds 0,
   finalizers [
     orphan
   ],
   ...
   ```

You can check that the Pods managed by the Deployment are still running

```shell
kubectl get pods -l appnginx
```

# #  heading whatsnext

* Learn about [owners and dependents](docsconceptsoverviewworking-with-objectsowners-dependents) in Kubernetes.
* Learn about Kubernetes [finalizers](docsconceptsoverviewworking-with-objectsfinalizers).
* Learn about [garbage collection](docsconceptsarchitecturegarbage-collection).
