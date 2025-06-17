---
title Example Deploying Cassandra with a StatefulSet
reviewers
- ahmetb
content_type tutorial
weight 30
---

This tutorial shows you how to run [Apache Cassandra](httpscassandra.apache.org) on Kubernetes.
Cassandra, a database, needs persistent storage to provide data durability (application _state_).
In this example, a custom Cassandra seed provider lets the database discover new Cassandra instances as they join the Cassandra cluster.

*StatefulSets* make it easier to deploy stateful applications into your Kubernetes cluster.
For more information on the features used in this tutorial, see
[StatefulSet](docsconceptsworkloadscontrollersstatefulset).

Cassandra and Kubernetes both use the term _node_ to mean a member of a cluster. In this
tutorial, the Pods that belong to the StatefulSet are Cassandra nodes and are members
of the Cassandra cluster (called a _ring_). When those Pods run in your Kubernetes cluster,
the Kubernetes control plane schedules those Pods onto Kubernetes
.

When a Cassandra node starts, it uses a _seed list_ to bootstrap discovery of other
nodes in the ring.
This tutorial deploys a custom Cassandra seed provider that lets the database discover
new Cassandra Pods as they appear inside your Kubernetes cluster.

# #  heading objectives

* Create and validate a Cassandra headless .
* Use a  to create a Cassandra ring.
* Validate the StatefulSet.
* Modify the StatefulSet.
* Delete the StatefulSet and its .

# #  heading prerequisites

To complete this tutorial, you should already have a basic familiarity with
,
, and
.

# # # Additional Minikube setup instructions

[Minikube](httpsminikube.sigs.k8s.iodocs) defaults to 2048MB of memory and 2 CPU.
Running Minikube with the default resource configuration results in insufficient resource
errors during this tutorial. To avoid these errors, start Minikube with the following settings

```shell
minikube start --memory 5120 --cpus4
```

# # Creating a headless Service for Cassandra #creating-a-cassandra-headless-service

In Kubernetes, a  describes a set of
 that perform the same task.

The following Service is used for DNS lookups between Cassandra Pods and clients within your cluster

 code_sample fileapplicationcassandracassandra-service.yaml

Create a Service to track all Cassandra StatefulSet members from the `cassandra-service.yaml` file

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationcassandracassandra-service.yaml
```

# # # Validating (optional) #validating

Get the Cassandra Service.

```shell
kubectl get svc cassandra
```

The response is

```
NAME        TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)    AGE
cassandra   ClusterIP   None                 9042TCP   45s
```

If you dont see a Service named `cassandra`, that means creation failed. Read
[Debug Services](docstasksdebugdebug-applicationdebug-service)
for help troubleshooting common issues.

# # Using a StatefulSet to create a Cassandra ring

The StatefulSet manifest, included below, creates a Cassandra ring that consists of three Pods.

This example uses the default provisioner for Minikube.
Please update the following StatefulSet for the cloud you are working with.

 code_sample fileapplicationcassandracassandra-statefulset.yaml

Create the Cassandra StatefulSet from the `cassandra-statefulset.yaml` file

```shell
# Use this if you are able to apply cassandra-statefulset.yaml unmodified
kubectl apply -f httpsk8s.ioexamplesapplicationcassandracassandra-statefulset.yaml
```

If you need to modify `cassandra-statefulset.yaml` to suit your cluster, download
httpsk8s.ioexamplesapplicationcassandracassandra-statefulset.yaml and then apply
that manifest, from the folder you saved the modified version into
```shell
# Use this if you needed to modify cassandra-statefulset.yaml locally
kubectl apply -f cassandra-statefulset.yaml
```

# # Validating the Cassandra StatefulSet

1. Get the Cassandra StatefulSet

    ```shell
    kubectl get statefulset cassandra
    ```

    The response should be similar to

    ```
    NAME        DESIRED   CURRENT   AGE
    cassandra   3         0         13s
    ```

    The `StatefulSet` resource deploys Pods sequentially.

1. Get the Pods to see the ordered creation status

    ```shell
    kubectl get pods -lappcassandra
    ```

    The response should be similar to

    ```shell
    NAME          READY     STATUS              RESTARTS   AGE
    cassandra-0   11       Running             0          1m
    cassandra-1   01       ContainerCreating   0          8s
    ```

    It can take several minutes for all three Pods to deploy. Once they are deployed, the same command
    returns output similar to

    ```
    NAME          READY     STATUS    RESTARTS   AGE
    cassandra-0   11       Running   0          10m
    cassandra-1   11       Running   0          9m
    cassandra-2   11       Running   0          8m
    ```

3. Run the Cassandra [nodetool](httpscwiki.apache.orgconfluencedisplayCASSANDRA2NodeTool) inside the first Pod, to
   display the status of the ring.

    ```shell
    kubectl exec -it cassandra-0 -- nodetool status
    ```

    The response should look something like

    ```
    Datacenter DC1-K8Demo

    StatusUpDown
     StateNormalLeavingJoiningMoving
    --  Address     Load       Tokens       Owns (effective)  Host ID                               Rack
    UN  172.17.0.5  83.57 KiB  32           74.0             e2dd09e6-d9d3-477e-96c5-45094c08db0f  Rack1-K8Demo
    UN  172.17.0.4  101.04 KiB  32           58.8             f89d6835-3a42-4419-92b3-0e62cae1479c  Rack1-K8Demo
    UN  172.17.0.6  84.74 KiB  32           67.1             a6a1e8c2-3dc5-4417-b1a0-26507af2aaad  Rack1-K8Demo
    ```

# # Modifying the Cassandra StatefulSet

Use `kubectl edit` to modify the size of a Cassandra StatefulSet.

1. Run the following command

    ```shell
    kubectl edit statefulset cassandra
    ```

    This command opens an editor in your terminal. The line you need to change is the `replicas` field.
    The following sample is an excerpt of the StatefulSet file

    ```yaml
    # Please edit the object below. Lines beginning with a # will be ignored,
    # and an empty file will abort the edit. If an error occurs while saving this file will be
    # reopened with the relevant failures.
    #
    apiVersion appsv1
    kind StatefulSet
    metadata
      creationTimestamp 2016-08-13T184058Z
      generation 1
      labels
      app cassandra
      name cassandra
      namespace default
      resourceVersion 323
      uid 7a219483-6185-11e6-a910-42010a8a0fc0
    spec
      replicas 3
    ```

1. Change the number of replicas to 4, and then save the manifest.

    The StatefulSet now scales to run with 4 Pods.

1. Get the Cassandra StatefulSet to verify your change

    ```shell
    kubectl get statefulset cassandra
    ```

    The response should be similar to

    ```
    NAME        DESIRED   CURRENT   AGE
    cassandra   4         4         36m
    ```

# #  heading cleanup

Deleting or scaling a StatefulSet down does not delete the volumes associated with the StatefulSet.
This setting is for your safety because your data is more valuable than automatically purging all related StatefulSet resources.

Depending on the storage class and reclaim policy, deleting the *PersistentVolumeClaims* may cause the associated volumes
to also be deleted. Never assume youll be able to access data if its volume claims are deleted.

1. Run the following commands (chained together into a single command) to delete everything in the Cassandra StatefulSet

    ```shell
    grace(kubectl get pod cassandra-0 -ojsonpath.spec.terminationGracePeriodSeconds)
       kubectl delete statefulset -l appcassandra
       echo Sleeping grace seconds 12
       sleep grace
       kubectl delete persistentvolumeclaim -l appcassandra
    ```

1. Run the following command to delete the Service you set up for Cassandra

    ```shell
    kubectl delete service -l appcassandra
    ```

# # Cassandra container environment variables

The Pods in this tutorial use the [`gcr.iogoogle-samplescassandrav13`](httpsgithub.comkubernetesexamplesblobmastercassandraimageDockerfile)
image from Googles [container registry](httpscloud.google.comcontainer-registrydocs).
The Docker image above is based on [debian-base](httpsgithub.comkubernetesreleasetreemasterimagesbuilddebian-base)
and includes OpenJDK 8.

This image includes a standard Cassandra installation from the Apache Debian repo.
By using environment variables you can change values that are inserted into `cassandra.yaml`.

 Environment variable      Default value
 ------------------------ ---------------
 `CASSANDRA_CLUSTER_NAME`  `Test Cluster`
 `CASSANDRA_NUM_TOKENS`    `32`
 `CASSANDRA_RPC_ADDRESS`   `0.0.0.0`

# #  heading whatsnext

* Learn how to [Scale a StatefulSet](docstasksrun-applicationscale-stateful-set).
* Learn more about the [*KubernetesSeedProvider*](httpsgithub.comkubernetesexamplesblobmastercassandrajavasrcmainjavaiok8scassandraKubernetesSeedProvider.java)
* See more custom [Seed Provider Configurations](httpsgit.k8s.ioexamplescassandrajavaREADME.md)
