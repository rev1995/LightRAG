---
title Assign Pods to Nodes
content_type task
weight 150
---

This page shows how to assign a Kubernetes Pod to a particular node in a
Kubernetes cluster.

# #  heading prerequisites

# # Add a label to a node

1. List the  in your cluster, along with their labels

    ```shell
    kubectl get nodes --show-labels
    ```

    The output is similar to this

    ```shell
    NAME      STATUS    ROLES    AGE     VERSION        LABELS
    worker0   Ready        1d      v1.13.0        ...,kubernetes.iohostnameworker0
    worker1   Ready        1d      v1.13.0        ...,kubernetes.iohostnameworker1
    worker2   Ready        1d      v1.13.0        ...,kubernetes.iohostnameworker2
    ```
1. Choose one of your nodes, and add a label to it

    ```shell
    kubectl label nodes  disktypessd
    ```

    where `` is the name of your chosen node.

1. Verify that your chosen node has a `disktypessd` label

    ```shell
    kubectl get nodes --show-labels
    ```

    The output is similar to this

    ```shell
    NAME      STATUS    ROLES    AGE     VERSION        LABELS
    worker0   Ready        1d      v1.13.0        ...,disktypessd,kubernetes.iohostnameworker0
    worker1   Ready        1d      v1.13.0        ...,kubernetes.iohostnameworker1
    worker2   Ready        1d      v1.13.0        ...,kubernetes.iohostnameworker2
    ```

    In the preceding output, you can see that the `worker0` node has a
    `disktypessd` label.

# # Create a pod that gets scheduled to your chosen node

This pod configuration file describes a pod that has a node selector,
`disktype ssd`. This means that the pod will get scheduled on a node that has
a `disktypessd` label.

 code_sample filepodspod-nginx.yaml

1. Use the configuration file to create a pod that will get scheduled on your
   chosen node

    ```shell
    kubectl apply -f httpsk8s.ioexamplespodspod-nginx.yaml
    ```

1. Verify that the pod is running on your chosen node

    ```shell
    kubectl get pods --outputwide
    ```

    The output is similar to this

    ```shell
    NAME     READY     STATUS    RESTARTS   AGE    IP           NODE
    nginx    11       Running   0          13s    10.200.0.4   worker0
    ```
# # Create a pod that gets scheduled to specific node

You can also schedule a pod to one specific node via setting `nodeName`.

 code_sample filepodspod-nginx-specific-node.yaml

Use the configuration file to create a pod that will get scheduled on `foo-node` only.

# #  heading whatsnext

* Learn more about [labels and selectors](docsconceptsoverviewworking-with-objectslabels).
* Learn more about [nodes](docsconceptsarchitecturenodes).
