---
reviewers
- jpeeler
- pmorie
title Configure a Pod to Use a Projected Volume for Storage
content_type task
weight 100
---

This page shows how to use a [`projected`](docsconceptsstoragevolumes#projected) Volume to mount
several existing volume sources into the same directory. Currently, `secret`, `configMap`, `downwardAPI`,
and `serviceAccountToken` volumes can be projected.

`serviceAccountToken` is not a volume type.

# #  heading prerequisites

# # Configure a projected volume for a pod

In this exercise, you create username and password  from local files. You then create a Pod that runs one container, using a [`projected`](docsconceptsstoragevolumes#projected) Volume to mount the Secrets into the same shared directory.

Here is the configuration file for the Pod

 code_sample filepodsstorageprojected.yaml

1. Create the Secrets

    ```shell
    # Create files containing the username and password
    echo -n admin  .username.txt
    echo -n 1f2d1e2e67df  .password.txt

    # Package these files into secrets
    kubectl create secret generic user --from-file.username.txt
    kubectl create secret generic pass --from-file.password.txt
    ```
1. Create the Pod

    ```shell
    kubectl apply -f httpsk8s.ioexamplespodsstorageprojected.yaml
    ```
1. Verify that the Pods container is running, and then watch for changes to
the Pod

    ```shell
    kubectl get --watch pod test-projected-volume
    ```
    The output looks like this
    ```
    NAME                    READY     STATUS    RESTARTS   AGE
    test-projected-volume   11       Running   0          14s
    ```
1. In another terminal, get a shell to the running container

    ```shell
    kubectl exec -it test-projected-volume -- binsh
    ```
1. In your shell, verify that the `projected-volume` directory contains your projected sources

    ```shell
    ls projected-volume
    ```

# # Clean up

Delete the Pod and the Secrets

```shell
kubectl delete pod test-projected-volume
kubectl delete secret user pass
```

# #  heading whatsnext

* Learn more about [`projected`](docsconceptsstoragevolumes#projected) volumes.
* Read the [all-in-one volume](httpsgit.k8s.iodesign-proposals-archivenodeall-in-one-volume.md) design document.
