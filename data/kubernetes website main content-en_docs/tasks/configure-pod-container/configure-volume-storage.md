---
title Configure a Pod to Use a Volume for Storage
content_type task
weight 80
---

This page shows how to configure a Pod to use a Volume for storage.

A Containers file system lives only as long as the Container does. So when a
Container terminates and restarts, filesystem changes are lost. For more
consistent storage that is independent of the Container, you can use a
[Volume](docsconceptsstoragevolumes). This is especially important for stateful
applications, such as key-value stores (such as Redis) and databases.

# #  heading prerequisites

# # Configure a volume for a Pod

In this exercise, you create a Pod that runs one Container. This Pod has a
Volume of type
[emptyDir](docsconceptsstoragevolumes#emptydir)
that lasts for the life of the Pod, even if the Container terminates and
restarts. Here is the configuration file for the Pod

 code_sample filepodsstorageredis.yaml

1. Create the Pod

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodsstorageredis.yaml
   ```

1. Verify that the Pods Container is running, and then watch for changes to
   the Pod

   ```shell
   kubectl get pod redis --watch
   ```

   The output looks like this

   ```console
   NAME      READY     STATUS    RESTARTS   AGE
   redis     11       Running   0          13s
   ```

1. In another terminal, get a shell to the running Container

   ```shell
   kubectl exec -it redis -- binbash
   ```

1. In your shell, go to `dataredis`, and then create a file

   ```shell
   rootredisdata# cd dataredis
   rootredisdataredis# echo Hello  test-file
   ```

1. In your shell, list the running processes

   ```shell
   rootredisdataredis# apt-get update
   rootredisdataredis# apt-get install procps
   rootredisdataredis# ps aux
   ```

   The output is similar to this

   ```console
   USER       PID CPU MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
   redis        1  0.1  0.1  33308  3828         Ssl  0046   000 redis-server *6379
   root        12  0.0  0.0  20228  3020         Ss   0047   000 binbash
   root        15  0.0  0.0  17500  2072         R   0048   000 ps aux
   ```

1. In your shell, kill the Redis process

   ```shell
   rootredisdataredis# kill
   ```

   where `` is the Redis process ID (PID).

1. In your original terminal, watch for changes to the Redis Pod. Eventually,
   you will see something like this

   ```console
   NAME      READY     STATUS     RESTARTS   AGE
   redis     11       Running    0          13s
   redis     01       Completed  0         6m
   redis     11       Running    1         6m
   ```

At this point, the Container has terminated and restarted. This is because the
Redis Pod has a
[restartPolicy](docsreferencegeneratedkubernetes-api#podspec-v1-core)
of `Always`.

1. Get a shell into the restarted Container

   ```shell
   kubectl exec -it redis -- binbash
   ```

1. In your shell, go to `dataredis`, and verify that `test-file` is still there.

   ```shell
   rootredisdataredis# cd dataredis
   rootredisdataredis# ls
   test-file
   ```

1. Delete the Pod that you created for this exercise

   ```shell
   kubectl delete pod redis
   ```

# #  heading whatsnext

- See [Volume](docsreferencegeneratedkubernetes-api#volume-v1-core).

- See [Pod](docsreferencegeneratedkubernetes-api#pod-v1-core).

- In addition to the local disk storage provided by `emptyDir`, Kubernetes
  supports many different network-attached storage solutions, including PD on
  GCE and EBS on EC2, which are preferred for critical data and will handle
  details such as mounting and unmounting the devices on the nodes. See
  [Volumes](docsconceptsstoragevolumes) for more details.
