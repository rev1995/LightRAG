---
title Use an Image Volume With a Pod
reviewers
content_type task
weight 210
min-kubernetes-server-version v1.31
---

This page shows how to configure a pod using image volumes. This allows you to
mount content from OCI registries inside containers.

# #  heading prerequisites

- The container runtime needs to support the image volumes feature
- You need to exec commands in the host
- You need to be able to exec into pods
- You need to enable the `ImageVolume` [feature gate](docsreferencecommand-line-tools-referencefeature-gates)

# # Run a Pod that uses an image volume #create-pod

An image volume for a pod is enabled by setting the `volumes.[*].image` field of `.spec`
to a valid reference and consuming it in the `volumeMounts` of the container. For example

 code_sample filepodsimage-volumes.yaml

1. Create the pod on your cluster

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodsimage-volumes.yaml
   ```

1. Attach to the container

   ```shell
   kubectl attach -it image-volume bash
   ```

1. Check the content of a file in the volume

   ```shell
   cat volumedirfile
   ```

   The output is similar to

   ```none
   1
   ```

   You can also check another file in a different path

   ```shell
   cat volumefile
   ```

   The output is similar to

   ```none
   2
   ```

# # Use `subPath` (or `subPathExpr`)

It is possible to utilize
[`subPath`](docsconceptsstoragevolumes#using-subpath) or
[`subPathExpr`](docsconceptsstoragevolumes#using-subpath-expanded-environment)
from Kubernetes v1.33 when using the image volume feature.

 code_sample filepodsimage-volumes-subpath.yaml

1. Create the pod on your cluster

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodsimage-volumes-subpath.yaml
   ```

1. Attach to the container

   ```shell
   kubectl attach -it image-volume bash
   ```

1. Check the content of the file from the `dir` sub path in the volume

   ```shell
   cat volumefile
   ```

   The output is similar to

   ```none
   1
   ```

# # Further reading

- [`image` volumes](docsconceptsstoragevolumes#image)
