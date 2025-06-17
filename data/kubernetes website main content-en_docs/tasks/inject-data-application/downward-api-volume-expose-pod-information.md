---
title Expose Pod Information to Containers Through Files
content_type task
weight 40
---

This page shows how a Pod can use a
[`downwardAPI` volume](docsconceptsstoragevolumes#downwardapi),
to expose information about itself to containers running in the Pod.
A `downwardAPI` volume can expose Pod fields and container fields.

In Kubernetes, there are two ways to expose Pod and container fields to a running container

* [Environment variables](docstasksinject-data-applicationenvironment-variable-expose-pod-information)
* Volume files, as explained in this task

Together, these two ways of exposing Pod and container fields are called the
_downward API_.

# #  heading prerequisites

# # Store Pod fields

In this part of exercise, you create a Pod that has one container, and you
project Pod-level fields into the running container as files.
Here is the manifest for the Pod

 code_sample filepodsinjectdapi-volume.yaml

In the manifest, you can see that the Pod has a `downwardAPI` Volume,
and the container mounts the volume at `etcpodinfo`.

Look at the `items` array under `downwardAPI`. Each element of the array
defines a `downwardAPI` volume.
The first element specifies that the value of the Pods
`metadata.labels` field should be stored in a file named `labels`.
The second element specifies that the value of the Pods `annotations`
field should be stored in a file named `annotations`.

The fields in this example are Pod fields. They are not
fields of the container in the Pod.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsinjectdapi-volume.yaml
```

Verify that the container in the Pod is running

```shell
kubectl get pods
```

View the containers logs

```shell
kubectl logs kubernetes-downwardapi-volume-example
```

The output shows the contents of the `labels` file and the `annotations` file

```
clustertest-cluster1
rackrack-22
zoneus-est-coast

buildtwo
builderjohn-doe
```

Get a shell into the container that is running in your Pod

```shell
kubectl exec -it kubernetes-downwardapi-volume-example -- sh
```

In your shell, view the `labels` file

```shell
# cat etcpodinfolabels
```

The output shows that all of the Pods labels have been written
to the `labels` file

```shell
clustertest-cluster1
rackrack-22
zoneus-est-coast
```

Similarly, view the `annotations` file

```shell
# cat etcpodinfoannotations
```

View the files in the `etcpodinfo` directory

```shell
# ls -laR etcpodinfo
```

In the output, you can see that the `labels` and `annotations` files
are in a temporary subdirectory in this example,
`..2982_06_02_21_47_53.299460680`. In the `etcpodinfo` directory, `..data` is
a symbolic link to the temporary subdirectory. Also in the `etcpodinfo` directory,
`labels` and `annotations` are symbolic links.

```
drwxr-xr-x  ... Feb 6 2147 ..2982_06_02_21_47_53.299460680
lrwxrwxrwx  ... Feb 6 2147 ..data - ..2982_06_02_21_47_53.299460680
lrwxrwxrwx  ... Feb 6 2147 annotations - ..dataannotations
lrwxrwxrwx  ... Feb 6 2147 labels - ..datalabels

etc..2982_06_02_21_47_53.299460680
total 8
-rw-r--r--  ... Feb  6 2147 annotations
-rw-r--r--  ... Feb  6 2147 labels
```

Using symbolic links enables dynamic atomic refresh of the metadata updates are
written to a new temporary directory, and the `..data` symlink is updated
atomically using [rename(2)](httpman7.orglinuxman-pagesman2rename.2.html).

A container using Downward API as a
[subPath](docsconceptsstoragevolumes#using-subpath) volume mount will not
receive Downward API updates.

Exit the shell

```shell
# exit
```

# # Store container fields

The preceding exercise, you made Pod-level fields accessible using the
downward API.
In this next exercise, you are going to pass fields that are part of the Pod
definition, but taken from the specific
[container](docsreferencekubernetes-apiworkload-resourcespod-v1#Container)
rather than from the Pod overall. Here is a manifest for a Pod that again has
just one container

 code_sample filepodsinjectdapi-volume-resources.yaml

In the manifest, you can see that the Pod has a
[`downwardAPI` volume](docsconceptsstoragevolumes#downwardapi),
and that the single container in that Pod mounts the volume at `etcpodinfo`.

Look at the `items` array under `downwardAPI`. Each element of the array
defines a file in the downward API volume.

The first element specifies that in the container named `client-container`,
the value of the `limits.cpu` field in the format specified by `1m` should be
published as a file named `cpu_limit`. The `divisor` field is optional and has the
default value of `1`. A divisor of 1 means cores for `cpu` resources, or
bytes for `memory` resources.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsinjectdapi-volume-resources.yaml
```

Get a shell into the container that is running in your Pod

```shell
kubectl exec -it kubernetes-downwardapi-volume-example-2 -- sh
```

In your shell, view the `cpu_limit` file

```shell
# Run this in a shell inside the container
cat etcpodinfocpu_limit
```

You can use similar commands to view the `cpu_request`, `mem_limit` and
`mem_request` files.

# # Project keys to specific paths and file permissions

You can project keys to specific paths and specific permissions on a per-file
basis. For more information, see
[Secrets](docsconceptsconfigurationsecret).

# #  heading whatsnext

* Read the [`spec`](docsreferencekubernetes-apiworkload-resourcespod-v1#PodSpec)
  API definition for Pod. This includes the definition of Container (part of Pod).
* Read the list of [available fields](docsconceptsworkloadspodsdownward-api#available-fields) that you
  can expose using the downward API.

Read about volumes in the legacy API reference
* Check the [`Volume`](docsreferencegeneratedkubernetes-api#volume-v1-core)
  API definition which defines a generic volume in a Pod for containers to access.
* Check the [`DownwardAPIVolumeSource`](docsreferencegeneratedkubernetes-api#downwardapivolumesource-v1-core)
  API definition which defines a volume that contains Downward API information.
* Check the [`DownwardAPIVolumeFile`](docsreferencegeneratedkubernetes-api#downwardapivolumefile-v1-core)
  API definition which contains references to object or resource fields for
  populating a file in the Downward API volume.
* Check the [`ResourceFieldSelector`](docsreferencegeneratedkubernetes-api#resourcefieldselector-v1-core)
  API definition which specifies the container resources and their output format.
