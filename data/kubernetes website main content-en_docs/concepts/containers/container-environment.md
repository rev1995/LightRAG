---
reviewers
- mikedanese
- thockin
title Container Environment
content_type concept
weight 20
---

This page describes the resources available to Containers in the Container environment.

# # Container environment

The Kubernetes Container environment provides several important resources to Containers

* A filesystem, which is a combination of an [image](docsconceptscontainersimages) and one or more [volumes](docsconceptsstoragevolumes).
* Information about the Container itself.
* Information about other objects in the cluster.

# # # Container information

The *hostname* of a Container is the name of the Pod in which the Container is running.
It is available through the `hostname` command or the
[`gethostname`](httpsman7.orglinuxman-pagesman2gethostname.2.html)
function call in libc.

The Pod name and namespace are available as environment variables through the
[downward API](docstasksinject-data-applicationdownward-api-volume-expose-pod-information).

User defined environment variables from the Pod definition are also available to the Container,
as are any environment variables specified statically in the container image.

# # # Cluster information

A list of all services that were running when a Container was created is available to that Container as environment variables.
This list is limited to services within the same namespace as the new Containers Pod and Kubernetes control plane services.

For a service named *foo* that maps to a Container named *bar*,
the following variables are defined

```shell
FOO_SERVICE_HOST
FOO_SERVICE_PORT
```

Services have dedicated IP addresses and are available to the Container via DNS,
if [DNS addon](httpsreleases.k8s.iovclusteraddonsdns) is enabled. 

# #  heading whatsnext

* Learn more about [Container lifecycle hooks](docsconceptscontainerscontainer-lifecycle-hooks).
* Get hands-on experience
  [attaching handlers to Container lifecycle events](docstasksconfigure-pod-containerattach-handler-lifecycle-event).
