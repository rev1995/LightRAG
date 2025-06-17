---
title Container Runtime Interface (CRI)
content_type concept
weight 60
---

The CRI is a plugin interface which enables the kubelet to use a wide variety of
container runtimes, without having a need to recompile the cluster components.

You need a working
 on
each Node in your cluster, so that the
 can launch
 and their containers.

# # The API #api

The kubelet acts as a client when connecting to the container runtime via gRPC.
The runtime and image service endpoints have to be available in the container
runtime, which can be configured separately within the kubelet by using the
`--image-service-endpoint` [command line flags](docsreferencecommand-line-tools-referencekubelet).

For Kubernetes v, the kubelet prefers to use CRI `v1`.
If a container runtime does not support `v1` of the CRI, then the kubelet tries to
negotiate any older supported version.
The v kubelet can also negotiate CRI `v1alpha2`, but
this version is considered as deprecated.
If the kubelet cannot negotiate a supported CRI version, the kubelet gives up
and doesnt register as a node.

# # Upgrading

When upgrading Kubernetes, the kubelet tries to automatically select the
latest CRI version on restart of the component. If that fails, then the fallback
will take place as mentioned above. If a gRPC re-dial was required because the
container runtime has been upgraded, then the container runtime must also
support the initially selected version or the redial is expected to fail. This
requires a restart of the kubelet.

# #  heading whatsnext

- Learn more about the CRI [protocol definition](httpsgithub.comkubernetescri-apiblobc75ef5bpkgapisruntimev1api.proto)
