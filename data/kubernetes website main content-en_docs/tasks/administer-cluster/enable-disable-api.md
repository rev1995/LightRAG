---
title Enable Or Disable A Kubernetes API
content_type task
weight 200
---

This page shows how to enable or disable an API version from your clusters
.

Specific API versions can be turned on or off by passing `--runtime-configapi` as a
command line argument to the API server. The values for this argument are a comma-separated
list of API versions. Later values override earlier values.

The `runtime-config` command line argument also supports 2 special keys

- `apiall`, representing all known APIs
- `apilegacy`, representing only legacy APIs. Legacy APIs are any APIs that have been
   explicitly [deprecated](docsreferenceusing-apideprecation-policy).

For example, to turn off all API versions except v1, pass `--runtime-configapiallfalse,apiv1true`
to the `kube-apiserver`.

# #  heading whatsnext

Read the [full documentation](docsreferencecommand-line-tools-referencekube-apiserver)
for the `kube-apiserver` component.
