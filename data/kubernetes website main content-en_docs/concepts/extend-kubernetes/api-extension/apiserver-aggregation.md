---
title Kubernetes API Aggregation Layer
reviewers
- lavalamp
- cheftako
- chenopis
content_type concept
weight 20
---

The aggregation layer allows Kubernetes to be extended with additional APIs, beyond what is
offered by the core Kubernetes APIs.
The additional APIs can either be ready-made solutions such as a
[metrics server](httpsgithub.comkubernetes-sigsmetrics-server), or APIs that you develop yourself.

The aggregation layer is different from
[Custom Resource Definitions](docsconceptsextend-kubernetesapi-extensioncustom-resources),
which are a way to make the
recognise new kinds of object.

# # Aggregation layer

The aggregation layer runs in-process with the kube-apiserver. Until an extension resource is
registered, the aggregation layer will do nothing. To register an API, you add an _APIService_
object, which claims the URL path in the Kubernetes API. At that point, the aggregation layer
will proxy anything sent to that API path (e.g. `apismyextension.mycompany.iov1`) to the
registered APIService.

The most common way to implement the APIService is to run an *extension API server* in Pod(s) that
run in your cluster. If youre using the extension API server to manage resources in your cluster,
the extension API server (also written as extension-apiserver) is typically paired with one or
more . The apiserver-builder
library provides a skeleton for both extension API servers and the associated controller(s).

# # # Response latency

Extension API servers should have low latency networking to and from the kube-apiserver.
Discovery requests are required to round-trip from the kube-apiserver in five seconds or less.

If your extension API server cannot achieve that latency requirement, consider making changes that
let you meet it.

# #  heading whatsnext

* To get the aggregator working in your environment, [configure the aggregation layer](docstasksextend-kubernetesconfigure-aggregation-layer).
* Then, [setup an extension api-server](docstasksextend-kubernetessetup-extension-api-server) to work with the aggregation layer.
* Read about [APIService](docsreferencekubernetes-apicluster-resourcesapi-service-v1) in the API reference
*   Learn about [Declarative Validation Concepts](docsreferenceusing-apideclarative-validation.md), an internal mechanism for defining validation rules that in the future will help support validation for extension API server development.

Alternatively learn how to
[extend the Kubernetes API using Custom Resource Definitions](docstasksextend-kubernetescustom-resourcescustom-resource-definitions).
