---
title Set up an Extension API Server
reviewers
- lavalamp
- cheftako
- chenopis
content_type task
weight 15
---

Setting up an extension API server to work with the aggregation layer allows the Kubernetes apiserver to be extended with additional APIs, which are not part of the core Kubernetes APIs.

# #  heading prerequisites

* You must [configure the aggregation layer](docstasksextend-kubernetesconfigure-aggregation-layer) and enable the apiserver flags.

# # Set up an extension api-server to work with the aggregation layer

The following steps describe how to set up an extension-apiserver *at a high level*. These steps apply regardless if youre using YAML configs or using APIs. An attempt is made to specifically identify any differences between the two. For a concrete example of how they can be implemented using YAML configs, you can look at the [sample-apiserver](httpsgithub.comkubernetessample-apiserverblobmasterREADME.md) in the Kubernetes repo.

Alternatively, you can use an existing 3rd party solution, such as [apiserver-builder](httpsgithub.comkubernetes-sigsapiserver-builder-alphablobmasterREADME.md), which should generate a skeleton and automate all of the following steps for you.

1. Make sure the APIService API is enabled (check `--runtime-config`). It should be on by default, unless its been deliberately turned off in your cluster.
1. You may need to make an RBAC rule allowing you to add APIService objects, or get your cluster administrator to make one. (Since API extensions affect the entire cluster, it is not recommended to do testingdevelopmentdebug of an API extension in a live cluster.)
1. Create the Kubernetes namespace you want to run your extension api-service in.
1. Createget a CA cert to be used to sign the server cert the extension api-server uses for HTTPS.
1. Create a server certkey for the api-server to use for HTTPS. This cert should be signed by the above CA. It should also have a CN of the Kube DNS name. This is derived from the Kubernetes service and be of the form `..svc`
1. Create a Kubernetes secret with the server certkey in your namespace.
1. Create a Kubernetes deployment for the extension api-server and make sure you are loading the secret as a volume. It should contain a reference to a working image of your extension api-server. The deployment should also be in your namespace.
1. Make sure that your extension-apiserver loads those certs from that volume and that they are used in the HTTPS handshake.
1. Create a Kubernetes service account in your namespace.
1. Create a Kubernetes cluster role for the operations you want to allow on your resources.
1. Create a Kubernetes cluster role binding from the service account in your namespace to the cluster role you created.
1. Create a Kubernetes cluster role binding from the service account in your namespace to the `systemauth-delegator` cluster role to delegate auth decisions to the Kubernetes core API server.
1. Create a Kubernetes role binding from the service account in your namespace to the `extension-apiserver-authentication-reader` role. This allows your extension api-server to access the `extension-apiserver-authentication` configmap.
1. Create a Kubernetes apiservice. The CA cert above should be base64 encoded, stripped of new lines and used as the spec.caBundle in the apiservice. This should not be namespaced. If using the [kube-aggregator API](httpsgithub.comkuberneteskube-aggregator), only pass in the PEM encoded CA bundle because the base 64 encoding is done for you.
1. Use kubectl to get your resource. When run, kubectl should return No resources found.. This message
indicates that everything worked but you currently have no objects of that resource type created.

# #  heading whatsnext

* Walk through the steps to [configure the API aggregation layer](docstasksextend-kubernetesconfigure-aggregation-layer) and enable the apiserver flags.
* For a high level overview, see [Extending the Kubernetes API with the aggregation layer](docsconceptsextend-kubernetesapi-extensionapiserver-aggregation).
* Learn how to [Extend the Kubernetes API using Custom Resource Definitions](docstasksextend-kubernetescustom-resourcescustom-resource-definitions).
