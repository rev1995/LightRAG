---
title Policies
weight 90
no_list true
description
  Manage security and best-practices with policies.
---

Kubernetes policies are configurations that manage other configurations or runtime behaviors. Kubernetes offers various forms of policies, described below

# # Apply policies using API objects

 Some API objects act as policies. Here are some examples
* [NetworkPolicies](docsconceptsservices-networkingnetwork-policies) can be used to restrict ingress and egress traffic for a workload.
* [LimitRanges](docsconceptspolicylimit-range) manage resource allocation constraints across different object kinds.
* [ResourceQuotas](docsconceptspolicyresource-quotas) limit resource consumption for a .

# # Apply policies using admission controllers

An
runs in the API server
and can validate or mutate API requests. Some admission controllers act to apply policies.
For example, the [AlwaysPullImages](docsreferenceaccess-authn-authzadmission-controllers#alwayspullimages) admission controller modifies a new Pod to set the image pull policy to `Always`.

Kubernetes has several built-in admission controllers that are configurable via the API server `--enable-admission-plugins` flag.

Details on admission controllers, with the complete list of available admission controllers, are documented in a dedicated section

* [Admission Controllers](docsreferenceaccess-authn-authzadmission-controllers)

# # Apply policies using ValidatingAdmissionPolicy

Validating admission policies allow configurable validation checks to be executed in the API server using the Common Expression Language (CEL). For example, a `ValidatingAdmissionPolicy` can be used to disallow use of the `latest` image tag.

A `ValidatingAdmissionPolicy` operates on an API request and can be used to block, audit, and warn users about non-compliant configurations.

Details on the `ValidatingAdmissionPolicy` API, with examples, are documented in a dedicated section
* [Validating Admission Policy](docsreferenceaccess-authn-authzvalidating-admission-policy)

# # Apply policies using dynamic admission control

Dynamic admission controllers (or admission webhooks) run outside the API server as separate applications that register to receive webhooks requests to perform validation or mutation of API requests.

Dynamic admission controllers can be used to apply policies on API requests and trigger other policy-based workflows. A dynamic admission controller can perform complex checks including those that require retrieval of other cluster resources and external data. For example, an image verification check can lookup data from OCI registries to validate the container image signatures and attestations.

Details on dynamic admission control are documented in a dedicated section
* [Dynamic Admission Control](docsreferenceaccess-authn-authzextensible-admission-controllers)

# # # Implementations #implementations-admission-control

 thirdparty-content

Dynamic Admission Controllers that act as flexible policy engines are being developed in the Kubernetes ecosystem, such as
- [Kubewarden](httpsgithub.comkubewarden)
- [Kyverno](httpskyverno.io)
- [OPA Gatekeeper](httpsgithub.comopen-policy-agentgatekeeper)
- [Polaris](httpspolaris.docs.fairwinds.comadmission-controller)

# # Apply policies using Kubelet configurations

Kubernetes allows configuring the Kubelet on each worker node.  Some Kubelet configurations act as policies
* [Process ID limits and reservations](docsconceptspolicypid-limiting) are used to limit and reserve allocatable PIDs.
* [Node Resource Managers](docsconceptspolicynode-resource-managers) can manage compute, memory, and device resources for latency-critical and high-throughput workloads.
