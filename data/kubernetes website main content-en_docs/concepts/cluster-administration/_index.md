---
title Cluster Administration
reviewers
- davidopp
- lavalamp
weight 100
content_type concept
description
  Lower-level detail relevant to creating or administering a Kubernetes cluster.
no_list true
card
  name setup
  weight 60
  anchors
  - anchor #securing-a-cluster
    title Securing a cluster
---

The cluster administration overview is for anyone creating or administering a Kubernetes cluster.
It assumes some familiarity with core Kubernetes [concepts](docsconcepts).

# # Planning a cluster

See the guides in [Setup](docssetup) for examples of how to plan, set up, and configure
Kubernetes clusters. The solutions listed in this article are called *distros*.

Not all distros are actively maintained. Choose distros which have been tested with a recent
version of Kubernetes.

Before choosing a guide, here are some considerations

- Do you want to try out Kubernetes on your computer, or do you want to build a high-availability,
  multi-node cluster Choose distros best suited for your needs.
- Will you be using **a hosted Kubernetes cluster**, such as
  [Google Kubernetes Engine](httpscloud.google.comkubernetes-engine), or **hosting your own cluster**
- Will your cluster be **on-premises**, or **in the cloud (IaaS)** Kubernetes does not directly
  support hybrid clusters. Instead, you can set up multiple clusters.
- **If you are configuring Kubernetes on-premises**, consider which
  [networking model](docsconceptscluster-administrationnetworking) fits best.
- Will you be running Kubernetes on **bare metal hardware** or on **virtual machines (VMs)**
- Do you **want to run a cluster**, or do you expect to do **active development of Kubernetes project code**
  If the latter, choose an actively-developed distro. Some distros only use binary releases, but
  offer a greater variety of choices.
- Familiarize yourself with the [components](docsconceptsoverviewcomponents) needed to run a cluster.

# # Managing a cluster

* Learn how to [manage nodes](docsconceptsarchitecturenodes).
  * Read about [Node autoscaling](docsconceptscluster-administrationnode-autoscaling).

* Learn how to set up and manage the [resource quota](docsconceptspolicyresource-quotas) for shared clusters.

# # Securing a cluster

* [Generate Certificates](docstasksadminister-clustercertificates) describes the steps to
  generate certificates using different tool chains.

* [Kubernetes Container Environment](docsconceptscontainerscontainer-environment) describes
  the environment for Kubelet managed containers on a Kubernetes node.

* [Controlling Access to the Kubernetes API](docsconceptssecuritycontrolling-access) describes
  how Kubernetes implements access control for its own API.

* [Authenticating](docsreferenceaccess-authn-authzauthentication) explains authentication in
  Kubernetes, including the various authentication options.

* [Authorization](docsreferenceaccess-authn-authzauthorization) is separate from
  authentication, and controls how HTTP calls are handled.

* [Using Admission Controllers](docsreferenceaccess-authn-authzadmission-controllers)
  explains plug-ins which intercepts requests to the Kubernetes API server after authentication
  and authorization.

* [Admission Webhook Good Practices](docsconceptscluster-administrationadmission-webhooks-good-practices)
  provides good practices and considerations when designing mutating admission
  webhooks and validating admission webhooks.

* [Using Sysctls in a Kubernetes Cluster](docstasksadminister-clustersysctl-cluster)
  describes to an administrator how to use the `sysctl` command-line tool to set kernel parameters
.

* [Auditing](docstasksdebugdebug-clusteraudit) describes how to interact with Kubernetes
  audit logs.

# # # Securing the kubelet

* [Control Plane-Node communication](docsconceptsarchitecturecontrol-plane-node-communication)
* [TLS bootstrapping](docsreferenceaccess-authn-authzkubelet-tls-bootstrapping)
* [Kubelet authenticationauthorization](docsreferenceaccess-authn-authzkubelet-authn-authz)

# # Optional Cluster Services

* [DNS Integration](docsconceptsservices-networkingdns-pod-service) describes how to resolve
  a DNS name directly to a Kubernetes service.

* [Logging and Monitoring Cluster Activity](docsconceptscluster-administrationlogging)
  explains how logging in Kubernetes works and how to implement it.
