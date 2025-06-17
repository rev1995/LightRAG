---
reviewers
- johnbelamaric
title Using CoreDNS for Service Discovery
min-kubernetes-server-version v1.9
content_type task
weight 380
---

This page describes the CoreDNS upgrade process and how to install CoreDNS instead of kube-dns.

# #  heading prerequisites

# # About CoreDNS

[CoreDNS](httpscoredns.io) is a flexible, extensible DNS server
that can serve as the Kubernetes cluster DNS.
Like Kubernetes, the CoreDNS project is hosted by the
.

You can use CoreDNS instead of kube-dns in your cluster by replacing
kube-dns in an existing deployment, or by using tools like kubeadm
that will deploy and upgrade the cluster for you.

# # Installing CoreDNS

For manual deployment or replacement of kube-dns, see the documentation at the
[CoreDNS website](httpscoredns.iomanualinstallation).

# # Migrating to CoreDNS

# # # Upgrading an existing cluster with kubeadm

In Kubernetes version 1.21, kubeadm removed its support for `kube-dns` as a DNS application.
For `kubeadm` v, the only supported cluster DNS application
is CoreDNS.

You can move to CoreDNS when you use `kubeadm` to upgrade a cluster that is
using `kube-dns`. In this case, `kubeadm` generates the CoreDNS configuration
(Corefile) based upon the `kube-dns` ConfigMap, preserving configurations for
stub domains, and upstream name server.

# # Upgrading CoreDNS

You can check the version of CoreDNS that kubeadm installs for each version of
Kubernetes in the page
[CoreDNS version in Kubernetes](httpsgithub.comcorednsdeploymentblobmasterkubernetesCoreDNS-k8s_version.md).

CoreDNS can be upgraded manually in case you want to only upgrade CoreDNS
or use your own custom image.
There is a helpful [guideline and walkthrough](httpsgithub.comcorednsdeploymentblobmasterkubernetesUpgrading_CoreDNS.md)
available to ensure a smooth upgrade.
Make sure the existing CoreDNS configuration (Corefile) is retained when
upgrading your cluster.

If you are upgrading your cluster using the `kubeadm` tool, `kubeadm`
can take care of retaining the existing CoreDNS configuration automatically.

# # Tuning CoreDNS

When resource utilisation is a concern, it may be useful to tune the
configuration of CoreDNS. For more details, check out the
[documentation on scaling CoreDNS](httpsgithub.comcorednsdeploymentblobmasterkubernetesScaling_CoreDNS.md).

# #  heading whatsnext

You can configure [CoreDNS](httpscoredns.io) to support many more use cases than
kube-dns does by modifying the CoreDNS configuration (Corefile).
For more information, see the [documentation](httpscoredns.iopluginskubernetes)
for the `kubernetes` CoreDNS plugin, or read the
[Custom DNS Entries for Kubernetes](httpscoredns.io20170508custom-dns-entries-for-kubernetes).
in the CoreDNS blog.
