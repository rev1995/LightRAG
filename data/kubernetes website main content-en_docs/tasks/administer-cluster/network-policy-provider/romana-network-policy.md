---
reviewers
- chrismarino
title Romana for NetworkPolicy
content_type task
weight 50
---

This page shows how to use Romana for NetworkPolicy.

# #  heading prerequisites

Complete steps 1, 2, and 3 of the [kubeadm getting started guide](docsreferencesetup-toolskubeadm).

# # Installing Romana with kubeadm

Follow the [containerized installation guide](httpsgithub.comromanaromanatreemastercontainerize) for kubeadm.

# # Applying network policies

To apply network policies use one of the following

* [Romana network policies](httpsgithub.comromanaromanawikiRomana-policies).
    * [Example of Romana network policy](httpsgithub.comromanacoreblobmasterdocpolicy.md).
* The NetworkPolicy API.

# #  heading whatsnext

Once you have installed Romana, you can follow the
[Declare Network Policy](docstasksadminister-clusterdeclare-network-policy)
to try out Kubernetes NetworkPolicy.
