---
reviewers
- jpbetz
title Coordinated Leader Election
content_type concept
weight 200
---

Kubernetes  includes a beta feature that allows  components to
deterministically select a leader via _coordinated leader election_.
This is useful to satisfy Kubernetes version skew constraints during cluster upgrades.
Currently, the only builtin selection strategy is `OldestEmulationVersion`,
preferring the leader with the lowest emulation version, followed by binary
version, followed by creation timestamp.

# # Enabling coordinated leader election

Ensure that `CoordinatedLeaderElection` [feature
gate](docsreferencecommand-line-tools-referencefeature-gates) is enabled
when you start the  and that the `coordination.k8s.iov1beta1` API group is
enabled.

This can be done by setting flags `--feature-gatesCoordinatedLeaderElectiontrue` and
`--runtime-configcoordination.k8s.iov1beta1true`.

# # Component configuration

Provided that you have enabled the `CoordinatedLeaderElection` feature gate _and_
have the `coordination.k8s.iov1beta1` API group enabled, compatible control plane
components automatically use the LeaseCandidate and Lease APIs to elect a leader
as needed.

For Kubernetes , two control plane components
(kube-controller-manager and kube-scheduler) automatically use coordinated
leader election when the feature gate and API group are enabled.