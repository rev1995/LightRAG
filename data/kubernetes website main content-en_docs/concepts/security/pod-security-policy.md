---
title Pod Security Policies
content_type concept
weight 30
---

 alert titleRemoved feature colorwarning
PodSecurityPolicy was [deprecated](blog20210408kubernetes-1-21-release-announcement#podsecuritypolicy-deprecation)
in Kubernetes v1.21, and removed from Kubernetes in v1.25.
 alert

Instead of using PodSecurityPolicy, you can enforce similar restrictions on Pods using
either or both

- [Pod Security Admission](docsconceptssecuritypod-security-admission)
- a 3rd party admission plugin, that you deploy and configure yourself

For a migration guide, see [Migrate from PodSecurityPolicy to the Built-In PodSecurity Admission Controller](docstasksconfigure-pod-containermigrate-from-psp).
For more information on the removal of this API,
see [PodSecurityPolicy Deprecation Past, Present, and Future](blog20210406podsecuritypolicy-deprecation-past-present-and-future).

If you are not running Kubernetes v, check the documentation for
your version of Kubernetes.
