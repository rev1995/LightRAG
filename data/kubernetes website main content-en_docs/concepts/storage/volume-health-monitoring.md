---
reviewers
- jsafrane
- saad-ali
- msau42
- xing-yang
title Volume Health Monitoring
content_type concept
weight 100
---

 volume health monitoring allows
CSI Drivers to detect abnormal volume conditions from the underlying storage systems
and report them as events on
or .

# # Volume health monitoring

Kubernetes _volume health monitoring_ is part of how Kubernetes implements the
Container Storage Interface (CSI). Volume health monitoring feature is implemented
in two components an External Health Monitor controller, and the
.

If a CSI Driver supports Volume Health Monitoring feature from the controller side,
an event will be reported on the related
 (PVC)
when an abnormal volume condition is detected on a CSI volume.

The External Health Monitor
also watches for node failure events. You can enable node failure monitoring by setting
the `enable-node-watcher` flag to true. When the external health monitor detects a node
failure event, the controller reports an Event will be reported on the PVC to indicate
that pods using this PVC are on a failed node.

If a CSI Driver supports Volume Health Monitoring feature from the node side,
an Event will be reported on every Pod using the PVC when an abnormal volume
condition is detected on a CSI volume. In addition, Volume Health information
is exposed as Kubelet VolumeStats metrics. A new metric kubelet_volume_stats_health_status_abnormal
is added. This metric includes two labels `namespace` and `persistentvolumeclaim`.
The count is either 1 or 0. 1 indicates the volume is unhealthy, 0 indicates volume
is healthy. For more information, please check
[KEP](httpsgithub.comkubernetesenhancementstreemasterkepssig-storage1432-volume-health-monitor#kubelet-metrics-changes).

You need to enable the `CSIVolumeHealth` [feature gate](docsreferencecommand-line-tools-referencefeature-gates)
to use this feature from the node side.

# #  heading whatsnext

See the [CSI driver documentation](httpskubernetes-csi.github.iodocsdrivers.html)
to find out which CSI drivers have implemented this feature.
