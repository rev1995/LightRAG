---
reviewers
- jsafrane
- saad-ali
- thockin
- msau42
title Node-specific Volume Limits
content_type concept
weight 90
---

This page describes the maximum number of volumes that can be attached
to a Node for various cloud providers.

Cloud providers like Google, Amazon, and Microsoft typically have a limit on
how many volumes can be attached to a Node. It is important for Kubernetes to
respect those limits. Otherwise, Pods scheduled on a Node could get stuck
waiting for volumes to attach.

# # Kubernetes default limits

The Kubernetes scheduler has default limits on the number of volumes
that can be attached to a Node

  Cloud serviceMaximum volumes per Node
  Amazon Elastic Block Store (EBS)39
  Google Persistent Disk16
  Microsoft Azure Disk Storage16

# # Custom limits

You can change these limits by setting the value of the
`KUBE_MAX_PD_VOLS` environment variable, and then starting the scheduler.
CSI drivers might have a different procedure, see their documentation
on how to customize their limits.

Use caution if you set a limit that is higher than the default limit. Consult
the cloud providers documentation to make sure that Nodes can actually support
the limit you set.

The limit applies to the entire cluster, so it affects all Nodes.

# # Dynamic volume limits

Dynamic volume limits are supported for following volume types.

- Amazon EBS
- Google Persistent Disk
- Azure Disk
- CSI

For volumes managed by in-tree volume plugins, Kubernetes automatically determines the Node
type and enforces the appropriate maximum number of volumes for the node. For example

* On
Google Compute Engine,
up to 127 volumes can be attached to a node, [depending on the node
type](httpscloud.google.comcomputedocsdisks#pdnumberlimits).

* For Amazon EBS disks on M5,C5,R5,T3 and Z1D instance types, Kubernetes allows only 25
volumes to be attached to a Node. For other instance types on
Amazon Elastic Compute Cloud (EC2),
Kubernetes allows 39 volumes to be attached to a Node.

* On Azure, up to 64 disks can be attached to a node, depending on the node type. For more details, refer to [Sizes for virtual machines in Azure](httpsdocs.microsoft.comen-usazurevirtual-machineswindowssizes).

* If a CSI storage driver advertises a maximum number of volumes for a Node (using `NodeGetInfo`), the  honors that limit.
Refer to the [CSI specifications](httpsgithub.comcontainer-storage-interfacespecblobmasterspec.md#nodegetinfo) for details.

* For volumes managed by in-tree plugins that have been migrated to a CSI driver, the maximum number of volumes will be the one reported by the CSI driver.

# # # Mutable CSI Node Allocatable Count

CSI drivers can dynamically adjust the maximum number of volumes that can be attached to a Node at runtime. This enhances scheduling accuracy and reduces pod scheduling failures due to changes in resource availability.

This is an alpha feature and is disabled by default.

To use this feature, you must enable the `MutableCSINodeAllocatableCount` feature gate on the following components

- `kube-apiserver`
- `kubelet`

# # # # Periodic Updates

When enabled, CSI drivers can request periodic updates to their volume limits by setting the `nodeAllocatableUpdatePeriodSeconds` field in the `CSIDriver` specification. For example

```yaml
apiVersion storage.k8s.iov1
kind CSIDriver
metadata
  name hostpath.csi.k8s.io
spec
  nodeAllocatableUpdatePeriodSeconds 60
```

Kubelet will periodically call the corresponding CSI drivers `NodeGetInfo` endpoint to refresh the maximum number of attachable volumes, using the interval specified in `nodeAllocatableUpdatePeriodSeconds`. The minimum allowed value for this field is 10 seconds.

Additionally, if a volume attachment operation fails with a `ResourceExhausted` error (gRPC code 8), Kubernetes triggers an immediate update to the allocatable volume count for that Node.
