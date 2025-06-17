---
title Compute, Storage, and Networking Extensions
weight 30
no_list true
---

This section covers extensions to your cluster that do not come as part as Kubernetes itself.
You can use these extensions to enhance the nodes in your cluster, or to provide the network
fabric that links Pods together.

* [CSI](docsconceptsstoragevolumes#csi) and [FlexVolume](docsconceptsstoragevolumes#flexvolume) storage plugins

   (CSI) plugins
  provide a way to extend Kubernetes with supports for new kinds of volumes. The volumes can
  be backed by durable external storage, or provide ephemeral storage, or they might offer a
  read-only interface to information using a filesystem paradigm.

  Kubernetes also includes support for [FlexVolume](docsconceptsstoragevolumes#flexvolume)
  plugins, which are deprecated since Kubernetes v1.23 (in favour of CSI).

  FlexVolume plugins allow users to mount volume types that arent natively
  supported by Kubernetes. When you run a Pod that relies on FlexVolume
  storage, the kubelet calls a binary plugin to mount the volume. The archived
  [FlexVolume](httpsgit.k8s.iodesign-proposals-archivestorageflexvolume-deployment.md)
  design proposal has more detail on this approach.

  The [Kubernetes Volume Plugin FAQ for Storage Vendors](httpsgithub.comkubernetescommunityblobmastersig-storagevolume-plugin-faq.md#kubernetes-volume-plugin-faq-for-storage-vendors)
  includes general information on storage plugins.

* [Device plugins](docsconceptsextend-kubernetescompute-storage-netdevice-plugins)

  Device plugins allow a node to discover new Node facilities (in addition to the
  built-in node resources such as `cpu` and `memory`), and provide these custom node-local
  facilities to Pods that request them.

* [Network plugins](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins)

  Network plugins allow Kubernetes to work with different networking topologies and technologies.
  Your Kubernetes cluster needs a _network plugin_ in order to have a working Pod network
  and to support other aspects of the Kubernetes network model.

  Kubernetes  is compatible with
  network plugins.
