---
reviewers
- brendandburns
- erictune
- mikedanese
title Getting started
main_menu true
weight 20
content_type concept
no_list true
card
  name setup
  weight 20
  anchors
  - anchor #learning-environment
    title Learning environment
  - anchor #production-environment
    title Production environment
---

This section lists the different ways to set up and run Kubernetes.
When you install Kubernetes, choose an installation type based on ease of maintenance, security,
control, available resources, and expertise required to operate and manage a cluster.

You can [download Kubernetes](releasesdownload) to deploy a Kubernetes cluster
on a local machine, into the cloud, or for your own datacenter.

Several [Kubernetes components](docsconceptsoverviewcomponents) such as  or  can also be
deployed as [container images](releasesdownload#container-images) within the cluster.

It is **recommended** to run Kubernetes components as container images wherever
that is possible, and to have Kubernetes manage those components.
Components that run containers - notably, the kubelet - cant be included in this category.

If you dont want to manage a Kubernetes cluster yourself, you could pick a managed service, including
[certified platforms](docssetupproduction-environmentturnkey-solutions).
There are also other standardized and custom solutions across a wide range of cloud and
bare metal environments.

# # Learning environment

If youre learning Kubernetes, use the tools supported by the Kubernetes community,
or tools in the ecosystem to set up a Kubernetes cluster on a local machine.
See [Install tools](docstaskstools).

# # Production environment

When evaluating a solution for a
[production environment](docssetupproduction-environment), consider which aspects of
operating a Kubernetes cluster (or _abstractions_) you want to manage yourself and which you
prefer to hand off to a provider.

For a cluster youre managing yourself, the officially supported tool
for deploying Kubernetes is [kubeadm](docssetupproduction-environmenttoolskubeadm).

# #  heading whatsnext

- [Download Kubernetes](releasesdownload)
- Download and [install tools](docstaskstools) including `kubectl`
- Select a [container runtime](docssetupproduction-environmentcontainer-runtimes) for your new cluster
- Learn about [best practices](docssetupbest-practices) for cluster setup

Kubernetes is designed for its  to
run on Linux. Within your cluster you can run applications on Linux or other operating systems, including
Windows.

- Learn to [set up clusters with Windows nodes](docsconceptswindows)
