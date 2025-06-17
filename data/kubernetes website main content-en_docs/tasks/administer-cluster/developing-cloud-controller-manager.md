---
reviewers
- luxas
- thockin
- wlan0
title Developing Cloud Controller Manager
content_type concept
weight 190
---

# # Background

Since cloud providers develop and release at a different pace compared to the Kubernetes project, abstracting the provider-specific code to the `cloud-controller-manager` binary allows cloud vendors to evolve independently from the core Kubernetes code.

The Kubernetes project provides skeleton cloud-controller-manager code with Go interfaces to allow you (or your cloud provider) to plug in your own implementations. This means that a cloud provider can implement a cloud-controller-manager by importing packages from Kubernetes core each cloudprovider will register their own code by calling `cloudprovider.RegisterCloudProvider` to update a global variable of available cloud providers.

# # Developing

# # # Out of tree

To build an out-of-tree cloud-controller-manager for your cloud

1. Create a go package with an implementation that satisfies [cloudprovider.Interface](httpsgithub.comkubernetescloud-providerblobmastercloud.go).
2. Use [`main.go` in cloud-controller-manager](httpsgithub.comkuberneteskubernetesblobmastercmdcloud-controller-managermain.go) from Kubernetes core as a template for your `main.go`. As mentioned above, the only difference should be the cloud package that will be imported.
3. Import your cloud package in `main.go`, ensure your package has an `init` block to run [`cloudprovider.RegisterCloudProvider`](httpsgithub.comkubernetescloud-providerblobmasterplugins.go).

Many cloud providers publish their controller manager code as open source. If you are creating
a new cloud-controller-manager from scratch, you could take an existing out-of-tree cloud
controller manager as your starting point.

# # # In tree

For in-tree cloud providers, you can run the in-tree cloud controller manager as a  in your cluster. See [Cloud Controller Manager Administration](docstasksadminister-clusterrunning-cloud-controller) for more details.
