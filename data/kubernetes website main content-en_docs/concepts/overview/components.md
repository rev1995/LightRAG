---
reviewers
- lavalamp
title Kubernetes Components
content_type concept
description
  An overview of the key components that make up a Kubernetes cluster.
weight 10
card
  title Components of a cluster
  name concepts
  weight 20
---

This page provides a high-level overview of the essential components that make up a Kubernetes cluster.

# # Core Components

A Kubernetes cluster consists of a control plane and one or more worker nodes.
Heres a brief overview of the main components

# # # Control Plane Components

Manage the overall state of the cluster

[kube-apiserver](docsconceptsarchitecture#kube-apiserver)
 The core component server that exposes the Kubernetes HTTP API.

[etcd](docsconceptsarchitecture#etcd)
 Consistent and highly-available key value store for all API server data.

[kube-scheduler](docsconceptsarchitecture#kube-scheduler)
 Looks for Pods not yet bound to a node, and assigns each Pod to a suitable node.

[kube-controller-manager](docsconceptsarchitecture#kube-controller-manager)
 Runs  to implement Kubernetes API behavior.

[cloud-controller-manager](docsconceptsarchitecture#cloud-controller-manager) (optional)
 Integrates with underlying cloud provider(s).

# # # Node Components

Run on every node, maintaining running pods and providing the Kubernetes runtime environment

[kubelet](docsconceptsarchitecture#kubelet)
 Ensures that Pods are running, including their containers.

[kube-proxy](docsconceptsarchitecture#kube-proxy) (optional)
 Maintains network rules on nodes to implement .

[Container runtime](docsconceptsarchitecture#container-runtime)
 Software responsible for running containers. Read
  [Container Runtimes](docssetupproduction-environmentcontainer-runtimes) to learn more.

 thirdparty-content singletrue

Your cluster may require additional software on each node for example, you might also
run [systemd](httpssystemd.io) on a Linux node to supervise local components.

# # Addons

Addons extend the functionality of Kubernetes. A few important examples include

[DNS](docsconceptsarchitecture#dns)
 For cluster-wide DNS resolution.

[Web UI](docsconceptsarchitecture#web-ui-dashboard) (Dashboard)
 For cluster management via a web interface.

[Container Resource Monitoring](docsconceptsarchitecture#container-resource-monitoring)
 For collecting and storing container metrics.

[Cluster-level Logging](docsconceptsarchitecture#cluster-level-logging)
 For saving container logs to a central log store.

# # Flexibility in Architecture

Kubernetes allows for flexibility in how these components are deployed and managed.
The architecture can be adapted to various needs, from small development environments
to large-scale production deployments.

For more detailed information about each component and various ways to configure your
cluster architecture, see the [Cluster Architecture](docsconceptsarchitecture) page.
