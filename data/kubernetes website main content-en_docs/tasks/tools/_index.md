---
title Install Tools
description Set up Kubernetes tools on your computer.
weight 10
no_list true
card
  name tasks
  weight 20
  anchors
  - anchor #kubectl
    title Install kubectl
---

# # kubectl

The Kubernetes command-line tool, [kubectl](docsreferencekubectlkubectl), allows
you to run commands against Kubernetes clusters.
You can use kubectl to deploy applications, inspect and manage cluster resources,
and view logs. For more information including a complete list of kubectl operations, see the
[`kubectl` reference documentation](docsreferencekubectl).

kubectl is installable on a variety of Linux platforms, macOS and Windows.
Find your preferred operating system below.

- [Install kubectl on Linux](docstaskstoolsinstall-kubectl-linux)
- [Install kubectl on macOS](docstaskstoolsinstall-kubectl-macos)
- [Install kubectl on Windows](docstaskstoolsinstall-kubectl-windows)

# # kind

[`kind`](httpskind.sigs.k8s.io) lets you run Kubernetes on
your local computer. This tool requires that you have either
[Docker](httpswww.docker.com) or [Podman](httpspodman.io) installed.

The kind [Quick Start](httpskind.sigs.k8s.iodocsuserquick-start) page
shows you what you need to do to get up and running with kind.

View kind Quick Start Guide

# # minikube

Like `kind`, [`minikube`](httpsminikube.sigs.k8s.io) is a tool that lets you run Kubernetes
locally. `minikube` runs an all-in-one or a multi-node local Kubernetes cluster on your personal
computer (including Windows, macOS and Linux PCs) so that you can try out
Kubernetes, or for daily development work.

You can follow the official
[Get Started!](httpsminikube.sigs.k8s.iodocsstart) guide if your focus is
on getting the tool installed.

View minikube Get Started! Guide

Once you have `minikube` working, you can use it to
[run a sample application](docstutorialshello-minikube).

# # kubeadm

You can use the  tool to create and manage Kubernetes clusters.
It performs the actions necessary to get a minimum viable, secure cluster up and running in a user friendly way.

[Installing kubeadm](docssetupproduction-environmenttoolskubeadminstall-kubeadm) shows you how to install kubeadm.
Once installed, you can use it to [create a cluster](docssetupproduction-environmenttoolskubeadmcreate-cluster-kubeadm).

View kubeadm Install Guide
