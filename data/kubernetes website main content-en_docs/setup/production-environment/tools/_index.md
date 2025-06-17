---
title Installing Kubernetes with deployment tools
weight 30
no_list true
---

There are many methods and tools for setting up your own production Kubernetes cluster.
For example

- [kubeadm](docssetupproduction-environmenttoolskubeadm)

- [Cluster API](httpscluster-api.sigs.k8s.io) A Kubernetes sub-project focused on
  providing declarative APIs and tooling to simplify provisioning, upgrading, and operating
  multiple Kubernetes clusters.

- [kops](httpskops.sigs.k8s.io) An automated cluster provisioning tool.
  For tutorials, best practices, configuration options  and information on
  reaching out to the community, please check the
  [`kOps` website](httpskops.sigs.k8s.io) for details.

- [kubespray](httpskubespray.io)
  A composition of [Ansible](httpsdocs.ansible.com) playbooks,
  [inventory](httpsgithub.comkubernetes-sigskubesprayblobmasterdocsansibleinventory.md),
  provisioning tools, and domain knowledge for generic OSKubernetes clusters configuration
  management tasks. You can reach out to the community on Slack channel
  [#kubespray](httpskubernetes.slack.commessageskubespray).
