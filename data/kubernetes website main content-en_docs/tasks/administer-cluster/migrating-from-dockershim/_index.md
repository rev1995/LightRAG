---
title Migrating from dockershim
weight 20
content_type task
no_list true
---

This section presents information you need to know when migrating from
dockershim to other container runtimes.

Since the announcement of [dockershim deprecation](blog20201208kubernetes-1-20-release-announcement#dockershim-deprecation)
in Kubernetes 1.20, there were questions on how this will affect various workloads and Kubernetes
installations. Our [Dockershim Removal FAQ](blog20220217dockershim-faq) is there to help you
to understand the problem better.

Dockershim was removed from Kubernetes with the release of v1.24.
If you use Docker Engine via dockershim as your container runtime and wish to upgrade to v1.24,
it is recommended that you either migrate to another runtime or find an alternative means to obtain Docker Engine support.
Check out the [container runtimes](docssetupproduction-environmentcontainer-runtimes)
section to know your options.

The version of Kubernetes with dockershim (1.23) is out of support and the v1.24
will run out of support [soon](releases#release-v1-24). Make sure to
[report issues](httpsgithub.comkuberneteskubernetesissues) you encountered
with the migration so the issues can be fixed in a timely manner and your cluster would be
ready for dockershim removal. After v1.24 running out of support, you will need
to contact your Kubernetes provider for support or upgrade multiple versions at a time
if there are critical issues affecting your cluster.

Your cluster might have more than one kind of node, although this is not a common
configuration.

These tasks will help you to migrate

* [Check whether Dockershim removal affects you](docstasksadminister-clustermigrating-from-dockershimcheck-if-dockershim-removal-affects-you)
* [Migrating telemetry and security agents from dockershim](docstasksadminister-clustermigrating-from-dockershimmigrating-telemetry-and-security-agents)

# #  heading whatsnext

* Check out [container runtimes](docssetupproduction-environmentcontainer-runtimes)
  to understand your options for an alternative.
* If you find a defect or other technical concern relating to migrating away from dockershim,
  you can [report an issue](httpsgithub.comkuberneteskubernetesissuesnewchoose)
  to the Kubernetes project.
