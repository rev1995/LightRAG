---
title Migrating telemetry and security agents from dockershim
content_type task
reviewers
- SergeyKanzhelev
weight 60
---

 thirdparty-content

Kubernetes support for direct integration with Docker Engine is deprecated and
has been removed. Most apps do not have a direct dependency on runtime hosting
containers. However, there are still a lot of telemetry and monitoring agents
that have a dependency on Docker to collect containers metadata, logs, and
metrics. This document aggregates information on how to detect these
dependencies as well as links on how to migrate these agents to use generic tools or
alternative runtimes.

# # Telemetry and security agents

Within a Kubernetes cluster there are a few different ways to run telemetry or
security agents.  Some agents have a direct dependency on Docker Engine when
they run as DaemonSets or directly on nodes.

# # # Why do some telemetry agents communicate with Docker Engine

Historically, Kubernetes was written to work specifically with Docker Engine.
Kubernetes took care of networking and scheduling, relying on Docker Engine for
launching and running containers (within Pods) on a node. Some information that
is relevant to telemetry, such as a pod name, is only available from Kubernetes
components. Other data, such as container metrics, is not the responsibility of
the container runtime. Early telemetry agents needed to query the container
runtime *and* Kubernetes to report an accurate picture. Over time, Kubernetes
gained the ability to support multiple runtimes, and now supports any runtime
that is compatible with the [container runtime interface](docsconceptsarchitecturecri).

Some telemetry agents rely specifically on Docker Engine tooling. For example, an agent
might run a command such as
[`docker ps`](httpsdocs.docker.comenginereferencecommandlineps)
or [`docker top`](httpsdocs.docker.comenginereferencecommandlinetop) to list
containers and processes or [`docker logs`](httpsdocs.docker.comenginereferencecommandlinelogs)
to receive streamed logs. If nodes in your existing cluster use
Docker Engine, and you switch to a different container runtime,
these commands will not work any longer.

# # # Identify DaemonSets that depend on Docker Engine #identify-docker-dependency

If a pod wants to make calls to the `dockerd` running on the node, the pod must either

- mount the filesystem containing the Docker daemons privileged socket, as a
   or
- mount the specific path of the Docker daemons privileged socket directly, also as a volume.

For example on COS images, Docker exposes its Unix domain socket at
`varrundocker.sock` This means that the pod spec will include a
`hostPath` volume mount of `varrundocker.sock`.

Heres a sample shell script to find Pods that have a mount directly mapping the
Docker socket. This script outputs the namespace and name of the pod. You can
remove the `grep varrundocker.sock` to review other mounts.

```bash
kubectl get pods --all-namespaces
-ojsonpathrange .items[*]n.metadata.namespacet.metadata.nametrange .spec.volumes[*].hostPath.path, endend
 sort
 grep varrundocker.sock
```

There are alternative ways for a pod to access Docker on the host. For instance, the parent
directory `varrun` may be mounted instead of the full path (like in [this
example](httpsgist.github.comitaysk7bc3e56d69c4d72a549286d98fd557dd)).
The script above only detects the most common uses.

# # # Detecting Docker dependency from node agents

If your cluster nodes are customized and install additional security and
telemetry agents on the node, check with the agent vendor
to verify whether it has any dependency on Docker.

# # # Telemetry and security agent vendors

This section is intended to aggregate information about various telemetry and
security agents that may have a dependency on container runtimes.

We keep the work in progress version of migration instructions for various telemetry and security agent vendors
in [Google doc](httpsdocs.google.comdocumentd1ZFi4uKit63ga5sxEiZblfb-c23lFhvy6RXVPikS8wf0edit#).
Please contact the vendor to get up to date instructions for migrating from dockershim.

# # Migration from dockershim

# # # [Aqua](httpswww.aquasec.com)

No changes are needed everything should work seamlessly on the runtime switch.

# # # [Datadog](httpswww.datadoghq.comproduct)

How to migrate
[Docker deprecation in Kubernetes](httpsdocs.datadoghq.comagentguidedocker-deprecation)
The pod that accesses Docker Engine may have a name containing any of

- `datadog-agent`
- `datadog`
- `dd-agent`

# # # [Dynatrace](httpswww.dynatrace.com)

How to migrate
[Migrating from Docker-only to generic container metrics in Dynatrace](httpscommunity.dynatrace.comt5Best-practicesMigrating-from-Docker-only-to-generic-container-metrics-inm-p167030#M49)

Containerd support announcement [Get automated full-stack visibility into
containerd-based Kubernetes
environments](httpswww.dynatrace.comnewsblogget-automated-full-stack-visibility-into-containerd-based-kubernetes-environments)

CRI-O support announcement [Get automated full-stack visibility into your CRI-O Kubernetes containers (Beta)](httpswww.dynatrace.comnewsblogget-automated-full-stack-visibility-into-your-cri-o-kubernetes-containers-beta)

The pod accessing Docker may have name containing
- `dynatrace-oneagent`

# # # [Falco](httpsfalco.org)

How to migrate

[Migrate Falco from dockershim](httpsfalco.orgdocsgetting-starteddeployment#docker-deprecation-in-kubernetes)
Falco supports any CRI-compatible runtime (containerd is used in the default configuration) the documentation explains all details.
The pod accessing Docker may have name containing
- `falco`

# # # [Prisma Cloud Compute](httpsdocs.paloaltonetworks.comprismaprisma-cloud.html)

Check [documentation for Prisma Cloud](httpsdocs.paloaltonetworks.comprismaprisma-cloudprisma-cloud-admin-computeinstallinstall_kubernetes.html),
under the Install Prisma Cloud on a CRI (non-Docker) cluster section.
The pod accessing Docker may be named like
- `twistlock-defender-ds`

# # # [SignalFx (Splunk)](httpswww.splunk.comen_usinvestor-relationsacquisitionssignalfx.html)

The SignalFx Smart Agent (deprecated) uses several different monitors for Kubernetes including
`kubernetes-cluster`, `kubelet-statskubelet-metrics`, and `docker-container-stats`.
The `kubelet-stats` monitor was previously deprecated by the vendor, in favor of `kubelet-metrics`.
The `docker-container-stats` monitor is the one affected by dockershim removal.
Do not use the `docker-container-stats` with container runtimes other than Docker Engine.

How to migrate from dockershim-dependent agent
1. Remove `docker-container-stats` from the list of [configured monitors](httpsgithub.comsignalfxsignalfx-agentblobmaindocsmonitor-config.md).
   Note, keeping this monitor enabled with non-dockershim runtime will result in incorrect metrics
   being reported when docker is installed on node and no metrics when docker is not installed.
2. [Enable and configure `kubelet-metrics`](httpsgithub.comsignalfxsignalfx-agentblobmaindocsmonitorskubelet-metrics.md) monitor.

The set of collected metrics will change. Review your alerting rules and dashboards.

The Pod accessing Docker may be named something like

- `signalfx-agent`

# # # Yahoo Kubectl Flame

Flame does not support container runtimes other than Docker. See
[httpsgithub.comyahookubectl-flameissues51](httpsgithub.comyahookubectl-flameissues51)
