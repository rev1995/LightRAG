---
reviewers
- fgrzadkowski
- piosz
title Resource metrics pipeline
content_type concept
weight 15
---

For Kubernetes, the _Metrics API_ offers a basic set of metrics to support automatic scaling and
similar use cases.  This API makes information available about resource usage for node and pod,
including metrics for CPU and memory.  If you deploy the Metrics API into your cluster, clients of
the Kubernetes API can then query for this information, and you can use Kubernetes access control
mechanisms to manage permissions to do so.

The [HorizontalPodAutoscaler](docstasksrun-applicationhorizontal-pod-autoscale)  (HPA) and
[VerticalPodAutoscaler](httpsgithub.comkubernetesautoscalertreemastervertical-pod-autoscaler#readme) (VPA)
use data from the metrics API to adjust workload replicas and resources to meet customer demand.

You can also view the resource metrics using the
[`kubectl top`](docsreferencegeneratedkubectlkubectl-commands#top)
command.

The Metrics API, and the metrics pipeline that it enables, only offers the minimum
CPU and memory metrics to enable automatic scaling using HPA and  or VPA.
If you would like to provide a more complete set of metrics, you can complement
the simpler Metrics API by deploying a second
[metrics pipeline](docstasksdebugdebug-clusterresource-usage-monitoring#full-metrics-pipeline)
that uses the _Custom Metrics API_.

Figure 1 illustrates the architecture of the resource metrics pipeline.

flowchart RL
subgraph cluster[Cluster]
direction RL
S[  ]
A[Metrics-Server]
subgraph B[Nodes]
direction TB
D[cAdvisor] -- C[kubelet]
E[Containerruntime] -- D
E1[Containerruntime] -- D
P[pod data] -.- C
end
L[APIserver]
W[HPA]
C ----node levelresource metrics A --metricsAPI L -- W
end
L --- K[kubectltop]
classDef box fill#fff,stroke#000,stroke-width1px,color#000
class W,B,P,K,cluster,D,E,E1 box
classDef spacewhite fill#ffffff,stroke#fff,stroke-width0px,color#000
class S spacewhite
classDef k8s fill#326ce5,stroke#fff,stroke-width1px,color#fff
class A,L,C k8s

Figure 1. Resource Metrics Pipeline

The architecture components, from right to left in the figure, consist of the following

* [cAdvisor](httpsgithub.comgooglecadvisor) Daemon for collecting, aggregating and exposing
  container metrics included in Kubelet.
* [kubelet](docsconceptsarchitecture#kubelet) Node agent for managing container
  resources. Resource metrics are accessible using the `metricsresource` and `stats` kubelet
  API endpoints.
* [node level resource metrics](docsreferenceinstrumentationnode-metrics) API provided by the kubelet for discovering and retrieving
  per-node summarized stats available through the `metricsresource` endpoint.
* [metrics-server](#metrics-server) Cluster addon component that collects and aggregates resource
  metrics pulled from each kubelet. The API server serves Metrics API for use by HPA, VPA, and by
  the `kubectl top` command. Metrics Server is a reference implementation of the Metrics API.
* [Metrics API](#metrics-api) Kubernetes API supporting access to CPU and memory used for
  workload autoscaling. To make this work in your cluster, you need an API extension server that
  provides the Metrics API.

  cAdvisor supports reading metrics from cgroups, which works with typical container runtimes on Linux.
  If you use a container runtime that uses another resource isolation mechanism, for example
  virtualization, then that container runtime must support
  [CRI Container Metrics](httpsgithub.comkubernetescommunityblobmastercontributorsdevelsig-nodecri-container-stats.md)
  in order for metrics to be available to the kubelet.

# # Metrics API

The metrics-server implements the Metrics API. This API allows you to access CPU and memory usage
for the nodes and pods in your cluster. Its primary role is to feed resource usage metrics to K8s
autoscaler components.

Here is an example of the Metrics API request for a `minikube` node piped through `jq` for easier
reading

```shell
kubectl get --raw apismetrics.k8s.iov1beta1nodesminikube  jq .
```

Here is the same API call using `curl`

```shell
curl httplocalhost8080apismetrics.k8s.iov1beta1nodesminikube
```

Sample response

```json

  kind NodeMetrics,
  apiVersion metrics.k8s.iov1beta1,
  metadata
    name minikube,
    selfLink apismetrics.k8s.iov1beta1nodesminikube,
    creationTimestamp 2022-01-27T184843Z
  ,
  timestamp 2022-01-27T184833Z,
  window 30s,
  usage
    cpu 487558164n,
    memory 732212Ki

```

Here is an example of the Metrics API request for a `kube-scheduler-minikube` pod contained in the
`kube-system` namespace and piped through `jq` for easier reading

```shell
kubectl get --raw apismetrics.k8s.iov1beta1namespaceskube-systempodskube-scheduler-minikube  jq .
```

Here is the same API call using `curl`

```shell
curl httplocalhost8080apismetrics.k8s.iov1beta1namespaceskube-systempodskube-scheduler-minikube
```

Sample response

```json

  kind PodMetrics,
  apiVersion metrics.k8s.iov1beta1,
  metadata
    name kube-scheduler-minikube,
    namespace kube-system,
    selfLink apismetrics.k8s.iov1beta1namespaceskube-systempodskube-scheduler-minikube,
    creationTimestamp 2022-01-27T192500Z
  ,
  timestamp 2022-01-27T192431Z,
  window 30s,
  containers [

      name kube-scheduler,
      usage
        cpu 9559630n,
        memory 22244Ki

  ]

```

The Metrics API is defined in the [k8s.iometrics](httpsgithub.comkubernetesmetrics)
repository. You must enable the [API aggregation layer](docstasksextend-kubernetesconfigure-aggregation-layer)
and register an [APIService](docsreferencekubernetes-apicluster-resourcesapi-service-v1)
for the `metrics.k8s.io` API.

To learn more about the Metrics API, see [resource metrics API design](httpsgit.k8s.iodesign-proposals-archiveinstrumentationresource-metrics-api.md),
the [metrics-server repository](httpsgithub.comkubernetes-sigsmetrics-server) and the
[resource metrics API](httpsgithub.comkubernetesmetrics#resource-metrics-api).

You must deploy the metrics-server or alternative adapter that serves the Metrics API to be able
to access it.

# # Measuring resource usage

# # # CPU

CPU is reported as the average core usage measured in cpu units. One cpu, in Kubernetes, is
equivalent to 1 vCPUCore for cloud providers, and 1 hyper-thread on bare-metal Intel processors.

This value is derived by taking a rate over a cumulative CPU counter provided by the kernel (in
both Linux and Windows kernels). The time window used to calculate CPU is shown under window field
in Metrics API.

To learn more about how Kubernetes allocates and measures CPU resources, see
[meaning of CPU](docsconceptsconfigurationmanage-resources-containers#meaning-of-cpu).

# # # Memory

Memory is reported as the working set, measured in bytes, at the instant the metric was collected.

In an ideal world, the working set is the amount of memory in-use that cannot be freed under
memory pressure. However, calculation of the working set varies by host OS, and generally makes
heavy use of heuristics to produce an estimate.

The Kubernetes model for a containers working set expects that the container runtime counts
anonymous memory associated with the container in question. The working set metric typically also
includes some cached (file-backed) memory, because the host OS cannot always reclaim pages.

To learn more about how Kubernetes allocates and measures memory resources, see
[meaning of memory](docsconceptsconfigurationmanage-resources-containers#meaning-of-memory).

# # Metrics Server

The metrics-server fetches resource metrics from the kubelets and exposes them in the Kubernetes
API server through the Metrics API for use by the HPA and VPA. You can also view these metrics
using the `kubectl top` command.

The metrics-server uses the Kubernetes API to track nodes and pods in your cluster. The
metrics-server queries each node over HTTP to fetch metrics. The metrics-server also builds an
internal view of pod metadata, and keeps a cache of pod health. That cached pod health information
is available via the extension API that the metrics-server makes available.

For example with an HPA query, the metrics-server needs to identify which pods fulfill the label
selectors in the deployment.

The metrics-server calls the [kubelet](docsreferencecommand-line-tools-referencekubelet) API
to collect metrics from each node. Depending on the metrics-server version it uses

* Metrics resource endpoint `metricsresource` in version v0.6.0 or
* Summary API endpoint `statssummary` in older versions

# #  heading whatsnext

To learn more about the metrics-server, see the
[metrics-server repository](httpsgithub.comkubernetes-sigsmetrics-server).

You can also check out the following

* [metrics-server design](httpsgit.k8s.iodesign-proposals-archiveinstrumentationmetrics-server.md)
* [metrics-server FAQ](httpsgithub.comkubernetes-sigsmetrics-serverblobmasterFAQ.md)
* [metrics-server known issues](httpsgithub.comkubernetes-sigsmetrics-serverblobmasterKNOWN_ISSUES.md)
* [metrics-server releases](httpsgithub.comkubernetes-sigsmetrics-serverreleases)
* [Horizontal Pod Autoscaling](docstasksrun-applicationhorizontal-pod-autoscale)

To learn about how the kubelet serves node metrics, and how you can access those via
the Kubernetes API, read [Node Metrics Data](docsreferenceinstrumentationnode-metrics).
