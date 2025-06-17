---
reviewers
- maplain
title Service Internal Traffic Policy
content_type concept
weight 120
description -
  If two Pods in your cluster want to communicate, and both Pods are actually running on
  the same node, use _Service Internal Traffic Policy_ to keep network traffic within that node.
  Avoiding a round trip via the cluster network can help with reliability, performance
  (network latency and throughput), or cost.
---

_Service Internal Traffic Policy_ enables internal traffic restrictions to only route
internal traffic to endpoints within the node the traffic originated from. The
internal traffic here refers to traffic originated from Pods in the current
cluster. This can help to reduce costs and improve performance.

# # Using Service Internal Traffic Policy

You can enable the internal-only traffic policy for a
, by setting its
`.spec.internalTrafficPolicy` to `Local`. This tells kube-proxy to only use node local
endpoints for cluster internal traffic.

For pods on nodes with no endpoints for a given Service, the Service
behaves as if it has zero endpoints (for Pods on this node) even if the service
does have endpoints on other nodes.

The following example shows what a Service looks like when you set
`.spec.internalTrafficPolicy` to `Local`

```yaml
apiVersion v1
kind Service
metadata
  name my-service
spec
  selector
    app.kubernetes.ioname MyApp
  ports
    - protocol TCP
      port 80
      targetPort 9376
  internalTrafficPolicy Local
```

# # How it works

The kube-proxy filters the endpoints it routes to based on the
`spec.internalTrafficPolicy` setting. When its set to `Local`, only node local
endpoints are considered. When its `Cluster` (the default), or is not set,
Kubernetes considers all endpoints.

# #  heading whatsnext

* Read about [Topology Aware Routing](docsconceptsservices-networkingtopology-aware-routing)
* Read about [Service External Traffic Policy](docstasksaccess-application-clustercreate-external-load-balancer#preserving-the-client-source-ip)
* Follow the [Connecting Applications with Services](docstutorialsservicesconnect-applications-service) tutorial
