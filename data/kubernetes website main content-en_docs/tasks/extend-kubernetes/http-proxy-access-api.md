---
title Use an HTTP Proxy to Access the Kubernetes API
content_type task
weight 40
---

This page shows how to use an HTTP proxy to access the Kubernetes API.

# #  heading prerequisites

If you do not already have an application running in your cluster, start
a Hello world application by entering this command

```shell
kubectl create deployment hello-app --imagegcr.iogoogle-sampleshello-app2.0 --port8080
```

# # Using kubectl to start a proxy server

This command starts a proxy to the Kubernetes API server

    kubectl proxy --port8080

# # Exploring the Kubernetes API

When the proxy server is running, you can explore the API using `curl`, `wget`,
or a browser.

Get the API versions

    curl httplocalhost8080api

The output should look similar to this

      kind APIVersions,
      versions [
        v1
      ],
      serverAddressByClientCIDRs [

          clientCIDR 0.0.0.00,
          serverAddress 10.0.2.158443

      ]

Get a list of pods

    curl httplocalhost8080apiv1namespacesdefaultpods

The output should look similar to this

      kind PodList,
      apiVersion v1,
      metadata
        resourceVersion 33074
      ,
      items [

          metadata
            name kubernetes-bootcamp-2321272333-ix8pt,
            generateName kubernetes-bootcamp-2321272333-,
            namespace default,
            uid ba21457c-6b1d-11e6-85f7-1ef9f1dab92b,
            resourceVersion 33003,
            creationTimestamp 2016-08-25T234330Z,
            labels
              pod-template-hash 2321272333,
              run kubernetes-bootcamp
            ,
            ...

# #  heading whatsnext

Learn more about [kubectl proxy](docsreferencegeneratedkubectlkubectl-commands#proxy).
