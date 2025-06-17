---
title Accessing the Kubernetes API from a Pod
content_type task
weight 120
---

This guide demonstrates how to access the Kubernetes API from within a pod.

# #  heading prerequisites

# # Accessing the API from within a Pod

When accessing the API from within a Pod, locating and authenticating
to the API server are slightly different to the external client case.

The easiest way to use the Kubernetes API from a Pod is to use
one of the official [client libraries](docsreferenceusing-apiclient-libraries). These
libraries can automatically discover the API server and authenticate.

# # # Using Official Client Libraries

From within a Pod, the recommended ways to connect to the Kubernetes API are

- For a Go client, use the official
  [Go client library](httpsgithub.comkubernetesclient-go).
  The `rest.InClusterConfig()` function handles API host discovery and authentication automatically.
  See [an example here](httpsgit.k8s.ioclient-goexamplesin-cluster-client-configurationmain.go).

- For a Python client, use the official
  [Python client library](httpsgithub.comkubernetes-clientpython).
  The `config.load_incluster_config()` function handles API host discovery and authentication automatically.
  See [an example here](httpsgithub.comkubernetes-clientpythonblobmasterexamplesin_cluster_config.py).

- There are a number of other libraries available, please refer to the
  [Client Libraries](docsreferenceusing-apiclient-libraries) page.

In each case, the service account credentials of the Pod are used to communicate
securely with the API server.

# # # Directly accessing the REST API

While running in a Pod, your container can create an HTTPS URL for the Kubernetes API
server by fetching the `KUBERNETES_SERVICE_HOST` and `KUBERNETES_SERVICE_PORT_HTTPS`
environment variables. The API servers in-cluster address is also published to a
Service named `kubernetes` in the `default` namespace so that pods may reference
`kubernetes.default.svc` as a DNS name for the local API server.

Kubernetes does not guarantee that the API server has a valid certificate for
the hostname `kubernetes.default.svc`
however, the control plane **is** expected to present a valid certificate for the
hostname or IP address that `KUBERNETES_SERVICE_HOST` represents.

The recommended way to authenticate to the API server is with a
[service account](docstasksconfigure-pod-containerconfigure-service-account)
credential. By default, a Pod
is associated with a service account, and a credential (token) for that
service account is placed into the filesystem tree of each container in that Pod,
at `varrunsecretskubernetes.ioserviceaccounttoken`.

If available, a certificate bundle is placed into the filesystem tree of each
container at `varrunsecretskubernetes.ioserviceaccountca.crt`, and should be
used to verify the serving certificate of the API server.

Finally, the default namespace to be used for namespaced API operations is placed in a file
at `varrunsecretskubernetes.ioserviceaccountnamespace` in each container.

# # # Using kubectl proxy

If you would like to query the API without an official client library, you can run `kubectl proxy`
as the [command](docstasksinject-data-applicationdefine-command-argument-container)
of a new sidecar container in the Pod. This way, `kubectl proxy` will authenticate
to the API and expose it on the `localhost` interface of the Pod, so that other containers
in the Pod can use it directly.

# # # Without using a proxy

It is possible to avoid using the kubectl proxy by passing the authentication token
directly to the API server. The internal certificate secures the connection.

```shell
# Point to the internal API server hostname
APISERVERhttpskubernetes.default.svc

# Path to ServiceAccount token
SERVICEACCOUNTvarrunsecretskubernetes.ioserviceaccount

# Read this Pods namespace
NAMESPACE(cat SERVICEACCOUNTnamespace)

# Read the ServiceAccount bearer token
TOKEN(cat SERVICEACCOUNTtoken)

# Reference the internal certificate authority (CA)
CACERTSERVICEACCOUNTca.crt

# Explore the API with TOKEN
curl --cacert CACERT --header Authorization Bearer TOKEN -X GET APISERVERapi
```

The output will be similar to this

```json

  kind APIVersions,
  versions [v1],
  serverAddressByClientCIDRs [

      clientCIDR 0.0.0.00,
      serverAddress 10.0.1.149443

  ]

```
