---
title Gateway API
content_type concept
description -
  Gateway API is a family of API kinds that provide dynamic infrastructure provisioning
  and advanced traffic routing.
weight 55
---

Make network services available by using an extensible, role-oriented, protocol-aware configuration
mechanism. [Gateway API](httpsgateway-api.sigs.k8s.io) is an
containing API [kinds](httpsgateway-api.sigs.k8s.ioreferencesspec) that provide dynamic infrastructure
provisioning and advanced traffic routing.

# # Design principles

The following principles shaped the design and architecture of Gateway API

* __Role-oriented__ Gateway API kinds are modeled after organizational roles that are
  responsible for managing Kubernetes service networking
  * __Infrastructure Provider__ Manages infrastructure that allows multiple isolated clusters
    to serve multiple tenants, e.g. a cloud provider.
  * __Cluster Operator__ Manages clusters and is typically concerned with policies, network
    access, application permissions, etc.
  * __Application Developer__ Manages an application running in a cluster and is typically
    concerned with application-level configuration and [Service](docsconceptsservices-networkingservice)
    composition.
* __Portable__ Gateway API specifications are defined as [custom resources](docsconceptsextend-kubernetesapi-extensioncustom-resources)
  and are supported by many [implementations](httpsgateway-api.sigs.k8s.ioimplementations).
* __Expressive__ Gateway API kinds support functionality for common traffic routing use cases
  such as header-based matching, traffic weighting, and others that were only possible in
  [Ingress](docsconceptsservices-networkingingress) by using custom annotations.
* __Extensible__ Gateway allows for custom resources to be linked at various layers of the API.
  This makes granular customization possible at the appropriate places within the API structure.

# # Resource model

Gateway API has three stable API kinds

* __GatewayClass__ Defines a set of gateways with common configuration and managed by a controller
  that implements the class.

* __Gateway__ Defines an instance of traffic handling infrastructure, such as cloud load balancer.

* __HTTPRoute__ Defines HTTP-specific rules for mapping traffic from a Gateway listener to a
  representation of backend network endpoints. These endpoints are often represented as a
  .

Gateway API is organized into different API kinds that have interdependent relationships to support
the role-oriented nature of organizations. A Gateway object is associated with exactly one GatewayClass
the GatewayClass describes the gateway controller responsible for managing Gateways of this class.
One or more route kinds such as HTTPRoute, are then associated to Gateways. A Gateway can filter the routes
that may be attached to its `listeners`, forming a bidirectional trust model with routes.

The following figure illustrates the relationships of the three stable Gateway API kinds

# # # GatewayClass #api-kind-gateway-class

Gateways can be implemented by different controllers, often with different configurations. A Gateway
must reference a GatewayClass that contains the name of the controller that implements the
class.

A minimal GatewayClass example

```yaml
apiVersion gateway.networking.k8s.iov1
kind GatewayClass
metadata
  name example-class
spec
  controllerName example.comgateway-controller
```

In this example, a controller that has implemented Gateway API is configured to manage GatewayClasses
with the controller name `example.comgateway-controller`. Gateways of this class will be managed by
the implementations controller.

See the [GatewayClass](httpsgateway-api.sigs.k8s.ioreferencesspec#gateway.networking.k8s.iov1.GatewayClass)
reference for a full definition of this API kind.

# # # Gateway #api-kind-gateway

A Gateway describes an instance of traffic handling infrastructure. It defines a network endpoint
that can be used for processing traffic, i.e. filtering, balancing, splitting, etc. for backends
such as a Service. For example, a Gateway may represent a cloud load balancer or an in-cluster proxy
server that is configured to accept HTTP traffic.

A minimal Gateway resource example

```yaml
apiVersion gateway.networking.k8s.iov1
kind Gateway
metadata
  name example-gateway
spec
  gatewayClassName example-class
  listeners
  - name http
    protocol HTTP
    port 80
```

In this example, an instance of traffic handling infrastructure is programmed to listen for HTTP
traffic on port 80. Since the `addresses` field is unspecified, an address or hostname is assigned
to the Gateway by the implementations controller. This address is used as a network endpoint for
processing traffic of backend network endpoints defined in routes.

See the [Gateway](httpsgateway-api.sigs.k8s.ioreferencesspec#gateway.networking.k8s.iov1.Gateway)
reference for a full definition of this API kind.

# # # HTTPRoute #api-kind-httproute

The HTTPRoute kind specifies routing behavior of HTTP requests from a Gateway listener to backend network
endpoints. For a Service backend, an implementation may represent the backend network endpoint as a Service
IP or the backing EndpointSlices of the Service. An HTTPRoute represents configuration that is applied to the
underlying Gateway implementation. For example, defining a new HTTPRoute may result in configuring additional
traffic routes in a cloud load balancer or in-cluster proxy server.

A minimal HTTPRoute example

```yaml
apiVersion gateway.networking.k8s.iov1
kind HTTPRoute
metadata
  name example-httproute
spec
  parentRefs
  - name example-gateway
  hostnames
  - www.example.com
  rules
  - matches
    - path
        type PathPrefix
        value login
    backendRefs
    - name example-svc
      port 8080
```

In this example, HTTP traffic from Gateway `example-gateway` with the Host header set to `www.example.com`
and the request path specified as `login` will be routed to Service `example-svc` on port `8080`.

See the [HTTPRoute](httpsgateway-api.sigs.k8s.ioreferencesspec#gateway.networking.k8s.iov1.HTTPRoute)
reference for a full definition of this API kind.

# # Request flow

Here is a simple example of HTTP traffic being routed to a Service by using a Gateway and an HTTPRoute

In this example, the request flow for a Gateway implemented as a reverse proxy is

1. The client starts to prepare an HTTP request for the URL `httpwww.example.com`
2. The clients DNS resolver queries for the destination name and learns a mapping to
   one or more IP addresses associated with the Gateway.
3. The client sends a request to the Gateway IP address the reverse proxy receives the HTTP
   request and uses the Host header to match a configuration that was derived from the Gateway
   and attached HTTPRoute.
4. Optionally, the reverse proxy can perform request header andor path matching based
   on match rules of the HTTPRoute.
5. Optionally, the reverse proxy can modify the request for example, to add or remove headers,
   based on filter rules of the HTTPRoute.
6. Lastly, the reverse proxy forwards the request to one or more backends.

# # Conformance

Gateway API covers a broad set of features and is widely implemented. This combination requires
clear conformance definitions and tests to ensure that the API provides a consistent experience
wherever it is used.

See the [conformance](httpsgateway-api.sigs.k8s.ioconceptsconformance) documentation to
understand details such as release channels, support levels, and running conformance tests.

# # Migrating from Ingress

Gateway API is the successor to the [Ingress](docsconceptsservices-networkingingress) API.
However, it does not include the Ingress kind. As a result, a one-time conversion from your existing
Ingress resources to Gateway API resources is necessary.

Refer to the [ingress migration](httpsgateway-api.sigs.k8s.ioguidesmigrating-from-ingress#migrating-from-ingress)
guide for details on migrating Ingress resources to Gateway API resources.

# #  heading whatsnext

Instead of Gateway API resources being natively implemented by Kubernetes, the specifications
are defined as [Custom Resources](docsconceptsextend-kubernetesapi-extensioncustom-resources)
supported by a wide range of [implementations](httpsgateway-api.sigs.k8s.ioimplementations).
[Install](httpsgateway-api.sigs.k8s.ioguides#installing-gateway-api) the Gateway API CRDs or
follow the installation instructions of your selected implementation. After installing an
implementation, use the [Getting Started](httpsgateway-api.sigs.k8s.ioguides) guide to help
you quickly start working with Gateway API.

Make sure to review the documentation of your selected implementation to understand any caveats.

Refer to the [API specification](httpsgateway-api.sigs.k8s.ioreferencespec) for additional
details of all Gateway API kinds.
