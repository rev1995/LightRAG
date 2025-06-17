---
title Ingress Controllers
description -
  In order for an [Ingress](docsconceptsservices-networkingingress) to work in your cluster,
  there must be an _ingress controller_ running.
  You need to select at least one ingress controller and make sure it is set up in your cluster.
  This page lists common ingress controllers that you can deploy.
content_type concept
weight 50
---

In order for the Ingress resource to work, the cluster must have an ingress controller running.

Unlike other types of controllers which run as part of the `kube-controller-manager` binary, Ingress controllers
are not started automatically with a cluster. Use this page to choose the ingress controller implementation
that best fits your cluster.

Kubernetes as a project supports and maintains [AWS](httpsgithub.comkubernetes-sigsaws-load-balancer-controller#readme), [GCE](httpsgit.k8s.ioingress-gceREADME.md#readme), and
  [nginx](httpsgit.k8s.ioingress-nginxREADME.md#readme) ingress controllers.

# # Additional controllers

 thirdparty-content

* [AKS Application Gateway Ingress Controller](httpsdocs.microsoft.comazureapplication-gatewaytutorial-ingress-controller-add-on-existingtochttps3A2F2Fdocs.microsoft.com2Fen-us2Fazure2Faks2Ftoc.jsonbchttps3A2F2Fdocs.microsoft.com2Fen-us2Fazure2Fbread2Ftoc.json) is an ingress controller that configures the [Azure Application Gateway](httpsdocs.microsoft.comazureapplication-gatewayoverview).
* [Alibaba Cloud MSE Ingress](httpswww.alibabacloud.comhelpenmseuser-guideoverview-of-mse-ingress-gateways) is an ingress controller that configures the [Alibaba Cloud Native Gateway](httpswww.alibabacloud.comhelpenmseproduct-overviewcloud-native-gateway-overviewspma2c63.p38356.0.0.20563003HJK9is), which is also the commercial version of [Higress](httpsgithub.comalibabahigress).
* [Apache APISIX ingress controller](httpsgithub.comapacheapisix-ingress-controller) is an [Apache APISIX](httpsgithub.comapacheapisix)-based ingress controller.
* [Avi Kubernetes Operator](httpsgithub.comvmwareload-balancer-and-ingress-services-for-kubernetes) provides L4-L7 load-balancing using [VMware NSX Advanced Load Balancer](httpsavinetworks.com).
* [BFE Ingress Controller](httpsgithub.combfenetworksingress-bfe) is a [BFE](httpswww.bfe-networks.net)-based ingress controller.
* [Cilium Ingress Controller](httpsdocs.cilium.ioenstablenetworkservicemeshingress) is an ingress controller powered by [Cilium](httpscilium.io).
* The [Citrix ingress controller](httpsgithub.comcitrixcitrix-k8s-ingress-controller#readme) works with
  Citrix Application Delivery Controller.
* [Contour](httpsprojectcontour.io) is an [Envoy](httpswww.envoyproxy.io) based ingress controller.
* [Emissary-Ingress](httpswww.getambassador.ioproductsapi-gateway) API Gateway is an [Envoy](httpswww.envoyproxy.io)-based ingress
  controller.
* [EnRoute](httpsgetenroute.io) is an [Envoy](httpswww.envoyproxy.io) based API gateway that can run as an ingress controller.
* [Easegress IngressController](httpsmegaease.comdocseasegress04.cloud-native4.1.kubernetes-ingress-controller) is an [Easegress](httpsmegaease.comeasegress) based API gateway that can run as an ingress controller.
* F5 BIG-IP [Container Ingress Services for Kubernetes](httpsclouddocs.f5.comcontainerslatestuserguidekubernetes)
  lets you use an Ingress to configure F5 BIG-IP virtual servers.
* [FortiADC Ingress Controller](httpsdocs.fortinet.comdocumentfortiadc7.0.0fortiadc-ingress-controller742835fortiadc-ingress-controller-overview) support the Kubernetes Ingress resources and allows you to manage FortiADC objects from Kubernetes
* [Gloo](httpsgloo.solo.io) is an open-source ingress controller based on [Envoy](httpswww.envoyproxy.io),
  which offers API gateway functionality.
* [HAProxy Ingress](httpshaproxy-ingress.github.io) is an ingress controller for
  [HAProxy](httpswww.haproxy.org#desc).
* [Higress](httpsgithub.comalibabahigress) is an [Envoy](httpswww.envoyproxy.io) based API gateway that can run as an ingress controller.
* The [HAProxy Ingress Controller for Kubernetes](httpsgithub.comhaproxytechkubernetes-ingress#readme)
  is also an ingress controller for [HAProxy](httpswww.haproxy.org#desc).
* [Istio Ingress](httpsistio.iolatestdocstaskstraffic-managementingresskubernetes-ingress)
  is an [Istio](httpsistio.io) based ingress controller.
* The [Kong Ingress Controller for Kubernetes](httpsgithub.comKongkubernetes-ingress-controller#readme)
  is an ingress controller driving [Kong Gateway](httpskonghq.comkong).
* [Kusk Gateway](httpskusk.kubeshop.io) is an OpenAPI-driven ingress controller based on [Envoy](httpswww.envoyproxy.io).
* The [NGINX Ingress Controller for Kubernetes](httpswww.nginx.comproductsnginx-ingress-controller)
  works with the [NGINX](httpswww.nginx.comresourcesglossarynginx) webserver (as a proxy).
* The [ngrok Kubernetes Ingress Controller](httpsgithub.comngrokkubernetes-ingress-controller) is an open source controller for adding secure public access to your K8s services using the [ngrok platform](httpsngrok.com).
* The [OCI Native Ingress Controller](httpsgithub.comoracleoci-native-ingress-controller#readme) is an Ingress controller for Oracle Cloud Infrastructure which allows you to manage the [OCI Load Balancer](httpsdocs.oracle.comen-usiaasContentBalancehome.htm).
* [OpenNJet Ingress Controller](httpsgitee.comnjet-rdopen-njet-kic) is a [OpenNJet](httpsnjet.org.cn)-based ingress controller.
* The [Pomerium Ingress Controller](httpswww.pomerium.comdocsk8singress.html) is based on [Pomerium](httpspomerium.com), which offers context-aware access policy.
* [Skipper](httpsopensource.zalando.comskipperkubernetesingress-controller) HTTP router and reverse proxy for service composition, including use cases like Kubernetes Ingress, designed as a library to build your custom proxy.
* The [Traefik Kubernetes Ingress provider](httpsdoc.traefik.iotraefikproviderskubernetes-ingress) is an
  ingress controller for the [Traefik](httpstraefik.iotraefik) proxy.
* [Tyk Operator](httpsgithub.comTykTechnologiestyk-operator) extends Ingress with Custom Resources to bring API Management capabilities to Ingress. Tyk Operator works with the Open Source Tyk Gateway  Tyk Cloud control plane.
* [Voyager](httpsvoyagermesh.com) is an ingress controller for
  [HAProxy](httpswww.haproxy.org#desc).
* [Wallarm Ingress Controller](httpswww.wallarm.comsolutionswaf-for-kubernetes) is an Ingress Controller that provides WAAP (WAF) and API Security capabilities.

# # Using multiple Ingress controllers

You may deploy any number of ingress controllers using [ingress class](docsconceptsservices-networkingingress#ingress-class)
within a cluster. Note the `.metadata.name` of your ingress class resource. When you create an ingress you would need that name to specify the `ingressClassName` field on your Ingress object (refer to [IngressSpec v1 reference](docsreferencekubernetes-apiservice-resourcesingress-v1#IngressSpec)). `ingressClassName` is a replacement of the older [annotation method](docsconceptsservices-networkingingress#deprecated-annotation).

If you do not specify an IngressClass for an Ingress, and your cluster has exactly one IngressClass marked as default, then Kubernetes [applies](docsconceptsservices-networkingingress#default-ingress-class) the clusters default IngressClass to the Ingress.
You mark an IngressClass as default by setting the [`ingressclass.kubernetes.iois-default-class` annotation](docsreferencelabels-annotations-taints#ingressclass-kubernetes-io-is-default-class) on that IngressClass, with the string value `true`.

Ideally, all ingress controllers should fulfill this specification, but the various ingress
controllers operate slightly differently.

Make sure you review your ingress controllers documentation to understand the caveats of choosing it.

# #  heading whatsnext

* Learn more about [Ingress](docsconceptsservices-networkingingress).
* [Set up Ingress on Minikube with the NGINX Controller](docstasksaccess-application-clusteringress-minikube).
