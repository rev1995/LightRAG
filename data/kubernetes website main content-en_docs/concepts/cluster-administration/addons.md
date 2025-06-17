---
title Installing Addons
content_type concept
weight 150
---

 thirdparty-content

Add-ons extend the functionality of Kubernetes.

This page lists some of the available add-ons and links to their respective
installation instructions. The list does not try to be exhaustive.

# # Networking and Network Policy

* [ACI](httpswww.github.comnoironetworksaci-containers) provides integrated
  container networking and network security with Cisco ACI.
* [Antrea](httpsantrea.io) operates at Layer 34 to provide networking and
  security services for Kubernetes, leveraging Open vSwitch as the networking
  data plane. Antrea is a [CNCF project at the Sandbox level](httpswww.cncf.ioprojectsantrea).
* [Calico](httpswww.tigera.ioproject-calico) is a networking and network
  policy provider. Calico supports a flexible set of networking options so you
  can choose the most efficient option for your situation, including non-overlay
  and overlay networks, with or without BGP. Calico uses the same engine to
  enforce network policy for hosts, pods, and (if using Istio  Envoy)
  applications at the service mesh layer.
* [Canal](httpsprojectcalico.docs.tigera.iogetting-startedkubernetesflannelflannel)
  unites Flannel and Calico, providing networking and network policy.
* [Cilium](httpsgithub.comciliumcilium) is a networking, observability,
  and security solution with an eBPF-based data plane. Cilium provides a
  simple flat Layer 3 network with the ability to span multiple clusters
  in either a native routing or overlayencapsulation mode, and can enforce
  network policies on L3-L7 using an identity-based security model that is
  decoupled from network addressing. Cilium can act as a replacement for
  kube-proxy it also offers additional, opt-in observability and security features.
  Cilium is a [CNCF project at the Graduated level](httpswww.cncf.ioprojectscilium).
* [CNI-Genie](httpsgithub.comcni-genieCNI-Genie) enables Kubernetes to seamlessly
  connect to a choice of CNI plugins, such as Calico, Canal, Flannel, or Weave.
  CNI-Genie is a [CNCF project at the Sandbox level](httpswww.cncf.ioprojectscni-genie).
* [Contiv](httpscontivpp.io) provides configurable networking (native L3 using BGP,
  overlay using vxlan, classic L2, and Cisco-SDNACI) for various use cases and a rich
  policy framework. Contiv project is fully [open sourced](httpsgithub.comcontiv).
  The [installer](httpsgithub.comcontivinstall) provides both kubeadm and
  non-kubeadm based installation options.
* [Contrail](httpswww.juniper.netusenproducts-servicessdncontrailcontrail-networking),
  based on [Tungsten Fabric](httpstungsten.io), is an open source, multi-cloud
  network virtualization and policy management platform. Contrail and Tungsten
  Fabric are integrated with orchestration systems such as Kubernetes, OpenShift,
  OpenStack and Mesos, and provide isolation modes for virtual machines, containerspods
  and bare metal workloads.
* [Flannel](httpsgithub.comflannel-ioflannel#deploying-flannel-manually) is
  an overlay network provider that can be used with Kubernetes.
* [Gateway API](docsconceptsservices-networkinggateway) is an open source project managed by
  the [SIG Network](httpsgithub.comkubernetescommunitytreemastersig-network) community and
  provides an expressive, extensible, and role-oriented API for modeling service networking.
* [Knitter](httpsgithub.comZTEKnitter) is a plugin to support multiple network
  interfaces in a Kubernetes pod.
* [Multus](httpsgithub.comk8snetworkplumbingwgmultus-cni) is a Multi plugin for
  multiple network support in Kubernetes to support all CNI plugins
  (e.g. Calico, Cilium, Contiv, Flannel), in addition to SRIOV, DPDK, OVS-DPDK and
  VPP based workloads in Kubernetes.
* [OVN-Kubernetes](httpsgithub.comovn-orgovn-kubernetes) is a networking
  provider for Kubernetes based on [OVN (Open Virtual Network)](httpsgithub.comovn-orgovn),
  a virtual networking implementation that came out of the Open vSwitch (OVS) project.
  OVN-Kubernetes provides an overlay based networking implementation for Kubernetes,
  including an OVS based implementation of load balancing and network policy.
* [Nodus](httpsgithub.comakraino-edge-stackicn-nodus) is an OVN based CNI
  controller plugin to provide cloud native based Service function chaining(SFC).
* [NSX-T](httpsdocs.vmware.comenVMware-NSX-T-Data-Centerindex.html) Container Plug-in (NCP)
  provides integration between VMware NSX-T and container orchestrators such as
  Kubernetes, as well as integration between NSX-T and container-based CaaSPaaS
  platforms such as Pivotal Container Service (PKS) and OpenShift.
* [Nuage](httpsgithub.comnuagenetworksnuage-kubernetesblobv5.1.1-1docskubernetes-1-installation.rst)
  is an SDN platform that provides policy-based networking between Kubernetes
  Pods and non-Kubernetes environments with visibility and security monitoring.
* [Romana](httpsgithub.comromana) is a Layer 3 networking solution for pod
  networks that also supports the [NetworkPolicy](docsconceptsservices-networkingnetwork-policies) API.
* [Spiderpool](httpsgithub.comspidernet-iospiderpool) is an underlay and RDMA
  networking solution for Kubernetes. Spiderpool is supported on bare metal, virtual machines,
  and public cloud environments.
* [Weave Net](httpsgithub.comrajchweave#using-weave-on-kubernetes)
  provides networking and network policy, will carry on working on both sides
  of a network partition, and does not require an external database.

# # Service Discovery

* [CoreDNS](httpscoredns.io) is a flexible, extensible DNS server which can
  be [installed](httpsgithub.comcorednshelm)
  as the in-cluster DNS for pods.

# # Visualization amp Control

* [Dashboard](httpsgithub.comkubernetesdashboard#kubernetes-dashboard)
  is a dashboard web interface for Kubernetes.

# # Infrastructure

* [KubeVirt](httpskubevirt.iouser-guide#installationinstallation) is an add-on
  to run virtual machines on Kubernetes. Usually run on bare-metal clusters.
* The
  [node problem detector](httpsgithub.comkubernetesnode-problem-detector)
  runs on Linux nodes and reports system issues as either
  [Events](docsreferencekubernetes-apicluster-resourcesevent-v1) or
  [Node conditions](docsconceptsarchitecturenodes#condition).

# # Instrumentation

* [kube-state-metrics](docsconceptscluster-administrationkube-state-metrics)

# # Legacy Add-ons

There are several other add-ons documented in the deprecated
[clusteraddons](httpsgit.k8s.iokubernetesclusteraddons) directory.

Well-maintained ones should be linked to here. PRs welcome!
