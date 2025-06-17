---
reviewers
- aravindhp
- jayunit100
- jsturtevant
- marosset
title Networking on Windows
content_type concept
weight 110
---

Kubernetes supports running nodes on either Linux or Windows. You can mix both kinds of node
within a single cluster.
This page provides an overview to networking specific to the Windows operating system.

# # Container networking on Windows #networking

Networking for Windows containers is exposed through
[CNI plugins](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins).
Windows containers function similarly to virtual machines in regards to
networking. Each container has a virtual network adapter (vNIC) which is connected
to a Hyper-V virtual switch (vSwitch). The Host Networking Service (HNS) and the
Host Compute Service (HCS) work together to create containers and attach container
vNICs to networks. HCS is responsible for the management of containers whereas HNS
is responsible for the management of networking resources such as

* Virtual networks (including creation of vSwitches)
* Endpoints  vNICs
* Namespaces
* Policies including packet encapsulations, load-balancing rules, ACLs, and NAT rules.

The Windows HNS and vSwitch implement namespacing and can
create virtual NICs as needed for a pod or container. However, many configurations such
as DNS, routes, and metrics are stored in the Windows registry database rather than as
files inside `etc`, which is how Linux stores those configurations. The Windows registry for the container
is separate from that of the host, so concepts like mapping `etcresolv.conf` from
the host into a container dont have the same effect they would on Linux. These must
be configured using Windows APIs run in the context of that container. Therefore
CNI implementations need to call the HNS instead of relying on file mappings to pass
network details into the pod or container.

# # Network modes

Windows supports five different networking driversmodes L2bridge, L2tunnel,
Overlay (Beta), Transparent, and NAT. In a heterogeneous cluster with Windows and Linux
worker nodes, you need to select a networking solution that is compatible on both
Windows and Linux. The following table lists the out-of-tree plugins are supported on Windows,
with recommendations on when to use each CNI

 Network Driver  Description  Container Packet Modifications  Network Plugins  Network Plugin Characteristics
 --------------  -----------  ------------------------------  ---------------  ------------------------------
 L2bridge        Containers are attached to an external vSwitch. Containers are attached to the underlay network, although the physical network doesnt need to learn the container MACs because they are rewritten on ingressegress.  MAC is rewritten to host MAC, IP may be rewritten to host IP using HNS OutboundNAT policy.  [win-bridge](httpswww.cni.devpluginscurrentmainwin-bridge), [Azure-CNI](httpsgithub.comAzureazure-container-networkingblobmasterdocscni.md), [Flannel host-gateway](httpsgithub.comflannel-ioflannelblobmasterDocumentationbackends.md#host-gw) uses win-bridge  win-bridge uses L2bridge network mode, connects containers to the underlay of hosts, offering best performance. Requires user-defined routes (UDR) for inter-node connectivity.
 L2Tunnel  This is a special case of l2bridge, but only used on Azure. All packets are sent to the virtualization host where SDN policy is applied.  MAC rewritten, IP visible on the underlay network  [Azure-CNI](httpsgithub.comAzureazure-container-networkingblobmasterdocscni.md)  Azure-CNI allows integration of containers with Azure vNET, and allows them to leverage the set of capabilities that [Azure Virtual Network provides](httpsazure.microsoft.comen-usservicesvirtual-network). For example, securely connect to Azure services or use Azure NSGs. See [azure-cni for some examples](httpsdocs.microsoft.comazureaksconcepts-network#azure-cni-advanced-networking)
 Overlay  Containers are given a vNIC connected to an external vSwitch. Each overlay network gets its own IP subnet, defined by a custom IP prefix.The overlay network driver uses VXLAN encapsulation.  Encapsulated with an outer header.  [win-overlay](httpswww.cni.devpluginscurrentmainwin-overlay), [Flannel VXLAN](httpsgithub.comflannel-ioflannelblobmasterDocumentationbackends.md#vxlan) (uses win-overlay)  win-overlay should be used when virtual container networks are desired to be isolated from underlay of hosts (e.g. for security reasons). Allows for IPs to be re-used for different overlay networks (which have different VNID tags)  if you are restricted on IPs in your datacenter.  This option requires [KB4489899](httpssupport.microsoft.comhelp4489899) on Windows Server 2019.
 Transparent (special use case for [ovn-kubernetes](httpsgithub.comopenvswitchovn-kubernetes))  Requires an external vSwitch. Containers are attached to an external vSwitch which enables intra-pod communication via logical networks (logical switches and routers).  Packet is encapsulated either via [GENEVE](httpsdatatracker.ietf.orgdocdraft-gross-geneve) or [STT](httpsdatatracker.ietf.orgdocdraft-davie-stt) tunneling to reach pods which are not on the same host.   Packets are forwarded or dropped via the tunnel metadata information supplied by the ovn network controller.  NAT is done for north-south communication.  [ovn-kubernetes](httpsgithub.comopenvswitchovn-kubernetes)  [Deploy via ansible](httpsgithub.comopenvswitchovn-kubernetestreemastercontrib). Distributed ACLs can be applied via Kubernetes policies. IPAM support. Load-balancing can be achieved without kube-proxy. NATing is done without using iptablesnetsh.
 NAT (*not used in Kubernetes*)  Containers are given a vNIC connected to an internal vSwitch. DNSDHCP is provided using an internal component called [WinNAT](httpstechcommunity.microsoft.comt5virtualizationwindows-nat-winnat-capabilities-and-limitationsba-p382303)  MAC and IP is rewritten to host MACIP.  [nat](httpsgithub.comMicrosoftwindows-container-networkingtreemasterpluginsnat)  Included here for completeness

As outlined above, the [Flannel](httpsgithub.comcoreosflannel)
[CNI plugin](httpsgithub.comflannel-iocni-plugin)
is also [supported](httpsgithub.comflannel-iocni-plugin#windows-support-experimental) on Windows via the
[VXLAN network backend](httpsgithub.comcoreosflannelblobmasterDocumentationbackends.md#vxlan) (**Beta support**  delegates to win-overlay)
and [host-gateway network backend](httpsgithub.comcoreosflannelblobmasterDocumentationbackends.md#host-gw) (stable support delegates to win-bridge).

This plugin supports delegating to one of the reference CNI plugins (win-overlay,
win-bridge), to work in conjunction with Flannel daemon on Windows (Flanneld) for
automatic node subnet lease assignment and HNS network creation. This plugin reads
in its own configuration file (cni.conf), and aggregates it with the environment
variables from the FlannelD generated subnet.env file. It then delegates to one of
the reference CNI plugins for network plumbing, and sends the correct configuration
containing the node-assigned subnet to the IPAM plugin (for example `host-local`).

For Node, Pod, and Service objects, the following network flows are supported for
TCPUDP traffic

* Pod  Pod (IP)
* Pod  Pod (Name)
* Pod  Service (Cluster IP)
* Pod  Service (PQDN, but only if there are no .)
* Pod  Service (FQDN)
* Pod  external (IP)
* Pod  external (DNS)
* Node  Pod
* Pod  Node

# # IP address management (IPAM) #ipam

The following IPAM options are supported on Windows

* [host-local](httpsgithub.comcontainernetworkingpluginstreemasterpluginsipamhost-local)
* [azure-vnet-ipam](httpsgithub.comAzureazure-container-networkingblobmasterdocsipam.md) (for azure-cni only)
* [Windows Server IPAM](httpsdocs.microsoft.comwindows-servernetworkingtechnologiesipamipam-top) (fallback option if no IPAM is set)

# # Direct Server Return (DSR) #dsr

Load balancing mode where the IP address fixups and the LBNAT occurs at the container vSwitch port directly
service traffic arrives with the source IP set as the originating pod IP.
This provides performance optimizations by allowing the return traffic routed through load balancers
to bypass the load balancer and respond directly to the client
reducing load on the load balancer and also reducing overall latency.
For more information, read
[Direct Server Return (DSR) in a nutshell](httpstechcommunity.microsoft.comblognetworkingblogdirect-server-return-dsr-in-a-nutshell693710).

# # Load balancing and Services

A Kubernetes  is an abstraction
that defines a logical set of Pods and a means to access them over a network.
In a cluster that includes Windows nodes, you can use the following types of Service

* `NodePort`
* `ClusterIP`
* `LoadBalancer`
* `ExternalName`

Windows container networking differs in some important ways from Linux networking.
The [Microsoft documentation for Windows Container Networking](httpsdocs.microsoft.comen-usvirtualizationwindowscontainerscontainer-networkingarchitecture)
provides additional details and background.

On Windows, you can use the following settings to configure Services and load
balancing behavior

 Feature  Description  Minimum Supported Windows OS build  How to enable
 -------  -----------  --------------------------  -------------
 Session affinity  Ensures that connections from a particular client are passed to the same Pod each time.  Windows Server 2022  Set `service.spec.sessionAffinity` to ClientIP
 Direct Server Return (DSR)  See [DSR](#dsr) notes above.  Windows Server 2019  Set the following command line argument (assuming version ) ` --enable-dsrtrue`
 Preserve-Destination  Skips DNAT of service traffic, thereby preserving the virtual IP of the target service in packets reaching the backend Pod. Also disables node-node forwarding.  Windows Server, version 1903  Set `preserve-destination true` in service annotations and enable DSR in kube-proxy.
 IPv4IPv6 dual-stack networking  Native IPv4-to-IPv4 in parallel with IPv6-to-IPv6 communications to, from, and within a cluster  Windows Server 2019  See [IPv4IPv6 dual-stack](docsconceptsservices-networkingdual-stack#windows-support)
 Client IP preservation  Ensures that source IP of incoming ingress traffic gets preserved. Also disables node-node forwarding.   Windows Server 2019   Set `service.spec.externalTrafficPolicy` to Local and enable DSR in kube-proxy

# # Limitations

The following networking functionality is _not_ supported on Windows nodes

* Host networking mode
* Local NodePort access from the node itself (works for other nodes or external clients)
* More than 64 backend pods (or unique destination addresses) for a single Service
* IPv6 communication between Windows pods connected to overlay networks
* Local Traffic Policy in non-DSR mode
* Outbound communication using the ICMP protocol via the `win-overlay`, `win-bridge`, or using the Azure-CNI plugin.
  Specifically, the Windows data plane ([VFP](httpswww.microsoft.comresearchprojectazure-virtual-filtering-platform))
  doesnt support ICMP packet transpositions, and this means
  * ICMP packets directed to destinations within the same network (such as pod to pod communication via ping)
    work as expected
  * TCPUDP packets work as expected
  * ICMP packets directed to pass through a remote network (e.g. pod to external internet communication via ping)
    cannot be transposed and thus will not be routed back to their source
  * Since TCPUDP packets can still be transposed, you can substitute `ping ` with
    `curl ` when debugging connectivity with the outside world.

Other limitations

* Windows reference network plugins win-bridge and win-overlay do not implement
  [CNI spec](httpsgithub.comcontainernetworkingcniblobmasterSPEC.md) v0.4.0,
  due to a missing `CHECK` implementation.
* The Flannel VXLAN CNI plugin has the following limitations on Windows
  * Node-pod connectivity is only possible for local pods with Flannel v0.12.0 (or higher).
  * Flannel is restricted to using VNI 4096 and UDP port 4789. See the official
    [Flannel VXLAN](httpsgithub.comcoreosflannelblobmasterDocumentationbackends.md#vxlan)
    backend docs for more details on these parameters.
