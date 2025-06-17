---
reviewers
- jbelamaric
- bowei
- thockin
title DNS for Services and Pods
content_type concept
weight 80
description -
  Your workload can discover Services within your cluster using DNS
  this page explains how that works.
---

Kubernetes creates DNS records for Services and Pods. You can contact
Services with consistent DNS names instead of IP addresses.

Kubernetes publishes information about Pods and Services which is used
to program DNS. kubelet configures Pods DNS so that running containers
can look up Services by name rather than IP.

Services defined in the cluster are assigned DNS names. By default, a
client Pods DNS search list includes the Pods own namespace and the
clusters default domain.

# # # Namespaces of Services

A DNS query may return different results based on the namespace of the Pod making
it. DNS queries that dont specify a namespace are limited to the Pods
namespace. Access Services in other namespaces by specifying it in the DNS query.

For example, consider a Pod in a `test` namespace. A `data` Service is in
the `prod` namespace.

A query for `data` returns no results, because it uses the Pods `test` namespace.

A query for `data.prod` returns the intended result, because it specifies the
namespace.

DNS queries may be expanded using the Pods `etcresolv.conf`. kubelet
configures this file for each Pod. For example, a query for just `data` may be
expanded to `data.test.svc.cluster.local`. The values of the `search` option
are used to expand queries. To learn more about DNS queries, see
[the `resolv.conf` manual page](httpswww.man7.orglinuxman-pagesman5resolv.conf.5.html).

```
nameserver 10.32.0.10
search .svc.cluster.local svc.cluster.local cluster.local
options ndots5
```

In summary, a Pod in the _test_ namespace can successfully resolve either
`data.prod` or `data.prod.svc.cluster.local`.

# # # DNS Records

What objects get DNS records

1. Services
1. Pods

The following sections detail the supported DNS record types and layout that is
supported. Any other layout or names or queries that happen to work are
considered implementation details and are subject to change without warning.
For more up-to-date specification, see
[Kubernetes DNS-Based Service Discovery](httpsgithub.comkubernetesdnsblobmasterdocsspecification.md).

# # Services

# # # AAAAA records

Normal (not headless) Services are assigned DNS A andor AAAA records,
depending on the IP family or families of the Service, with a name of the form
`my-svc.my-namespace.svc.cluster-domain.example`. This resolves to the cluster IP
of the Service.

[Headless Services](docsconceptsservices-networkingservice#headless-services)
(without a cluster IP) are also assigned DNS A andor AAAA records,
with a name of the form `my-svc.my-namespace.svc.cluster-domain.example`. Unlike normal
Services, this resolves to the set of IPs of all of the Pods selected by the Service.
Clients are expected to consume the set or else use standard round-robin
selection from the set.

# # # SRV records

SRV Records are created for named ports that are part of normal or headless
services.

- For each named port, the SRV record has the form
  `_port-name._port-protocol.my-svc.my-namespace.svc.cluster-domain.example`.
- For a regular Service, this resolves to the port number and the domain name
  `my-svc.my-namespace.svc.cluster-domain.example`.
- For a headless Service, this resolves to multiple answers, one for each Pod
  that is backing the Service, and contains the port number and the domain name of the Pod
  of the form `hostname.my-svc.my-namespace.svc.cluster-domain.example`.

# # Pods

# # # AAAAA records

Kube-DNS versions, prior to the implementation of the
[DNS specification](httpsgithub.comkubernetesdnsblobmasterdocsspecification.md),
had the following DNS resolution

```
pod-ipv4-address.my-namespace.pod.cluster-domain.example
```

For example, if a Pod in the `default` namespace has the IP address 172.17.0.3,
and the domain name for your cluster is `cluster.local`, then the Pod has a DNS name

```
172-17-0-3.default.pod.cluster.local
```

Some cluster DNS mechanisms, like [CoreDNS](httpscoredns.io), also provide `A` records for

```
...svc.
```

# # # Pods hostname and subdomain fields

Currently when a Pod is created, its hostname (as observed from within the Pod)
is the Pods `metadata.name` value.

The Pod spec has an optional `hostname` field, which can be used to specify a
different hostname. When specified, it takes precedence over the Pods name to be
the hostname of the Pod (again, as observed from within the Pod). For example,
given a Pod with `spec.hostname` set to `my-host`, the Pod will have its
hostname set to `my-host`.

The Pod spec also has an optional `subdomain` field which can be used to indicate
that the pod is part of sub-group of the namespace. For example, a Pod with `spec.hostname`
set to `foo`, and `spec.subdomain` set to `bar`, in namespace `my-namespace`, will
have its hostname set to `foo` and its fully qualified domain name (FQDN) set to
`foo.bar.my-namespace.svc.cluster.local` (once more, as observed from within
the Pod).

If there exists a headless Service in the same namespace as the Pod, with
the same name as the subdomain, the clusters DNS Server also returns A andor AAAA
records for the Pods fully qualified hostname.

Example

```yaml
apiVersion v1
kind Service
metadata
  name busybox-subdomain
spec
  selector
    name busybox
  clusterIP None
  ports
  - name foo # name is not required for single-port Services
    port 1234
---
apiVersion v1
kind Pod
metadata
  name busybox1
  labels
    name busybox
spec
  hostname busybox-1
  subdomain busybox-subdomain
  containers
  - image busybox1.28
    command
      - sleep
      - 3600
    name busybox
---
apiVersion v1
kind Pod
metadata
  name busybox2
  labels
    name busybox
spec
  hostname busybox-2
  subdomain busybox-subdomain
  containers
  - image busybox1.28
    command
      - sleep
      - 3600
    name busybox
```

Given the above Service `busybox-subdomain` and the Pods which set `spec.subdomain`
to `busybox-subdomain`, the first Pod will see its own FQDN as
`busybox-1.busybox-subdomain.my-namespace.svc.cluster-domain.example`. DNS serves
A andor AAAA records at that name, pointing to the Pods IP. Both Pods `busybox1` and
`busybox2` will have their own address records.

An  can specify
the DNS hostname for any endpoint addresses, along with its IP.

A and AAAA records are not created for Pod names since `hostname` is missing for the Pod.
A Pod with no `hostname` but with `subdomain` will only create the
A or AAAA record for the headless Service (`busybox-subdomain.my-namespace.svc.cluster-domain.example`),
pointing to the Pods IP addresses. Also, the Pod needs to be ready in order to have a
record unless `publishNotReadyAddressesTrue` is set on the Service.

# # # Pods setHostnameAsFQDN field #pod-sethostnameasfqdn-field

When a Pod is configured to have fully qualified domain name (FQDN), its
hostname is the short hostname. For example, if you have a Pod with the fully
qualified domain name `busybox-1.busybox-subdomain.my-namespace.svc.cluster-domain.example`,
then by default the `hostname` command inside that Pod returns `busybox-1` and the
`hostname --fqdn` command returns the FQDN.

When you set `setHostnameAsFQDN true` in the Pod spec, the kubelet writes the Pods FQDN
into the hostname for that Pods namespace. In this case, both `hostname` and `hostname --fqdn`
return the Pods FQDN.

In Linux, the hostname field of the kernel (the `nodename` field of `struct utsname`) is limited to 64 characters.

If a Pod enables this feature and its FQDN is longer than 64 character, it will fail to start.
The Pod will remain in `Pending` status (`ContainerCreating` as seen by `kubectl`) generating
error events, such as Failed to construct FQDN from Pod hostname and cluster domain,
FQDN `long-FQDN` is too long (64 characters is the max, 70 characters requested).
One way of improving user experience for this scenario is to create an
[admission webhook controller](docsreferenceaccess-authn-authzextensible-admission-controllers#what-are-admission-webhooks)
to control FQDN size when users create top level objects, for example, Deployment.

# # # Pods DNS Policy

DNS policies can be set on a per-Pod basis. Currently Kubernetes supports the
following Pod-specific DNS policies. These policies are specified in the
`dnsPolicy` field of a Pod Spec.

- `Default` The Pod inherits the name resolution configuration from the node
  that the Pods run on.
  See [related discussion](docstasksadminister-clusterdns-custom-nameservers)
  for more details.
- `ClusterFirst` Any DNS query that does not match the configured cluster
  domain suffix, such as `www.kubernetes.io`, is forwarded to an upstream
  nameserver by the DNS server. Cluster administrators may have extra
  stub-domain and upstream DNS servers configured.
  See [related discussion](docstasksadminister-clusterdns-custom-nameservers)
  for details on how DNS queries are handled in those cases.
- `ClusterFirstWithHostNet` For Pods running with hostNetwork, you should
  explicitly set its DNS policy to `ClusterFirstWithHostNet`. Otherwise, Pods
  running with hostNetwork and `ClusterFirst` will fallback to the behavior
  of the `Default` policy.

  This is not supported on Windows. See [below](#dns-windows) for details.

- `None` It allows a Pod to ignore DNS settings from the Kubernetes
  environment. All DNS settings are supposed to be provided using the
  `dnsConfig` field in the Pod Spec.
  See [Pods DNS config](#pod-dns-config) subsection below.

Default is not the default DNS policy. If `dnsPolicy` is not
explicitly specified, then ClusterFirst is used.

The example below shows a Pod with its DNS policy set to
`ClusterFirstWithHostNet` because it has `hostNetwork` set to `true`.

```yaml
apiVersion v1
kind Pod
metadata
  name busybox
  namespace default
spec
  containers
  - image busybox1.28
    command
      - sleep
      - 3600
    imagePullPolicy IfNotPresent
    name busybox
  restartPolicy Always
  hostNetwork true
  dnsPolicy ClusterFirstWithHostNet
```

# # # Pods DNS Config #pod-dns-config

Pods DNS Config allows users more control on the DNS settings for a Pod.

The `dnsConfig` field is optional and it can work with any `dnsPolicy` settings.
However, when a Pods `dnsPolicy` is set to `None`, the `dnsConfig` field has
to be specified.

Below are the properties a user can specify in the `dnsConfig` field

- `nameservers` a list of IP addresses that will be used as DNS servers for the
  Pod. There can be at most 3 IP addresses specified. When the Pods `dnsPolicy`
  is set to `None`, the list must contain at least one IP address, otherwise
  this property is optional.
  The servers listed will be combined to the base nameservers generated from the
  specified DNS policy with duplicate addresses removed.
- `searches` a list of DNS search domains for hostname lookup in the Pod.
  This property is optional. When specified, the provided list will be merged
  into the base search domain names generated from the chosen DNS policy.
  Duplicate domain names are removed.
  Kubernetes allows up to 32 search domains.
- `options` an optional list of objects where each object may have a `name`
  property (required) and a `value` property (optional). The contents in this
  property will be merged to the options generated from the specified DNS policy.
  Duplicate entries are removed.

The following is an example Pod with custom DNS settings

 code_sample fileservicenetworkingcustom-dns.yaml

When the Pod above is created, the container `test` gets the following contents
in its `etcresolv.conf` file

```
nameserver 192.0.2.1
search ns1.svc.cluster-domain.example my.dns.search.suffix
options ndots2 edns0
```

For IPv6 setup, search path and name server should be set up like this

```shell
kubectl exec -it dns-example -- cat etcresolv.conf
```

The output is similar to this

```
nameserver 2001db830a
search default.svc.cluster-domain.example svc.cluster-domain.example cluster-domain.example
options ndots5
```

# # DNS search domain list limits

Kubernetes itself does not limit the DNS Config until the length of the search
domain list exceeds 32 or the total length of all search domains exceeds 2048.
This limit applies to the nodes resolver configuration file, the Pods DNS
Config, and the merged DNS Config respectively.

Some container runtimes of earlier versions may have their own restrictions on
the number of DNS search domains. Depending on the container runtime
environment, the pods with a large number of DNS search domains may get stuck in
the pending state.

It is known that containerd v1.5.5 or earlier and CRI-O v1.21 or earlier have
this problem.

# # DNS resolution on Windows nodes #dns-windows

- `ClusterFirstWithHostNet` is not supported for Pods that run on Windows nodes.
  Windows treats all names with a `.` as a FQDN and skips FQDN resolution.
- On Windows, there are multiple DNS resolvers that can be used. As these come with
  slightly different behaviors, using the
  [`Resolve-DNSName`](httpsdocs.microsoft.compowershellmodulednsclientresolve-dnsname)
  powershell cmdlet for name query resolutions is recommended.
- On Linux, you have a DNS suffix list, which is used after resolution of a name as fully
  qualified has failed.
  On Windows, you can only have 1 DNS suffix, which is the DNS suffix associated with that
  Pods namespace (example `mydns.svc.cluster.local`). Windows can resolve FQDNs, Services,
  or network name which can be resolved with this single suffix. For example, a Pod spawned
  in the `default` namespace, will have the DNS suffix `default.svc.cluster.local`.
  Inside a Windows Pod, you can resolve both `kubernetes.default.svc.cluster.local`
  and `kubernetes`, but not the partially qualified names (`kubernetes.default` or
  `kubernetes.default.svc`).

# #  heading whatsnext

For guidance on administering DNS configurations, check
[Configure DNS Service](docstasksadminister-clusterdns-custom-nameservers).
