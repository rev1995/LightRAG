---
title Using Source IP
content_type tutorial
min-kubernetes-server-version v1.5
weight 40
---

Applications running in a Kubernetes cluster find and communicate with each
other, and the outside world, through the Service abstraction. This document
explains what happens to the source IP of packets sent to different types
of Services, and how you can toggle this behavior according to your needs.

# #  heading prerequisites

# # # Terminology

This document makes use of the following terms

If localizing this section, link to the equivalent Wikipedia pages for
the target localization.

[NAT](httpsen.wikipedia.orgwikiNetwork_address_translation)
 Network address translation

[Source NAT](httpsen.wikipedia.orgwikiNetwork_address_translation#SNAT)
 Replacing the source IP on a packet in this page, that usually means replacing with the IP address of a node.

[Destination NAT](httpsen.wikipedia.orgwikiNetwork_address_translation#DNAT)
 Replacing the destination IP on a packet in this page, that usually means replacing with the IP address of a

[VIP](docsconceptsservices-networkingservice#virtual-ips-and-service-proxies)
 A virtual IP address, such as the one assigned to every  in Kubernetes

[kube-proxy](docsconceptsservices-networkingservice#virtual-ips-and-service-proxies)
 A network daemon that orchestrates Service VIP management on every node

# # # Prerequisites

The examples use a small nginx webserver that echoes back the source
IP of requests it receives through an HTTP header. You can create it as follows

The image in the following command only runs on AMD64 architectures.

```shell
kubectl create deployment source-ip-app --imageregistry.k8s.ioechoserver1.10
```
The output is
```
deployment.appssource-ip-app created
```

# #  heading objectives

* Expose a simple application through various types of Services
* Understand how each Service type handles source IP NAT
* Understand the tradeoffs involved in preserving source IP

# # Source IP for Services with `TypeClusterIP`

Packets sent to ClusterIP from within the cluster are never source NATd if
youre running kube-proxy in
[iptables mode](docsreferencenetworkingvirtual-ips#proxy-mode-iptables),
(the default). You can query the kube-proxy mode by fetching
`httplocalhost10249proxyMode` on the node where kube-proxy is running.

```console
kubectl get nodes
```
The output is similar to this
```
NAME                           STATUS     ROLES    AGE     VERSION
kubernetes-node-6jst   Ready         2h      v1.13.0
kubernetes-node-cx31   Ready         2h      v1.13.0
kubernetes-node-jj1t   Ready         2h      v1.13.0
```

Get the proxy mode on one of the nodes (kube-proxy listens on port 10249)
```shell
# Run this in a shell on the node you want to query.
curl httplocalhost10249proxyMode
```
The output is
```
iptables
```

You can test source IP preservation by creating a Service over the source IP app

```shell
kubectl expose deployment source-ip-app --nameclusterip --port80 --target-port8080
```
The output is
```
serviceclusterip exposed
```
```shell
kubectl get svc clusterip
```
The output is similar to
```
NAME         TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
clusterip    ClusterIP   10.0.170.92           80TCP    51s
```

And hitting the `ClusterIP` from a pod in the same cluster

```shell
kubectl run busybox -it --imagebusybox1.28 --restartNever --rm
```
The output is similar to this
```
Waiting for pod defaultbusybox to be running, status is Pending, pod ready false
If you dont see a command prompt, try pressing enter.

```
You can then run a command inside that Pod

```shell
# Run this inside the terminal from kubectl run
ip addr
```
```
1 lo  mtu 65536 qdisc noqueue
    linkloopback 000000000000 brd 000000000000
    inet 127.0.0.18 scope host lo
       valid_lft forever preferred_lft forever
    inet6 1128 scope host
       valid_lft forever preferred_lft forever
3 eth0  mtu 1460 qdisc noqueue
    linkether 0a580af40308 brd ffffffffffff
    inet 10.244.3.824 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80188a84fffeb026a564 scope link
       valid_lft forever preferred_lft forever
```

then use `wget` to query the local webserver
```shell
# Replace 10.0.170.92 with the IPv4 address of the Service named clusterip
wget -qO - 10.0.170.92
```
```
CLIENT VALUES
client_address10.244.3.8
commandGET
...
```
The `client_address` is always the client pods IP address, whether the client pod and server pod are in the same node or in different nodes.

# # Source IP for Services with `TypeNodePort`

Packets sent to Services with
[`TypeNodePort`](docsconceptsservices-networkingservice#type-nodeport)
are source NATd by default. You can test this by creating a `NodePort` Service

```shell
kubectl expose deployment source-ip-app --namenodeport --port80 --target-port8080 --typeNodePort
```
The output is
```
servicenodeport exposed
```

```shell
NODEPORT(kubectl get -o jsonpath.spec.ports[0].nodePort services nodeport)
NODES(kubectl get nodes -o jsonpath .items[*].status.addresses[(.typeInternalIP)].address )
```

If youre running on a cloud provider, you may need to open up a firewall-rule
for the `nodesnodeport` reported above.
Now you can try reaching the Service from outside the cluster through the node
port allocated above.

```shell
for node in NODES do curl -s nodeNODEPORT  grep -i client_address done
```
The output is similar to
```
client_address10.180.1.1
client_address10.240.0.5
client_address10.240.0.3
```

Note that these are not the correct client IPs, theyre cluster internal IPs. This is what happens

* Client sends packet to `node2nodePort`
* `node2` replaces the source IP address (SNAT) in the packet with its own IP address
* `node2` replaces the destination IP on the packet with the pod IP
* packet is routed to node 1, and then to the endpoint
* the pods reply is routed back to node2
* the pods reply is sent back to the client

Visually

To avoid this, Kubernetes has a feature to
[preserve the client source IP](docstasksaccess-application-clustercreate-external-load-balancer#preserving-the-client-source-ip).
If you set `service.spec.externalTrafficPolicy` to the value `Local`,
kube-proxy only proxies proxy requests to local endpoints, and does not
forward traffic to other nodes. This approach preserves the original
source IP address. If there are no local endpoints, packets sent to the
node are dropped, so you can rely on the correct source-ip in any packet
processing rules you might apply a packet that make it through to the
endpoint.

Set the `service.spec.externalTrafficPolicy` field as follows

```shell
kubectl patch svc nodeport -p specexternalTrafficPolicyLocal
```
The output is
```
servicenodeport patched
```

Now, re-run the test

```shell
for node in NODES do curl --connect-timeout 1 -s nodeNODEPORT  grep -i client_address done
```
The output is similar to
```
client_address198.51.100.79
```

Note that you only got one reply, with the *right* client IP, from the one node on which the endpoint pod
is running.

This is what happens

* client sends packet to `node2nodePort`, which doesnt have any endpoints
* packet is dropped
* client sends packet to `node1nodePort`, which *does* have endpoints
* node1 routes packet to endpoint with the correct source IP

Visually

# # Source IP for Services with `TypeLoadBalancer`

Packets sent to Services with
[`TypeLoadBalancer`](docsconceptsservices-networkingservice#loadbalancer)
are source NATd by default, because all schedulable Kubernetes nodes in the
`Ready` state are eligible for load-balanced traffic. So if packets arrive
at a node without an endpoint, the system proxies it to a node *with* an
endpoint, replacing the source IP on the packet with the IP of the node (as
described in the previous section).

You can test this by exposing the source-ip-app through a load balancer

```shell
kubectl expose deployment source-ip-app --nameloadbalancer --port80 --target-port8080 --typeLoadBalancer
```
The output is
```
serviceloadbalancer exposed
```

Print out the IP addresses of the Service
```console
kubectl get svc loadbalancer
```
The output is similar to this
```
NAME           TYPE           CLUSTER-IP    EXTERNAL-IP       PORT(S)   AGE
loadbalancer   LoadBalancer   10.0.65.118   203.0.113.140     80TCP    5m
```

Next, send a request to this Services external-ip

```shell
curl 203.0.113.140
```
The output is similar to this
```
CLIENT VALUES
client_address10.240.0.5
...
```

However, if youre running on Google Kubernetes EngineGCE, setting the same `service.spec.externalTrafficPolicy`
field to `Local` forces nodes *without* Service endpoints to remove
themselves from the list of nodes eligible for loadbalanced traffic by
deliberately failing health checks.

Visually

![Source IP with externalTrafficPolicy](imagesdocssourceip-externaltrafficpolicy.svg)

You can test this by setting the annotation

```shell
kubectl patch svc loadbalancer -p specexternalTrafficPolicyLocal
```

You should immediately see the `service.spec.healthCheckNodePort` field allocated
by Kubernetes

```shell
kubectl get svc loadbalancer -o yaml  grep -i healthCheckNodePort
```
The output is similar to this
```yaml
  healthCheckNodePort 32122
```

The `service.spec.healthCheckNodePort` field points to a port on every node
serving the health check at `healthz`. You can test this

```shell
kubectl get pod -o wide -l appsource-ip-app
```
The output is similar to this
```
NAME                            READY     STATUS    RESTARTS   AGE       IP             NODE
source-ip-app-826191075-qehz4   11       Running   0          20h       10.180.1.136   kubernetes-node-6jst
```

Use `curl` to fetch the `healthz` endpoint on various nodes
```shell
# Run this locally on a node you choose
curl localhost32122healthz
```
```
1 Service Endpoints found
```

On a different node you might get a different result
```shell
# Run this locally on a node you choose
curl localhost32122healthz
```
```
No Service Endpoints Found
```

A controller running on the
 is
responsible for allocating the cloud load balancer. The same controller also
allocates HTTP health checks pointing to this portpath on each node. Wait
about 10 seconds for the 2 nodes without endpoints to fail health checks,
then use `curl` to query the IPv4 address of the load balancer

```shell
curl 203.0.113.140
```
The output is similar to this
```
CLIENT VALUES
client_address198.51.100.79
...
```

# # Cross-platform support

Only some cloud providers offer support for source IP preservation through
Services with `TypeLoadBalancer`.
The cloud provider youre running on might fulfill the request for a loadbalancer
in a few different ways

1. With a proxy that terminates the client connection and opens a new connection
to your nodesendpoints. In such cases the source IP will always be that of the
cloud LB, not that of the client.

2. With a packet forwarder, such that requests from the client sent to the
loadbalancer VIP end up at the node with the source IP of the client, not
an intermediate proxy.

Load balancers in the first category must use an agreed upon
protocol between the loadbalancer and backend to communicate the true client IP
such as the HTTP [Forwarded](httpstools.ietf.orghtmlrfc7239#section-5.2)
or [X-FORWARDED-FOR](httpsen.wikipedia.orgwikiX-Forwarded-For)
headers, or the
[proxy protocol](httpswww.haproxy.orgdownload1.8docproxy-protocol.txt).
Load balancers in the second category can leverage the feature described above
by creating an HTTP health check pointing at the port stored in
the `service.spec.healthCheckNodePort` field on the Service.

# #  heading cleanup

Delete the Services

```shell
kubectl delete svc -l appsource-ip-app
```

Delete the Deployment, ReplicaSet and Pod

```shell
kubectl delete deployment source-ip-app
```

# #  heading whatsnext

* Learn more about [connecting applications via services](docstutorialsservicesconnect-applications-service)
* Read how to [Create an External Load Balancer](docstasksaccess-application-clustercreate-external-load-balancer)
