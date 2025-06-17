---
title Dual-stack support with kubeadm
content_type task
weight 100
min-kubernetes-server-version 1.21
---

Your Kubernetes cluster includes [dual-stack](docsconceptsservices-networkingdual-stack)
networking, which means that cluster networking lets you use either address family.
In a cluster, the control plane can assign both an IPv4 address and an IPv6 address to a single
 or a .

# #  heading prerequisites

You need to have installed the  tool,
following the steps from [Installing kubeadm](docssetupproduction-environmenttoolskubeadminstall-kubeadm).

For each server that you want to use as a ,
make sure it allows IPv6 forwarding.

# # # Enable IPv6 packet forwarding #prerequisite-ipv6-forwarding

To check if IPv6 packet forwarding is enabled

```bash
sysctl net.ipv6.conf.all.forwarding
```
If the output is `net.ipv6.conf.all.forwarding  1` it is already enabled.
Otherwise it is not enabled yet.

To manually enable IPv6 packet forwarding

```bash
# sysctl params required by setup, params persist across reboots
cat
If you are upgrading an existing cluster with the `kubeadm upgrade` command,
`kubeadm` does not support making modifications to the pod IP address range
(cluster CIDR) nor to the clusters Service address range (Service CIDR).

# # # Create a dual-stack cluster

To create a dual-stack cluster with `kubeadm init` you can pass command line arguments
similar to the following example

```shell
# These address ranges are examples
kubeadm init --pod-network-cidr10.244.0.016,2001db842056 --service-cidr10.96.0.016,2001db8421112
```

To make things clearer, here is an example kubeadm
[configuration file](docsreferenceconfig-apikubeadm-config.v1beta4)
`kubeadm-config.yaml` for the primary dual-stack control plane node.

```yaml
---
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
networking
  podSubnet 10.244.0.016,2001db842056
  serviceSubnet 10.96.0.016,2001db8421112
---
apiVersion kubeadm.k8s.iov1beta4
kind InitConfiguration
localAPIEndpoint
  advertiseAddress 10.100.0.1
  bindPort 6443
nodeRegistration
  kubeletExtraArgs
  - name node-ip
    value 10.100.0.2,fd001232
```

`advertiseAddress` in InitConfiguration specifies the IP address that the API Server
will advertise it is listening on. The value of `advertiseAddress` equals the
`--apiserver-advertise-address` flag of `kubeadm init`.

Run kubeadm to initiate the dual-stack control plane node

```shell
kubeadm init --configkubeadm-config.yaml
```

The kube-controller-manager flags `--node-cidr-mask-size-ipv4--node-cidr-mask-size-ipv6`
are set with default values. See [configure IPv4IPv6 dual stack](docsconceptsservices-networkingdual-stack#configure-ipv4-ipv6-dual-stack).

The `--apiserver-advertise-address` flag does not support dual-stack.

# # # Join a node to dual-stack cluster

Before joining a node, make sure that the node has IPv6 routable network interface and allows IPv6 forwarding.

Here is an example kubeadm [configuration file](docsreferenceconfig-apikubeadm-config.v1beta4)
`kubeadm-config.yaml` for joining a worker node to the cluster.

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind JoinConfiguration
discovery
  bootstrapToken
    apiServerEndpoint 10.100.0.16443
    token clvldh.vjjwg16ucnhp94qr
    caCertHashes
    - sha256a4863cde706cfc580a439f842cc65d5ef112b7b2be31628513a9881cf0d9fe0e
    # change auth info above to match the actual token and CA certificate hash for your cluster
nodeRegistration
  kubeletExtraArgs
  - name node-ip
    value 10.100.0.2,fd001233
```

Also, here is an example kubeadm [configuration file](docsreferenceconfig-apikubeadm-config.v1beta4)
`kubeadm-config.yaml` for joining another control plane node to the cluster.

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind JoinConfiguration
controlPlane
  localAPIEndpoint
    advertiseAddress 10.100.0.2
    bindPort 6443
discovery
  bootstrapToken
    apiServerEndpoint 10.100.0.16443
    token clvldh.vjjwg16ucnhp94qr
    caCertHashes
    - sha256a4863cde706cfc580a439f842cc65d5ef112b7b2be31628513a9881cf0d9fe0e
    # change auth info above to match the actual token and CA certificate hash for your cluster
nodeRegistration
  kubeletExtraArgs
  - name node-ip
    value 10.100.0.2,fd001234
```

`advertiseAddress` in JoinConfiguration.controlPlane specifies the IP address that the
API Server will advertise it is listening on. The value of `advertiseAddress` equals
the `--apiserver-advertise-address` flag of `kubeadm join`.

```shell
kubeadm join --configkubeadm-config.yaml
```

# # # Create a single-stack cluster

Dual-stack support doesnt mean that you need to use dual-stack addressing.
You can deploy a single-stack cluster that has the dual-stack networking feature enabled.

To make things more clear, here is an example kubeadm
[configuration file](docsreferenceconfig-apikubeadm-config.v1beta4)
`kubeadm-config.yaml` for the single-stack control plane node.

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
networking
  podSubnet 10.244.0.016
  serviceSubnet 10.96.0.016
```

# #  heading whatsnext

* [Validate IPv4IPv6 dual-stack](docstasksnetworkvalidate-dual-stack) networking
* Read about [Dual-stack](docsconceptsservices-networkingdual-stack) cluster networking
* Learn more about the kubeadm [configuration format](docsreferenceconfig-apikubeadm-config.v1beta4)
