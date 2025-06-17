---
reviewers
- lachie83
- khenidak
- bridgetkromhout
min-kubernetes-server-version v1.23
title Validate IPv4IPv6 dual-stack
content_type task
---

This document shares how to validate IPv4IPv6 dual-stack enabled Kubernetes clusters.

# #  heading prerequisites

* Provider support for dual-stack networking (Cloud provider or otherwise must be able to
  provide Kubernetes nodes with routable IPv4IPv6 network interfaces)
* A [network plugin](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins)
  that supports dual-stack networking.
* [Dual-stack enabled](docsconceptsservices-networkingdual-stack) cluster

While you can validate with an earlier version, the feature is only GA and officially supported since v1.23.

# # Validate addressing

# # # Validate node addressing

Each dual-stack Node should have a single IPv4 block and a single IPv6 block allocated.
Validate that IPv4IPv6 Pod address ranges are configured by running the following command.
Replace the sample node name with a valid dual-stack Node from your cluster. In this example,
the Nodes name is `k8s-linuxpool1-34450317-0`

```shell
kubectl get nodes k8s-linuxpool1-34450317-0 -o go-template --templaterange .spec.podCIDRsprintf sn .end
```

```
10.244.1.024
2001db864
```

There should be one IPv4 block and one IPv6 block allocated.

Validate that the node has an IPv4 and IPv6 interface detected.
Replace node name with a valid node from the cluster.
In this example the node name is `k8s-linuxpool1-34450317-0`

```shell
kubectl get nodes k8s-linuxpool1-34450317-0 -o go-template --templaterange .status.addressesprintf s sn .type .addressend
```

```
Hostname k8s-linuxpool1-34450317-0
InternalIP 10.0.0.5
InternalIP 2001db8105
```

# # # Validate Pod addressing

Validate that a Pod has an IPv4 and IPv6 address assigned. Replace the Pod name with
a valid Pod in your cluster. In this example the Pod name is `pod01`

```shell
kubectl get pods pod01 -o go-template --templaterange .status.podIPsprintf sn .ipend
```

```
10.244.1.4
2001db84
```

You can also validate Pod IPs using the Downward API via the `status.podIPs` fieldPath.
The following snippet demonstrates how you can expose the Pod IPs via an environment variable
called `MY_POD_IPS` within a container.

```yaml
        env
        - name MY_POD_IPS
          valueFrom
            fieldRef
              fieldPath status.podIPs
```

The following command prints the value of the `MY_POD_IPS` environment variable from
within a container. The value is a comma separated list that corresponds to the
Pods IPv4 and IPv6 addresses.

```shell
kubectl exec -it pod01 -- set  grep MY_POD_IPS
```

```
MY_POD_IPS10.244.1.4,2001db84
```

The Pods IP addresses will also be written to `etchosts` within a container.
The following command executes a cat on `etchosts` on a dual stack Pod.
From the output you can verify both the IPv4 and IPv6 IP address for the Pod.

```shell
kubectl exec -it pod01 -- cat etchosts
```

```
# Kubernetes-managed hosts file.
127.0.0.1    localhost
1    localhost ip6-localhost ip6-loopback
fe000    ip6-localnet
fe000    ip6-mcastprefix
fe001    ip6-allnodes
fe002    ip6-allrouters
10.244.1.4    pod01
2001db84    pod01
```

# # Validate Services

Create the following Service that does not explicitly define `.spec.ipFamilyPolicy`.
Kubernetes will assign a cluster IP for the Service from the first configured
`service-cluster-ip-range` and set the `.spec.ipFamilyPolicy` to `SingleStack`.

 code_sample fileservicenetworkingdual-stack-default-svc.yaml

Use `kubectl` to view the YAML for the Service.

```shell
kubectl get svc my-service -o yaml
```

The Service has `.spec.ipFamilyPolicy` set to `SingleStack` and `.spec.clusterIP` set
to an IPv4 address from the first configured range set via `--service-cluster-ip-range`
flag on kube-controller-manager.

```yaml
apiVersion v1
kind Service
metadata
  name my-service
  namespace default
spec
  clusterIP 10.0.217.164
  clusterIPs
  - 10.0.217.164
  ipFamilies
  - IPv4
  ipFamilyPolicy SingleStack
  ports
  - port 80
    protocol TCP
    targetPort 9376
  selector
    app.kubernetes.ioname MyApp
  sessionAffinity None
  type ClusterIP
status
  loadBalancer
```

Create the following Service that explicitly defines `IPv6` as the first array element in
`.spec.ipFamilies`. Kubernetes will assign a cluster IP for the Service from the IPv6 range
configured `service-cluster-ip-range` and set the `.spec.ipFamilyPolicy` to `SingleStack`.

 code_sample fileservicenetworkingdual-stack-ipfamilies-ipv6.yaml

Use `kubectl` to view the YAML for the Service.

```shell
kubectl get svc my-service -o yaml
```

The Service has `.spec.ipFamilyPolicy` set to `SingleStack` and `.spec.clusterIP` set to
an IPv6 address from the IPv6 range set via `--service-cluster-ip-range` flag on kube-controller-manager.

```yaml
apiVersion v1
kind Service
metadata
  labels
    app.kubernetes.ioname MyApp
  name my-service
spec
  clusterIP 2001db8fd005118
  clusterIPs
  - 2001db8fd005118
  ipFamilies
  - IPv6
  ipFamilyPolicy SingleStack
  ports
  - port 80
    protocol TCP
    targetPort 80
  selector
    app.kubernetes.ioname MyApp
  sessionAffinity None
  type ClusterIP
status
  loadBalancer
```

Create the following Service that explicitly defines `PreferDualStack` in `.spec.ipFamilyPolicy`.
Kubernetes will assign both IPv4 and IPv6 addresses (as this cluster has dual-stack enabled) and
select the `.spec.ClusterIP` from the list of `.spec.ClusterIPs` based on the address family of
the first element in the `.spec.ipFamilies` array.

 code_sample fileservicenetworkingdual-stack-preferred-svc.yaml

The `kubectl get svc` command will only show the primary IP in the `CLUSTER-IP` field.

```shell
kubectl get svc -l app.kubernetes.ionameMyApp
```

```
NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
my-service   ClusterIP   10.0.216.242           80TCP    5s
```

Validate that the Service gets cluster IPs from the IPv4 and IPv6 address blocks using
`kubectl describe`. You may then validate access to the service via the IPs and ports.

```shell
kubectl describe svc -l app.kubernetes.ionameMyApp
```

```
Name              my-service
Namespace         default
Labels            app.kubernetes.ionameMyApp
Annotations
Selector          app.kubernetes.ionameMyApp
Type              ClusterIP
IP Family Policy  PreferDualStack
IP Families       IPv4,IPv6
IP                10.0.216.242
IPs               10.0.216.242,2001db8fd00af55
Port                80TCP
TargetPort        9376TCP
Endpoints
Session Affinity  None
Events
```

# # # Create a dual-stack load balanced Service

If the cloud provider supports the provisioning of IPv6 enabled external load balancers,
create the following Service with `PreferDualStack` in `.spec.ipFamilyPolicy`, `IPv6` as
the first element of the `.spec.ipFamilies` array and the `type` field set to `LoadBalancer`.

 code_sample fileservicenetworkingdual-stack-prefer-ipv6-lb-svc.yaml

Check the Service

```shell
kubectl get svc -l app.kubernetes.ionameMyApp
```

Validate that the Service receives a `CLUSTER-IP` address from the IPv6 address block
along with an `EXTERNAL-IP`. You may then validate access to the service via the IP and port.

```
NAME         TYPE           CLUSTER-IP            EXTERNAL-IP        PORT(S)        AGE
my-service   LoadBalancer   2001db8fd007ebc   260310308055   8030790TCP   35s
```
