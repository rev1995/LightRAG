---
reviewers
- caseydavenport
- danwinship
title Declare Network Policy
min-kubernetes-server-version v1.8
content_type task
weight 180
---

This document helps you get started using the Kubernetes [NetworkPolicy API](docsconceptsservices-networkingnetwork-policies) to declare network policies that govern how pods communicate with each other.

 thirdparty-content

# #  heading prerequisites

Make sure youve configured a network provider with network policy support. There are a number of network providers that support NetworkPolicy, including

* [Antrea](docstasksadminister-clusternetwork-policy-providerantrea-network-policy)
* [Calico](docstasksadminister-clusternetwork-policy-providercalico-network-policy)
* [Cilium](docstasksadminister-clusternetwork-policy-providercilium-network-policy)
* [Kube-router](docstasksadminister-clusternetwork-policy-providerkube-router-network-policy)
* [Romana](docstasksadminister-clusternetwork-policy-providerromana-network-policy)
* [Weave Net](docstasksadminister-clusternetwork-policy-providerweave-network-policy)

# # Create an `nginx` deployment and expose it via a service

To see how Kubernetes network policy works, start off by creating an `nginx` Deployment.

```console
kubectl create deployment nginx --imagenginx
```
```none
deployment.appsnginx created
```

Expose the Deployment through a Service called `nginx`.

```console
kubectl expose deployment nginx --port80
```

```none
servicenginx exposed
```

The above commands create a Deployment with an nginx Pod and expose the Deployment through a Service named `nginx`. The `nginx` Pod and Deployment are found in the `default` namespace.

```console
kubectl get svc,pod
```

```none
NAME                        CLUSTER-IP    EXTERNAL-IP   PORT(S)    AGE
servicekubernetes          10.100.0.1            443TCP    46m
servicenginx               10.100.0.16           80TCP     33s

NAME                        READY         STATUS        RESTARTS   AGE
podnginx-701339712-e0qfq   11           Running       0          35s
```

# # Test the service by accessing it from another Pod

You should be able to access the new `nginx` service from other Pods. To access the `nginx` Service from another Pod in the `default` namespace, start a busybox container

```console
kubectl run busybox --rm -ti --imagebusybox -- binsh
```

In your shell, run the following command

```shell
wget --spider --timeout1 nginx
```

```none
Connecting to nginx (10.100.0.1680)
remote file exists
```

# # Limit access to the `nginx` service

To limit the access to the `nginx` service so that only Pods with the label `access true` can query it, create a NetworkPolicy object as follows

 code_sample fileservicenetworkingnginx-policy.yaml

The name of a NetworkPolicy object must be a valid
[DNS subdomain name](docsconceptsoverviewworking-with-objectsnames#dns-subdomain-names).

NetworkPolicy includes a `podSelector` which selects the grouping of Pods to which the policy applies. You can see this policy selects Pods with the label `appnginx`. The label was automatically added to the Pod in the `nginx` Deployment. An empty `podSelector` selects all pods in the namespace.

# # Assign the policy to the service

Use kubectl to create a NetworkPolicy from the above `nginx-policy.yaml` file

```console
kubectl apply -f httpsk8s.ioexamplesservicenetworkingnginx-policy.yaml
```

```none
networkpolicy.networking.k8s.ioaccess-nginx created
```

# # Test access to the service when access label is not defined
When you attempt to access the `nginx` Service from a Pod without the correct labels, the request times out

```console
kubectl run busybox --rm -ti --imagebusybox -- binsh
```

In your shell, run the command

```shell
wget --spider --timeout1 nginx
```

```none
Connecting to nginx (10.100.0.1680)
wget download timed out
```

# # Define access label and test again

You can create a Pod with the correct labels to see that the request is allowed

```console
kubectl run busybox --rm -ti --labelsaccesstrue --imagebusybox -- binsh
```

In your shell, run the command

```shell
wget --spider --timeout1 nginx
```

```none
Connecting to nginx (10.100.0.1680)
remote file exists
```
