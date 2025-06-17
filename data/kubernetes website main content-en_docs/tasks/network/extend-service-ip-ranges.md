---
reviewers
- thockin
- dwinship
min-kubernetes-server-version v1.29
title Extend Service IP Ranges
content_type task
---

This document shares how to extend the existing Service IP range assigned to a cluster.

# #  heading prerequisites

While you can use this feature with an earlier version, the feature is only GA and officially supported since v1.33.

# # Extend Service IP Ranges

Kubernetes clusters with kube-apiservers that have enabled the `MultiCIDRServiceAllocator`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates) and have the
`networking.k8s.iov1beta1` API group active, will create a ServiceCIDR object that takes
the well-known name `kubernetes`, and that specifies an IP address range
based on the value of the `--service-cluster-ip-range` command line argument to kube-apiserver.

```sh
kubectl get servicecidr
```

```
NAME         CIDRS          AGE
kubernetes   10.96.0.028   17d
```

The well-known `kubernetes` Service, that exposes the kube-apiserver endpoint to the Pods, calculates
the first IP address from the default ServiceCIDR range and uses that IP address as its
cluster IP address.

```sh
kubectl get service kubernetes
```

```
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1            443TCP   17d
```

The default Service, in this case, uses the ClusterIP 10.96.0.1, that has the corresponding IPAddress object.

```sh
kubectl get ipaddress 10.96.0.1
```

```
NAME        PARENTREF
10.96.0.1   servicesdefaultkubernetes
```

The ServiceCIDRs are protected with ,
to avoid leaving Service ClusterIPs orphans the finalizer is only removed if there is another subnet
that contains the existing IPAddresses or there are no IPAddresses belonging to the subnet.

# # Extend the number of available IPs for Services

There are cases that users will need to increase the number addresses available to Services,
previously, increasing the Service range was a disruptive operation that could also cause data loss.
With this new feature users only need to add a new ServiceCIDR to increase the number of available addresses.

# # # Adding a new ServiceCIDR

On a cluster with a 10.96.0.028 range for Services, there is only 2(32-28) - 2  14
IP addresses available. The `kubernetes.default` Service is always created for this example,
that leaves you with only 13 possible Services.

```sh
for i in (seq 1 13) do kubectl create service clusterip test-i --tcp 80 -o json  jq -r .spec.clusterIP done
```

```
10.96.0.11
10.96.0.5
10.96.0.12
10.96.0.13
10.96.0.14
10.96.0.2
10.96.0.3
10.96.0.4
10.96.0.6
10.96.0.7
10.96.0.8
10.96.0.9
error failed to create ClusterIP service Internal error occurred failed to allocate a serviceIP range is full
```

You can increase the number of IP addresses available for Services, by creating a new ServiceCIDR
that extends or adds new IP address ranges.

```sh
cat
The default kubernetes ServiceCIDR is created by the kube-apiserver
to provide consistency in the cluster and is required for the cluster to work,
so it always must be allowed. You can ensure your `ValidatingAdmissionPolicy`
doesnt restrict the default ServiceCIDR by adding the clause

```yaml
  matchConditions
  - name exclude-default-servicecidr
    expression object.metadata.name ! kubernetes
```

as in the examples below.

# # # # Restrict Service CIDR ranges to some specific ranges

The following is an example of a `ValidatingAdmissionPolicy` that only allows
ServiceCIDRs to be created if they are subranges of the given `allowed` ranges.
(So the example policy would allow a ServiceCIDR with `cidrs [10.96.1.024]`
or `cidrs [2001db800ffff80, 10.96.0.020]` but would not allow a
ServiceCIDR with `cidrs [172.20.0.016]`.) You can copy this policy and change
the value of `allowed` to something appropriate for you cluster.

```yaml
apiVersion admissionregistration.k8s.iov1
kind ValidatingAdmissionPolicy
metadata
  name servicecidrs.default
spec
  failurePolicy Fail
  matchConstraints
    resourceRules
    - apiGroups   [networking.k8s.io]
      apiVersions [v1,v1beta1]
      operations  [CREATE, UPDATE]
      resources   [servicecidrs]
  matchConditions
  - name exclude-default-servicecidr
    expression object.metadata.name ! kubernetes
  variables
  - name allowed
    expression [10.96.0.016,2001db864]
  validations
  - expression object.spec.cidrs.all(newCIDR, variables.allowed.exists(allowedCIDR, cidr(allowedCIDR).containsCIDR(newCIDR)))
  # For all CIDRs (newCIDR) listed in the spec.cidrs of the submitted ServiceCIDR
  # object, check if there exists at least one CIDR (allowedCIDR) in the `allowed`
  # list of the VAP such that the allowedCIDR fully contains the newCIDR.
---
apiVersion admissionregistration.k8s.iov1
kind ValidatingAdmissionPolicyBinding
metadata
  name servicecidrs-binding
spec
  policyName servicecidrs.default
  validationActions [Deny,Audit]
```

Consult the [CEL documentation](httpskubernetes.iodocsreferenceusing-apicel)
to learn more about CEL if you want to write your own validation `expression`.

# # # # Restrict any usage of the ServiceCIDR API

The following example demonstrates how to use a `ValidatingAdmissionPolicy` and
its binding to restrict the creation of any new Service CIDR ranges, excluding the default kubernetes ServiceCIDR

```yaml
apiVersion admissionregistration.k8s.iov1
kind ValidatingAdmissionPolicy
metadata
  name servicecidrs.deny
spec
  failurePolicy Fail
  matchConstraints
    resourceRules
    - apiGroups   [networking.k8s.io]
      apiVersions [v1,v1beta1]
      operations  [CREATE, UPDATE]
      resources   [servicecidrs]
  validations
  - expression object.metadata.name  kubernetes
---
apiVersion admissionregistration.k8s.iov1
kind ValidatingAdmissionPolicyBinding
metadata
  name servicecidrs-deny-binding
spec
  policyName servicecidrs.deny
  validationActions [Deny,Audit]
```
