---
title Apply Pod Security Standards at the Namespace Level
content_type tutorial
weight 20
---

 alert titleNote
This tutorial applies only for new clusters.
 alert

Pod Security Admission is an admission controller that applies
[Pod Security Standards](docsconceptssecuritypod-security-standards)
when pods are created.  It is a feature GAed in v1.25.
In this tutorial, you will enforce the `baseline` Pod Security Standard,
one namespace at a time.

You can also apply Pod Security Standards to multiple namespaces at once at the cluster
level. For instructions, refer to
[Apply Pod Security Standards at the cluster level](docstutorialssecuritycluster-level-pss).

# #  heading prerequisites

Install the following on your workstation

- [kind](httpskind.sigs.k8s.iodocsuserquick-start#installation)
- [kubectl](docstaskstools)

# # Create cluster

1. Create a `kind` cluster as follows

   ```shell
   kind create cluster --name psa-ns-level
   ```

   The output is similar to this

   ```
   Creating cluster psa-ns-level ...
     Ensuring node image (kindestnodev)
     Preparing nodes
     Writing configuration
     Starting control-plane
     Installing CNI
     Installing StorageClass
   Set kubectl context to kind-psa-ns-level
   You can now use your cluster with

   kubectl cluster-info --context kind-psa-ns-level

   Not sure what to do next   Check out httpskind.sigs.k8s.iodocsuserquick-start
   ```

1. Set the kubectl context to the new cluster

   ```shell
   kubectl cluster-info --context kind-psa-ns-level
   ```
   The output is similar to this

   ```
   Kubernetes control plane is running at https127.0.0.150996
   CoreDNS is running at https127.0.0.150996apiv1namespaceskube-systemserviceskube-dnsdnsproxy

   To further debug and diagnose cluster problems, use kubectl cluster-info dump.
   ```

# # Create a namespace

Create a new namespace called `example`

```shell
kubectl create ns example
```

The output is similar to this

```
namespaceexample created
```

# # Enable Pod Security Standards checking for that namespace

1. Enable Pod Security Standards on this namespace using labels supported by
   built-in Pod Security Admission. In this step you will configure a check to
   warn on Pods that dont meet the latest version of the _baseline_ pod
   security standard.

   ```shell
   kubectl label --overwrite ns example
      pod-security.kubernetes.iowarnbaseline
      pod-security.kubernetes.iowarn-versionlatest
   ```

2. You can configure multiple pod security standard checks on any namespace, using labels.
   The following command will `enforce` the `baseline` Pod Security Standard, but
   `warn` and `audit` for `restricted` Pod Security Standards as per the latest
   version (default value)

   ```shell
   kubectl label --overwrite ns example
     pod-security.kubernetes.ioenforcebaseline
     pod-security.kubernetes.ioenforce-versionlatest
     pod-security.kubernetes.iowarnrestricted
     pod-security.kubernetes.iowarn-versionlatest
     pod-security.kubernetes.ioauditrestricted
     pod-security.kubernetes.ioaudit-versionlatest
   ```

# # Verify the Pod Security Standard enforcement

1. Create a baseline Pod in the `example` namespace

   ```shell
   kubectl apply -n example -f httpsk8s.ioexamplessecurityexample-baseline-pod.yaml
   ```
   The Pod does start OK the output includes a warning. For example

   ```
   Warning would violate PodSecurity restrictedlatest allowPrivilegeEscalation ! false (container nginx must set securityContext.allowPrivilegeEscalationfalse), unrestricted capabilities (container nginx must set securityContext.capabilities.drop[ALL]), runAsNonRoot ! true (pod or container nginx must set securityContext.runAsNonRoottrue), seccompProfile (pod or container nginx must set securityContext.seccompProfile.type to RuntimeDefault or Localhost)
   podnginx created
   ```

1. Create a baseline Pod in the `default` namespace

   ```shell
   kubectl apply -n default -f httpsk8s.ioexamplessecurityexample-baseline-pod.yaml
   ```
   Output is similar to this

   ```
   podnginx created
   ```

The Pod Security Standards enforcement and warning settings were applied only
to the `example` namespace. You could create the same Pod in the `default`
namespace with no warnings.

# # Clean up

Now delete the cluster which you created above by running the following command

```shell
kind delete cluster --name psa-ns-level
```

# #  heading whatsnext

- Run a
  [shell script](examplessecuritykind-with-namespace-level-baseline-pod-security.sh)
  to perform all the preceding steps all at once.

  1. Create kind cluster
  2. Create new namespace
  3. Apply `baseline` Pod Security Standard in `enforce` mode while applying
     `restricted` Pod Security Standard also in `warn` and `audit` mode.
  4. Create a new pod with the following pod security standards applied

- [Pod Security Admission](docsconceptssecuritypod-security-admission)
- [Pod Security Standards](docsconceptssecuritypod-security-standards)
- [Apply Pod Security Standards at the cluster level](docstutorialssecuritycluster-level-pss)
