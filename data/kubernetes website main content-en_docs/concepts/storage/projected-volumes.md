---
reviewers
- marosset
- jsturtevant
- zshihang
title Projected Volumes
content_type concept
weight 21 # just after persistent volumes
---

This document describes _projected volumes_ in Kubernetes. Familiarity with [volumes](docsconceptsstoragevolumes) is suggested.

# # Introduction

A `projected` volume maps several existing volume sources into the same directory.

Currently, the following types of volume sources can be projected

* [`secret`](docsconceptsstoragevolumes#secret)
* [`downwardAPI`](docsconceptsstoragevolumes#downwardapi)
* [`configMap`](docsconceptsstoragevolumes#configmap)
* [`serviceAccountToken`](#serviceaccounttoken)
* [`clusterTrustBundle`](#clustertrustbundle)

All sources are required to be in the same namespace as the Pod. For more details,
see the [all-in-one volume](httpsgit.k8s.iodesign-proposals-archivenodeall-in-one-volume.md) design document.

# # # Example configuration with a secret, a downwardAPI, and a configMap #example-configuration-secret-downwardapi-configmap

 code_sample filepodsstorageprojected-secret-downwardapi-configmap.yaml

# # # Example configuration secrets with a non-default permission mode set #example-configuration-secrets-nondefault-permission-mode

 code_sample filepodsstorageprojected-secrets-nondefault-permission-mode.yaml

Each projected volume source is listed in the spec under `sources`. The
parameters are nearly the same with two exceptions

* For secrets, the `secretName` field has been changed to `name` to be consistent
  with ConfigMap naming.
* The `defaultMode` can only be specified at the projected level and not for each
  volume source. However, as illustrated above, you can explicitly set the `mode`
  for each individual projection.

# # serviceAccountToken projected volumes #serviceaccounttoken
You can inject the token for the current [service account](docsreferenceaccess-authn-authzauthentication#service-account-tokens)
into a Pod at a specified path. For example

 code_sample filepodsstorageprojected-service-account-token.yaml

The example Pod has a projected volume containing the injected service account
token. Containers in this Pod can use that token to access the Kubernetes API
server, authenticating with the identity of [the pods ServiceAccount](docstasksconfigure-pod-containerconfigure-service-account).
The `audience` field contains the intended audience of the
token. A recipient of the token must identify itself with an identifier specified
in the audience of the token, and otherwise should reject the token. This field
is optional and it defaults to the identifier of the API server.

The `expirationSeconds` is the expected duration of validity of the service account
token. It defaults to 1 hour and must be at least 10 minutes (600 seconds). An administrator
can also limit its maximum value by specifying the `--service-account-max-token-expiration`
option for the API server. The `path` field specifies a relative path to the mount point
of the projected volume.

A container using a projected volume source as a [`subPath`](docsconceptsstoragevolumes#using-subpath)
volume mount will not receive updates for those volume sources.

# # clusterTrustBundle projected volumes #clustertrustbundle

To use this feature in Kubernetes , you must enable support for ClusterTrustBundle objects with the `ClusterTrustBundle` [feature gate](docsreferencecommand-line-tools-referencefeature-gates) and `--runtime-configcertificates.k8s.iov1beta1clustertrustbundlestrue` kube-apiserver flag, then enable the `ClusterTrustBundleProjection` feature gate.

The `clusterTrustBundle` projected volume source injects the contents of one or more [ClusterTrustBundle](docsreferenceaccess-authn-authzcertificate-signing-requests#cluster-trust-bundles) objects as an automatically-updating file in the container filesystem.

ClusterTrustBundles can be selected either by [name](docsreferenceaccess-authn-authzcertificate-signing-requests#ctb-signer-unlinked) or by [signer name](docsreferenceaccess-authn-authzcertificate-signing-requests#ctb-signer-linked).

To select by name, use the `name` field to designate a single ClusterTrustBundle object.

To select by signer name, use the `signerName` field (and optionally the
`labelSelector` field) to designate a set of ClusterTrustBundle objects that use
the given signer name. If `labelSelector` is not present, then all
ClusterTrustBundles for that signer are selected.

The kubelet deduplicates the certificates in the selected ClusterTrustBundle objects, normalizes the PEM representations (discarding comments and headers), reorders the certificates, and writes them into the file named by `path`. As the set of selected ClusterTrustBundles or their content changes, kubelet keeps the file up-to-date.

By default, the kubelet will prevent the pod from starting if the named ClusterTrustBundle is not found, or if `signerName`  `labelSelector` do not match any ClusterTrustBundles.  If this behavior is not what you want, then set the `optional` field to `true`, and the pod will start up with an empty file at `path`.

 code_sample filepodsstorageprojected-clustertrustbundle.yaml

# # SecurityContext interactions

The [proposal](httpsgit.k8s.ioenhancementskepssig-storage2451-service-account-token-volumes#proposal) for file permission handling in projected service account volume enhancement introduced the projected files having the correct owner permissions set.

# # # Linux

In Linux pods that have a projected volume and `RunAsUser` set in the Pod
[`SecurityContext`](docsreferencekubernetes-apiworkload-resourcespod-v1#security-context),
the projected files have the correct ownership set including container user
ownership.

When all containers in a pod have the same `runAsUser` set in their
[`PodSecurityContext`](docsreferencekubernetes-apiworkload-resourcespod-v1#security-context)
or container
[`SecurityContext`](docsreferencekubernetes-apiworkload-resourcespod-v1#security-context-1),
then the kubelet ensures that the contents of the `serviceAccountToken` volume are owned by that user,
and the token file has its permission mode set to `0600`.

added to a Pod after it is created do *not* change volume permissions that were
set when the pod was created.

If a Pods `serviceAccountToken` volume permissions were set to `0600` because
all other containers in the Pod have the same `runAsUser`, ephemeral
containers must use the same `runAsUser` to be able to read the token.

# # # Windows

In Windows pods that have a projected volume and `RunAsUsername` set in the
Pod `SecurityContext`, the ownership is not enforced due to the way user
accounts are managed in Windows. Windows stores and manages local user and group
accounts in a database file called Security Account Manager (SAM). Each
container maintains its own instance of the SAM database, to which the host has
no visibility into while the container is running. Windows containers are
designed to run the user mode portion of the OS in isolation from the host,
hence the maintenance of a virtual SAM database. As a result, the kubelet running
on the host does not have the ability to dynamically configure host file
ownership for virtualized container accounts. It is recommended that if files on
the host machine are to be shared with the container then they should be placed
into their own volume mount outside of `C`.

By default, the projected files will have the following ownership as shown for
an example projected volume file

```powershell
PS C Get-Acl Cvarrunsecretskubernetes.ioserviceaccount..2021_08_31_22_22_18.318230061ca.crt  Format-List

Path    Microsoft.PowerShell.CoreFileSystemCvarrunsecretskubernetes.ioserviceaccount..2021_08_31_22_22_18.318230061ca.crt
Owner   BUILTINAdministrators
Group   NT AUTHORITYSYSTEM
Access  NT AUTHORITYSYSTEM Allow  FullControl
         BUILTINAdministrators Allow  FullControl
         BUILTINUsers Allow  ReadAndExecute, Synchronize
Audit
Sddl    OBAGSYDAI(AIDFASY)(AIDFABA)(AID0x1200a9BU)
```

This implies all administrator users like `ContainerAdministrator` will have
read, write and execute access while, non-administrator users will have read and
execute access.

In general, granting the container access to the host is discouraged as it can
open the door for potential security exploits.

Creating a Windows Pod with `RunAsUser` in its `SecurityContext` will result in
the Pod being stuck at `ContainerCreating` forever. So it is advised to not use
the Linux only `RunAsUser` option with Windows Pods.
