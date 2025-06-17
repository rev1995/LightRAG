---
title Changing The Kubernetes Package Repository
content_type task
weight 150
---

This page explains how to enable a package repository for the desired
Kubernetes minor release upon upgrading a cluster. This is only needed
for users of the community-owned package repositories hosted at `pkgs.k8s.io`.
Unlike the legacy package repositories, the community-owned package
repositories are structured in a way that theres a dedicated package
repository for each Kubernetes minor version.

This guide only covers a part of the Kubernetes upgrade process. Please see the
[upgrade guide](docstasksadminister-clusterkubeadmkubeadm-upgrade) for
more information about upgrading Kubernetes clusters.

This step is only needed upon upgrading a cluster to another **minor** release.
If youre upgrading to another patch release within the same minor release (e.g.
v.5 to v.7), you dont
need to follow this guide. However, if youre still using the legacy package
repositories, youll need to migrate to the new community-owned package
repositories before upgrading (see the next section for more details on how to
do this).

# #  heading prerequisites

This document assumes that youre already using the community-owned
package repositories (`pkgs.k8s.io`). If thats not the case, its strongly
recommended to migrate to the community-owned package repositories as described
in the [official announcement](blog20230815pkgs-k8s-io-introduction).

 legacy-repos-deprecation

# # # Verifying if the Kubernetes package repositories are used

If youre unsure whether youre using the community-owned package repositories or the
legacy package repositories, take the following steps to verify

 tab nameUbuntu, Debian or HypriotOS

Print the contents of the file that defines the Kubernetes `apt` repository

```shell
# On your system, this configuration file could have a different name
pager etcaptsources.list.dkubernetes.list
```

If you see a line similar to

```
deb [signed-byetcaptkeyringskubernetes-apt-keyring.gpg] httpspkgs.k8s.iocorestablevdeb
```

**Youre using the Kubernetes package repositories and this guide applies to you.**
Otherwise, its strongly recommended to migrate to the Kubernetes package repositories
as described in the [official announcement](blog20230815pkgs-k8s-io-introduction).

 tab
 tab nameCentOS, RHEL or Fedora

Print the contents of the file that defines the Kubernetes `yum` repository

```shell
# On your system, this configuration file could have a different name
cat etcyum.repos.dkubernetes.repo
```

If you see a `baseurl` similar to the `baseurl` in the output below

```
[kubernetes]
nameKubernetes
baseurlhttpspkgs.k8s.iocorestablevrpm
enabled1
gpgcheck1
gpgkeyhttpspkgs.k8s.iocorestablevrpmrepodatarepomd.xml.key
excludekubelet kubeadm kubectl
```

**Youre using the Kubernetes package repositories and this guide applies to you.**
Otherwise, its strongly recommended to migrate to the Kubernetes package repositories
as described in the [official announcement](blog20230815pkgs-k8s-io-introduction).

 tab

 tab nameopenSUSE or SLES

Print the contents of the file that defines the Kubernetes `zypper` repository

```shell
# On your system, this configuration file could have a different name
cat etczypprepos.dkubernetes.repo
```

If you see a `baseurl` similar to the `baseurl` in the output below

```
[kubernetes]
nameKubernetes
baseurlhttpspkgs.k8s.iocorestablevrpm
enabled1
gpgcheck1
gpgkeyhttpspkgs.k8s.iocorestablevrpmrepodatarepomd.xml.key
excludekubelet kubeadm kubectl
```

**Youre using the Kubernetes package repositories and this guide applies to you.**
Otherwise, its strongly recommended to migrate to the Kubernetes package repositories
as described in the [official announcement](blog20230815pkgs-k8s-io-introduction).

 tab

The URL used for the Kubernetes package repositories is not limited to `pkgs.k8s.io`,
it can also be one of

- `pkgs.k8s.io`
- `pkgs.kubernetes.io`
- `packages.kubernetes.io`

# # Switching to another Kubernetes package repository

This step should be done upon upgrading from one to another Kubernetes minor
release in order to get access to the packages of the desired Kubernetes minor
version.

 tab nameUbuntu, Debian or HypriotOS

1. Open the file that defines the Kubernetes `apt` repository using a text editor of your choice

   ```shell
   nano etcaptsources.list.dkubernetes.list
   ```

   You should see a single line with the URL that contains your current Kubernetes
   minor version. For example, if youre using v,
   you should see this

   ```
   deb [signed-byetcaptkeyringskubernetes-apt-keyring.gpg] httpspkgs.k8s.iocorestablevdeb
   ```

1. Change the version in the URL to **the next available minor release**, for example

   ```
   deb [signed-byetcaptkeyringskubernetes-apt-keyring.gpg] httpspkgs.k8s.iocorestabledeb
   ```

1. Save the file and exit your text editor. Continue following the relevant upgrade instructions.

 tab
 tab nameCentOS, RHEL or Fedora

1. Open the file that defines the Kubernetes `yum` repository using a text editor of your choice

   ```shell
   nano etcyum.repos.dkubernetes.repo
   ```

   You should see a file with two URLs that contain your current Kubernetes
   minor version. For example, if youre using v,
   you should see this

   ```
   [kubernetes]
   nameKubernetes
   baseurlhttpspkgs.k8s.iocorestablevrpm
   enabled1
   gpgcheck1
   gpgkeyhttpspkgs.k8s.iocorestablevrpmrepodatarepomd.xml.key
   excludekubelet kubeadm kubectl cri-tools kubernetes-cni
   ```

1. Change the version in these URLs to **the next available minor release**, for example

   ```
   [kubernetes]
   nameKubernetes
   baseurlhttpspkgs.k8s.iocorestablerpm
   enabled1
   gpgcheck1
   gpgkeyhttpspkgs.k8s.iocorestablerpmrepodatarepomd.xml.key
   excludekubelet kubeadm kubectl cri-tools kubernetes-cni
   ```

1. Save the file and exit your text editor. Continue following the relevant upgrade instructions.

 tab

# #  heading whatsnext

* See how to [Upgrade Linux nodes](docstasksadminister-clusterkubeadmupgrading-linux-nodes).
* See how to [Upgrade Windows nodes](docstasksadminister-clusterkubeadmupgrading-windows-nodes).
