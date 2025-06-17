---
reviewers
- sig-cluster-lifecycle
title Customizing components with the kubeadm API
content_type concept
weight 40
---

This page covers how to customize the components that kubeadm deploys. For control plane components
you can use flags in the `ClusterConfiguration` structure or patches per-node. For the kubelet
and kube-proxy you can use `KubeletConfiguration` and `KubeProxyConfiguration`, accordingly.

All of these options are possible via the kubeadm configuration API.
For more details on each field in the configuration you can navigate to our
[API reference pages](docsreferenceconfig-apikubeadm-config.v1beta4).

Customizing the CoreDNS deployment of kubeadm is currently not supported. You must manually
patch the `kube-systemcoredns`
and recreate the CoreDNS  after that. Alternatively,
you can skip the default CoreDNS deployment and deploy your own variant.
For more details on that see [Using init phases with kubeadm](docsreferencesetup-toolskubeadmkubeadm-init#init-phases).

To reconfigure a cluster that has already been created see
[Reconfiguring a kubeadm cluster](docstasksadminister-clusterkubeadmkubeadm-reconfigure).

# # Customizing the control plane with flags in `ClusterConfiguration`

The kubeadm `ClusterConfiguration` object exposes a way for users to override the default
flags passed to control plane components such as the APIServer, ControllerManager, Scheduler and Etcd.
The components are defined using the following structures

- `apiServer`
- `controllerManager`
- `scheduler`
- `etcd`

These structures contain a common `extraArgs` field, that consists of `name`  `value` pairs.
To override a flag for a control plane component

1.  Add the appropriate `extraArgs` to your configuration.
2.  Add flags to the `extraArgs` field.
3.  Run `kubeadm init` with `--config `.

You can generate a `ClusterConfiguration` object with default values by running `kubeadm config print init-defaults`
and saving the output to a file of your choice.

The `ClusterConfiguration` object is currently global in kubeadm clusters. This means that any flags that you add,
will apply to all instances of the same component on different nodes. To apply individual configuration per component
on different nodes you can use [patches](#patches).

Duplicate flags (keys), or passing the same flag `--foo` multiple times, is currently not supported.
To workaround that you must use [patches](#patches).

# # # APIServer flags

For details, see the [reference documentation for kube-apiserver](docsreferencecommand-line-tools-referencekube-apiserver).

Example usage

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
kubernetesVersion v1.16.0
apiServer
  extraArgs
  - name enable-admission-plugins
    value AlwaysPullImages,DefaultStorageClass
  - name audit-log-path
    value homejohndoeaudit.log
```

# # # ControllerManager flags

For details, see the [reference documentation for kube-controller-manager](docsreferencecommand-line-tools-referencekube-controller-manager).

Example usage

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
kubernetesVersion v1.16.0
controllerManager
  extraArgs
  - name cluster-signing-key-file
    value homejohndoekeysca.key
  - name deployment-controller-sync-period
    value 50
```

# # # Scheduler flags

For details, see the [reference documentation for kube-scheduler](docsreferencecommand-line-tools-referencekube-scheduler).

Example usage

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
kubernetesVersion v1.16.0
scheduler
  extraArgs
  - name config
    value etckubernetesscheduler-config.yaml
  extraVolumes
    - name schedulerconfig
      hostPath homejohndoeschedconfig.yaml
      mountPath etckubernetesscheduler-config.yaml
      readOnly true
      pathType File
```

# # # Etcd flags

For details, see the [etcd server documentation](httpsetcd.iodocs).

Example usage

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
etcd
  local
    extraArgs
    - name election-timeout
      value 1000
```

# # Customizing with patches #patches

Kubeadm allows you to pass a directory with patch files to `InitConfiguration` and `JoinConfiguration`
on individual nodes. These patches can be used as the last customization step before component configuration
is written to disk.

You can pass this file to `kubeadm init` with `--config `

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind InitConfiguration
patches
  directory homeusersomedir
```

For `kubeadm init` you can pass a file containing both a `ClusterConfiguration` and `InitConfiguration`
separated by `---`.

You can pass this file to `kubeadm join` with `--config `

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind JoinConfiguration
patches
  directory homeusersomedir
```

The directory must contain files named `target[suffix][patchtype].extension`.
For example, `kube-apiserver0merge.yaml` or just `etcd.json`.

- `target` can be one of `kube-apiserver`, `kube-controller-manager`, `kube-scheduler`, `etcd`
and `kubeletconfiguration`.
- `suffix` is an optional string that can be used to determine which patches are applied first
alpha-numerically.
- `patchtype` can be one of `strategic`, `merge` or `json` and these must match the patching formats
[supported by kubectl](docstasksmanage-kubernetes-objectsupdate-api-object-kubectl-patch).
The default `patchtype` is `strategic`.
- `extension` must be either `json` or `yaml`.

If you are using `kubeadm upgrade` to upgrade your kubeadm nodes you must again provide the same
patches, so that the customization is preserved after upgrade. To do that you can use the `--patches`
flag, which must point to the same directory. `kubeadm upgrade` currently does not support a configuration
API structure that can be used for the same purpose.

# # Customizing the kubelet #kubelet

To customize the kubelet you can add a [`KubeletConfiguration`](docsreferenceconfig-apikubelet-config.v1beta1)
next to the `ClusterConfiguration` or `InitConfiguration` separated by `---` within the same configuration file.
This file can then be passed to `kubeadm init` and kubeadm will apply the same base `KubeletConfiguration`
to all nodes in the cluster.

For applying instance-specific configuration over the base `KubeletConfiguration` you can use the
[`kubeletconfiguration` patch target](#patches).

Alternatively, you can use kubelet flags as overrides by passing them in the
`nodeRegistration.kubeletExtraArgs` field supported by both `InitConfiguration` and `JoinConfiguration`.
Some kubelet flags are deprecated, so check their status in the
[kubelet reference documentation](docsreferencecommand-line-tools-referencekubelet) before using them.

For additional details see [Configuring each kubelet in your cluster using kubeadm](docssetupproduction-environmenttoolskubeadmkubelet-integration)

# # Customizing kube-proxy

To customize kube-proxy you can pass a `KubeProxyConfiguration` next your `ClusterConfiguration` or
`InitConfiguration` to `kubeadm init` separated by `---`.

For more details you can navigate to our [API reference pages](docsreferenceconfig-apikubeadm-config.v1beta4).

kubeadm deploys kube-proxy as a , which means
that the `KubeProxyConfiguration` would apply to all instances of kube-proxy in the cluster.
