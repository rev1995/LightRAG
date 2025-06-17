---
reviewers
- erictune
- mikedanese
- thockin
title Configure a Security Context for a Pod or Container
content_type task
weight 110
---

A security context defines privilege and access control settings for
a Pod or Container. Security context settings include, but are not limited to

* Discretionary Access Control Permission to access an object, like a file, is based on
  [user ID (UID) and group ID (GID)](httpswiki.archlinux.orgindex.phpusers_and_groups).

* [Security Enhanced Linux (SELinux)](httpsen.wikipedia.orgwikiSecurity-Enhanced_Linux)
  Objects are assigned security labels.

* Running as privileged or unprivileged.

* [Linux Capabilities](httpslinux-audit.comlinux-capabilities-hardening-linux-binaries-by-removing-setuid)
  Give a process some privileges, but not all the privileges of the root user.

* [AppArmor](docstutorialssecurityapparmor)
  Use program profiles to restrict the capabilities of individual programs.

* [Seccomp](docstutorialssecurityseccomp) Filter a processs system calls.

* `allowPrivilegeEscalation` Controls whether a process can gain more privileges than
  its parent process. This bool directly controls whether the
  [`no_new_privs`](httpswww.kernel.orgdocDocumentationprctlno_new_privs.txt)
  flag gets set on the container process.
  `allowPrivilegeEscalation` is always true when the container

  - is run as privileged, or
  - has `CAP_SYS_ADMIN`

* `readOnlyRootFilesystem` Mounts the containers root filesystem as read-only.

The above bullets are not a complete set of security context settings -- please see
[SecurityContext](docsreferencegeneratedkubernetes-api#securitycontext-v1-core)
for a comprehensive list.

# #  heading prerequisites

# # Set the security context for a Pod

To specify security settings for a Pod, include the `securityContext` field
in the Pod specification. The `securityContext` field is a
[PodSecurityContext](docsreferencegeneratedkubernetes-api#podsecuritycontext-v1-core) object.
The security settings that you specify for a Pod apply to all Containers in the Pod.
Here is a configuration file for a Pod that has a `securityContext` and an `emptyDir` volume

 code_sample filepodssecuritysecurity-context.yaml

In the configuration file, the `runAsUser` field specifies that for any Containers in
the Pod, all processes run with user ID 1000. The `runAsGroup` field specifies the primary group ID of 3000 for
all processes within any containers of the Pod. If this field is omitted, the primary group ID of the containers
will be root(0). Any files created will also be owned by user 1000 and group 3000 when `runAsGroup` is specified.
Since `fsGroup` field is specified, all processes of the container are also part of the supplementary group ID 2000.
The owner for volume `datademo` and any files created in that volume will be Group ID 2000.
Additionally, when the `supplementalGroups` field is specified, all processes of the container are also part of the
specified groups. If this field is omitted, it means empty.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context.yaml
```

Verify that the Pods Container is running

```shell
kubectl get pod security-context-demo
```

Get a shell to the running Container

```shell
kubectl exec -it security-context-demo -- sh
```

In your shell, list the running processes

```shell
ps
```

The output shows that the processes are running as user 1000, which is the value of `runAsUser`

```none
PID   USER     TIME  COMMAND
    1 1000      000 sleep 1h
    6 1000      000 sh
...
```

In your shell, navigate to `data`, and list the one directory

```shell
cd data
ls -l
```

The output shows that the `datademo` directory has group ID 2000, which is
the value of `fsGroup`.

```none
drwxrwsrwx 2 root 2000 4096 Jun  6 2008 demo
```

In your shell, navigate to `datademo`, and create a file

```shell
cd demo
echo hello  testfile
```

List the file in the `datademo` directory

```shell
ls -l
```

The output shows that `testfile` has group ID 2000, which is the value of `fsGroup`.

```none
-rw-r--r-- 1 1000 2000 6 Jun  6 2008 testfile
```

Run the following command

```shell
id
```

The output is similar to this

```none
uid1000 gid3000 groups2000,3000,4000
```

From the output, you can see that `gid` is 3000 which is same as the `runAsGroup` field.
If the `runAsGroup` was omitted, the `gid` would remain as 0 (root) and the process will
be able to interact with files that are owned by the root(0) group and groups that have
the required group permissions for the root (0) group. You can also see that `groups`
contains the group IDs which are specified by `fsGroup` and `supplementalGroups`,
in addition to `gid`.

Exit your shell

```shell
exit
```

# # # Implicit group memberships defined in `etcgroup` in the container image

By default, kubernetes merges group information from the Pod with information defined in `etcgroup` in the container image.

 code_sample filepodssecuritysecurity-context-5.yaml

This Pod security context contains `runAsUser`, `runAsGroup` and `supplementalGroups`.
However, you can see that the actual supplementary groups attached to the container process
will include group IDs which come from `etcgroup` in the container image.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context-5.yaml
```

Verify that the Pods Container is running

```shell
kubectl get pod security-context-demo
```

Get a shell to the running Container

```shell
kubectl exec -it security-context-demo -- sh
```

Check the process identity

```shell
 id
```

The output is similar to this

```none
uid1000 gid3000 groups3000,4000,50000
```

You can see that `groups` includes group ID `50000`. This is because the user (`uid1000`),
which is defined in the image, belongs to the group (`gid50000`), which is defined in `etcgroup`
inside the container image.

Check the `etcgroup` in the container image

```shell
 cat etcgroup
```

You can see that uid `1000` belongs to group `50000`.

```none
...
user-defined-in-imagex1000
group-defined-in-imagex50000user-defined-in-image
```

Exit your shell

```shell
exit
```

_Implicitly merged_ supplementary groups may cause security problems particularly when accessing
the volumes (see [kuberneteskubernetes#112879](httpsissue.k8s.io112879) for details).
If you want to avoid this. Please see the below section.

# # Configure fine-grained SupplementalGroups control for a Pod #supplementalgroupspolicy

This feature can be enabled by setting the `SupplementalGroupsPolicy`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates) for kubelet and
kube-apiserver, and setting the `.spec.securityContext.supplementalGroupsPolicy` field for a pod.

The `supplementalGroupsPolicy` field defines the policy for calculating the
supplementary groups for the container processes in a pod. There are two valid
values for this field

* `Merge` The group membership defined in `etcgroup` for the containers primary user will be merged.
  This is the default policy if not specified.

* `Strict` Only group IDs in `fsGroup`, `supplementalGroups`, or `runAsGroup` fields
  are attached as the supplementary groups of the container processes.
  This means no group membership from `etcgroup` for the containers primary user will be merged.

When the feature is enabled, it also exposes the process identity attached to the first container process
in `.status.containerStatuses[].user.linux` field. It would be useful for detecting if
implicit group IDs are attached.

 code_sample filepodssecuritysecurity-context-6.yaml

This pod manifest defines `supplementalGroupsPolicyStrict`. You can see that no group memberships
defined in `etcgroup` are merged to the supplementary groups for container processes.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context-6.yaml
```

Verify that the Pods Container is running

```shell
kubectl get pod security-context-demo
```

Check the process identity

```shell
kubectl exec -it security-context-demo -- id
```

The output is similar to this

```none
uid1000 gid3000 groups3000,4000
```

See the Pods status

```shell
kubectl get pod security-context-demo -o yaml
```

You can see that the `status.containerStatuses[].user.linux` field exposes the process identitiy
attached to the first container process.

```none
...
status
  containerStatuses
  - name sec-ctx-demo
    user
      linux
        gid 3000
        supplementalGroups
        - 3000
        - 4000
        uid 1000
...
```

Please note that the values in the `status.containerStatuses[].user.linux` field is _the first attached_
process identity to the first container process in the container. If the container has sufficient privilege
to make system calls related to process identity
(e.g. [`setuid(2)`](httpsman7.orglinuxman-pagesman2setuid.2.html),
[`setgid(2)`](httpsman7.orglinuxman-pagesman2setgid.2.html) or
[`setgroups(2)`](httpsman7.orglinuxman-pagesman2setgroups.2.html), etc.),
the container process can change its identity. Thus, the _actual_ process identity will be dynamic.

# # # Implementations #implementations-supplementalgroupspolicy

 thirdparty-content

The following container runtimes are known to support fine-grained SupplementalGroups control.

CRI-level
- [containerd](httpscontainerd.io), since v2.0
- [CRI-O](httpscri-o.io), since v1.31

You can see if the feature is supported in the Node status.

```yaml
apiVersion v1
kind Node
...
status
  features
    supplementalGroupsPolicy true
```

At this alpha release(from v1.31 to v1.32), when a pod with `SupplementalGroupsPolicyStrict` are scheduled to a node that does NOT support this feature(i.e. `.status.features.supplementalGroupsPolicyfalse`), the pods supplemental groups policy falls back to the `Merge` policy _silently_.

However, since the beta release (v1.33), to enforce the policy more strictly, __such pod creation will be rejected by kubelet because the node cannot ensure the specified policy__. When your pod is rejected, you will see warning events with `reasonSupplementalGroupsPolicyNotSupported` like below

```yaml
apiVersion v1
kind Event
...
type Warning
reason SupplementalGroupsPolicyNotSupported
message SupplementalGroupsPolicyStrict is not supported in this node
involvedObject
  apiVersion v1
  kind Pod
  ...
```

# # Configure volume permission and ownership change policy for Pods

By default, Kubernetes recursively changes ownership and permissions for the contents of each
volume to match the `fsGroup` specified in a Pods `securityContext` when that volume is
mounted.
For large volumes, checking and changing ownership and permissions can take a lot of time,
slowing Pod startup. You can use the `fsGroupChangePolicy` field inside a `securityContext`
to control the way that Kubernetes checks and manages ownership and permissions
for a volume.

**fsGroupChangePolicy** - `fsGroupChangePolicy` defines behavior for changing ownership
  and permission of the volume before being exposed inside a Pod.
  This field only applies to volume types that support `fsGroup` controlled ownership and permissions.
  This field has two possible values

* _OnRootMismatch_ Only change permissions and ownership if the permission and the ownership of
  root directory does not match with expected permissions of the volume.
  This could help shorten the time it takes to change ownership and permission of a volume.
* _Always_ Always change permission and ownership of the volume when volume is mounted.

For example

```yaml
securityContext
  runAsUser 1000
  runAsGroup 3000
  fsGroup 2000
  fsGroupChangePolicy OnRootMismatch
```

This field has no effect on ephemeral volume types such as
[`secret`](docsconceptsstoragevolumes#secret),
[`configMap`](docsconceptsstoragevolumes#configmap),
and [`emptydir`](docsconceptsstoragevolumes#emptydir).

# # Delegating volume permission and ownership change to CSI driver

If you deploy a [Container Storage Interface (CSI)](httpsgithub.comcontainer-storage-interfacespecblobmasterspec.md)
driver which supports the `VOLUME_MOUNT_GROUP` `NodeServiceCapability`, the
process of setting file ownership and permissions based on the
`fsGroup` specified in the `securityContext` will be performed by the CSI driver
instead of Kubernetes. In this case, since Kubernetes doesnt perform any
ownership and permission change, `fsGroupChangePolicy` does not take effect, and
as specified by CSI, the driver is expected to mount the volume with the
provided `fsGroup`, resulting in a volume that is readablewritable by the
`fsGroup`.

# # Set the security context for a Container

To specify security settings for a Container, include the `securityContext` field
in the Container manifest. The `securityContext` field is a
[SecurityContext](docsreferencegeneratedkubernetes-api#securitycontext-v1-core) object.
Security settings that you specify for a Container apply only to
the individual Container, and they override settings made at the Pod level when
there is overlap. Container settings do not affect the Pods Volumes.

Here is the configuration file for a Pod that has one Container. Both the Pod
and the Container have a `securityContext` field

 code_sample filepodssecuritysecurity-context-2.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context-2.yaml
```

Verify that the Pods Container is running

```shell
kubectl get pod security-context-demo-2
```

Get a shell into the running Container

```shell
kubectl exec -it security-context-demo-2 -- sh
```

In your shell, list the running processes

```shell
ps aux
```

The output shows that the processes are running as user 2000. This is the value
of `runAsUser` specified for the Container. It overrides the value 1000 that is
specified for the Pod.

```
USER       PID CPU MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
2000         1  0.0  0.0   4336   764         Ss   2036   000 binsh -c node server.js
2000         8  0.1  0.5 772124 22604         Sl   2036   000 node server.js
...
```

Exit your shell

```shell
exit
```

# # Set capabilities for a Container

With [Linux capabilities](httpsman7.orglinuxman-pagesman7capabilities.7.html),
you can grant certain privileges to a process without granting all the privileges
of the root user. To add or remove Linux capabilities for a Container, include the
`capabilities` field in the `securityContext` section of the Container manifest.

First, see what happens when you dont include a `capabilities` field.
Here is configuration file that does not add or remove any Container capabilities

 code_sample filepodssecuritysecurity-context-3.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context-3.yaml
```

Verify that the Pods Container is running

```shell
kubectl get pod security-context-demo-3
```

Get a shell into the running Container

```shell
kubectl exec -it security-context-demo-3 -- sh
```

In your shell, list the running processes

```shell
ps aux
```

The output shows the process IDs (PIDs) for the Container

```
USER  PID CPU MEM    VSZ   RSS TTY   STAT START   TIME COMMAND
root    1  0.0  0.0   4336   796      Ss   1817   000 binsh -c node server.js
root    5  0.1  0.5 772124 22700      Sl   1817   000 node server.js
```

In your shell, view the status for process 1

```shell
cd proc1
cat status
```

The output shows the capabilities bitmap for the process

```
...
CapPrm	00000000a80425fb
CapEff	00000000a80425fb
...
```

Make a note of the capabilities bitmap, and then exit your shell

```shell
exit
```

Next, run a Container that is the same as the preceding container, except
that it has additional capabilities set.

Here is the configuration file for a Pod that runs one Container. The configuration
adds the `CAP_NET_ADMIN` and `CAP_SYS_TIME` capabilities

 code_sample filepodssecuritysecurity-context-4.yaml

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodssecuritysecurity-context-4.yaml
```

Get a shell into the running Container

```shell
kubectl exec -it security-context-demo-4 -- sh
```

In your shell, view the capabilities for process 1

```shell
cd proc1
cat status
```

The output shows capabilities bitmap for the process

```
...
CapPrm	00000000aa0435fb
CapEff	00000000aa0435fb
...
```

Compare the capabilities of the two Containers

```
00000000a80425fb
00000000aa0435fb
```

In the capability bitmap of the first container, bits 12 and 25 are clear. In the second container,
bits 12 and 25 are set. Bit 12 is `CAP_NET_ADMIN`, and bit 25 is `CAP_SYS_TIME`.
See [capability.h](httpsgithub.comtorvaldslinuxblobmasterincludeuapilinuxcapability.h)
for definitions of the capability constants.

Linux capability constants have the form `CAP_XXX`.
But when you list capabilities in your container manifest, you must
omit the `CAP_` portion of the constant.
For example, to add `CAP_SYS_TIME`, include `SYS_TIME` in your list of capabilities.

# # Set the Seccomp Profile for a Container

To set the Seccomp profile for a Container, include the `seccompProfile` field
in the `securityContext` section of your Pod or Container manifest. The
`seccompProfile` field is a
[SeccompProfile](docsreferencegeneratedkubernetes-api#seccompprofile-v1-core) object consisting of `type` and `localhostProfile`.
Valid options for `type` include `RuntimeDefault`, `Unconfined`, and
`Localhost`. `localhostProfile` must only be set if `type Localhost`. It
indicates the path of the pre-configured profile on the node, relative to the
kubelets configured Seccomp profile location (configured with the `--root-dir`
flag).

Here is an example that sets the Seccomp profile to the nodes container runtime
default profile

```yaml
...
securityContext
  seccompProfile
    type RuntimeDefault
```

Here is an example that sets the Seccomp profile to a pre-configured file at
`seccompmy-profilesprofile-allow.json`

```yaml
...
securityContext
  seccompProfile
    type Localhost
    localhostProfile my-profilesprofile-allow.json
```

# # Set the AppArmor Profile for a Container

To set the AppArmor profile for a Container, include the `appArmorProfile` field
in the `securityContext` section of your Container. The `appArmorProfile` field
is a
[AppArmorProfile](docsreferencegeneratedkubernetes-api#apparmorprofile-v1-core) object consisting of `type` and `localhostProfile`.
Valid options for `type` include `RuntimeDefault`(default), `Unconfined`, and
`Localhost`. `localhostProfile` must only be set if `type` is `Localhost`. It
indicates the name of the pre-configured profile on the node. The profile needs
to be loaded onto all nodes suitable for the Pod, since you dont know where the
pod will be scheduled.
Approaches for setting up custom profiles are discussed in
[Setting up nodes with profiles](docstutorialssecurityapparmor#setting-up-nodes-with-profiles).

Note If `containers[*].securityContext.appArmorProfile.type` is explicitly set
to `RuntimeDefault`, then the Pod will not be admitted if AppArmor is not
enabled on the Node. However if `containers[*].securityContext.appArmorProfile.type`
is not specified, then the default (which is also `RuntimeDefault`) will only
be applied if the node has AppArmor enabled. If the node has AppArmor disabled
the Pod will be admitted but the Container will not be restricted by the
`RuntimeDefault` profile.

Here is an example that sets the AppArmor profile to the nodes container runtime
default profile

```yaml
...
containers
- name container-1
  securityContext
    appArmorProfile
      type RuntimeDefault
```

Here is an example that sets the AppArmor profile to a pre-configured profile
named `k8s-apparmor-example-deny-write`

```yaml
...
containers
- name container-1
  securityContext
    appArmorProfile
      type Localhost
      localhostProfile k8s-apparmor-example-deny-write
```

For more details please see, [Restrict a Containers Access to Resources with AppArmor](docstutorialssecurityapparmor).

# # Assign SELinux labels to a Container

To assign SELinux labels to a Container, include the `seLinuxOptions` field in
the `securityContext` section of your Pod or Container manifest. The
`seLinuxOptions` field is an
[SELinuxOptions](docsreferencegeneratedkubernetes-api#selinuxoptions-v1-core)
object. Heres an example that applies an SELinux level

```yaml
...
securityContext
  seLinuxOptions
    level s0c123,c456
```

To assign SELinux labels, the SELinux security module must be loaded on the host operating system.
On Windows and Linux worker nodes without SELinux support, this field and any SELinux feature gates described
below have no effect.

# # # Efficient SELinux volume relabeling

Kubernetes v1.27 introduced an early limited form of this behavior that was only applicable
to volumes (and PersistentVolumeClaims) using the `ReadWriteOncePod` access mode.

Kubernetes v1.33 promotes `SELinuxChangePolicy` and `SELinuxMount`
[feature gates](docsreferencecommand-line-tools-referencefeature-gates)
as beta to widen that performance improvement to other kinds of PersistentVolumeClaims,
as explained in detail below. While in beta, `SELinuxMount` is still disabled by default.

With `SELinuxMount` feature gate disabled (the default in Kubernetes 1.33 and any previous release),
the container runtime recursively assigns SELinux label to all
files on all Pod volumes by default. To speed up this process, Kubernetes can change the
SELinux label of a volume instantly by using a mount option
`-o context`.

To benefit from this speedup, all these conditions must be met

* The [feature gate](docsreferencecommand-line-tools-referencefeature-gates)
  `SELinuxMountReadWriteOncePod` must be enabled.
* Pod must use PersistentVolumeClaim with applicable `accessModes` and [feature gates](docsreferencecommand-line-tools-referencefeature-gates)
  * Either the volume has `accessModes [ReadWriteOncePod]`, and feature gate `SELinuxMountReadWriteOncePod` is enabled.
  * Or the volume can use any other access modes and all feature gates
    `SELinuxMountReadWriteOncePod`, `SELinuxChangePolicy` and `SELinuxMount` must be enabled
    and the Pod has `spec.securityContext.seLinuxChangePolicy` either nil (default) or `MountOption`.
* Pod (or all its Containers that use the PersistentVolumeClaim) must
  have `seLinuxOptions` set.
* The corresponding PersistentVolume must be either
  * A volume that uses the legacy in-tree `iscsi`, `rbd` or `fc` volume type.
  * Or a volume that uses a  driver.
    The CSI driver must announce that it supports mounting with `-o context` by setting
    `spec.seLinuxMount true` in its CSIDriver instance.

When any of these conditions is not met, SELinux relabelling happens another way the container
runtime  recursively changes the SELinux label for all inodes (files and directories)
in the volume. Calling out explicitly, this applies to Kubernetes ephemeral volumes like
`secret`, `configMap` and `projected`, and all volumes whose CSIDriver instance does not
explicitly announce mounting with `-o context`.

When this speedup is used, all Pods that use the same applicable volume concurrently on the same node
**must have the same SELinux label**. A Pod with a different SELinux label will fail to start and will be
`ContainerCreating` until all Pods with other SELinux labels that use the volume are deleted.

For Pods that want to opt-out from relabeling using mount options, they can set
`spec.securityContext.seLinuxChangePolicy` to `Recursive`. This is required
when multiple pods share a single volume on the same node, but they run with
different SELinux labels that allows simultaneous access to the volume. For example, a privileged pod
running with label `spc_t` and an unprivileged pod running with the default label `container_file_t`.
With unset `spec.securityContext.seLinuxChangePolicy` (or with the default value `MountOption`),
only one of such pods is able to run on a node, the other one gets ContainerCreating with error
`conflicting SELinux labels of volume   and `.

# # # # SELinuxWarningController
To make it easier to identify Pods that are affected by the change in SELinux volume relabeling,
a new controller called `SELinuxWarningController` has been introduced in kube-controller-manager.
It is disabled by default and can be enabled by either setting the `--controllers*,selinux-warning-controller`
[command line flag](docsreferencecommand-line-tools-referencekube-controller-manager),
or by setting `genericControllerManagerConfiguration.controllers`
[field in KubeControllerManagerConfiguration](docsreferenceconfig-apikube-controller-manager-config.v1alpha1#controllermanager-config-k8s-io-v1alpha1-GenericControllerManagerConfiguration).
This controller requires `SELinuxChangePolicy` feature gate to be enabled.

When enabled, the controller observes running Pods and when it detects that two Pods use the same volume
with different SELinux labels
1. It emits an event to both of the Pods. `kubectl describe pod ` the shows
  `SELinuxLabel  conflicts with pod  that uses the same volume as this pod
  with SELinuxLabel . If both pods land on the same node, only one of them may access the volume`.
2. Raise `selinux_warning_controller_selinux_volume_conflict` metric. The metric has both pod
  names  namespaces as labels to identify the affected pods easily.

A cluster admin can use this information to identify pods affected by the planning change and
proactively opt-out Pods from the optimization (i.e. set `spec.securityContext.seLinuxChangePolicy Recursive`).

We strongly recommend clusters that use SELinux to enable this controller and make sure that
`selinux_warning_controller_selinux_volume_conflict` metric does not report any conflicts before enabling `SELinuxMount`
feature gate or upgrading to a version where `SELinuxMount` is enabled by default.

# # # # Feature gates

The following feature gates control the behavior of SELinux volume relabeling

* `SELinuxMountReadWriteOncePod` enables the optimization for volumes with `accessModes [ReadWriteOncePod]`.
  This is a very safe feature gate to enable, as it cannot happen that two pods can share one single volume with
  this access mode. This feature gate is enabled by default sine v1.28.
* `SELinuxChangePolicy` enables `spec.securityContext.seLinuxChangePolicy` field in Pod and related SELinuxWarningController
  in kube-controller-manager. This feature can be used before enabling `SELinuxMount` to check Pods running on a cluster,
  and to pro-actively opt-out Pods from the optimization.
  This feature gate requires `SELinuxMountReadWriteOncePod` enabled. It is beta and enabled by default in 1.33.
* `SELinuxMount` enables the optimization for all eligible volumes. Since it can break existing workloads, we recommend
  enabling `SELinuxChangePolicy` feature gate  SELinuxWarningController first to check the impact of the change.
  This feature gate requires `SELinuxMountReadWriteOncePod` and `SELinuxChangePolicy` enabled. It is beta, but disabled
  by default in 1.33.

# # Managing access to the `proc` filesystem #proc-access

For runtimes that follow the OCI runtime specification, containers default to running in a mode where
there are multiple paths that are both masked and read-only.
The result of this is the container has these paths present inside the containers mount namespace, and they can function similarly to if
the container was an isolated host, but the container process cannot write to
them. The list of masked and read-only paths are as follows

- Masked Paths
  - `procasound`
  - `procacpi`
  - `prockcore`
  - `prockeys`
  - `proclatency_stats`
  - `proctimer_list`
  - `proctimer_stats`
  - `procsched_debug`
  - `procscsi`
  - `sysfirmware`
  - `sysdevicesvirtualpowercap`

- Read-Only Paths
  - `procbus`
  - `procfs`
  - `procirq`
  - `procsys`
  - `procsysrq-trigger`

For some Pods, you might want to bypass that default masking of paths.
The most common context for wanting this is if you are trying to run containers within
a Kubernetes container (within a pod).

The `securityContext` field `procMount` allows a user to request a containers `proc`
be `Unmasked`, or be mounted as read-write by the container process. This also
applies to `sysfirmware` which is not in `proc`.

```yaml
...
securityContext
  procMount Unmasked
```

Setting `procMount` to Unmasked requires the `spec.hostUsers` value in the pod
spec to be `false`. In other words a container that wishes to have an Unmasked
`proc` or unmasked `sys` must also be in a
[user namespace](docsconceptsworkloadspodsuser-namespaces).
Kubernetes v1.12 to v1.29 did not enforce that requirement.

# # Discussion

The security context for a Pod applies to the Pods Containers and also to
the Pods Volumes when applicable. Specifically `fsGroup` and `seLinuxOptions` are
applied to Volumes as follows

* `fsGroup` Volumes that support ownership management are modified to be owned
  and writable by the GID specified in `fsGroup`. See the
  [Ownership Management design document](httpsgit.k8s.iodesign-proposals-archivestoragevolume-ownership-management.md)
  for more details.

* `seLinuxOptions` Volumes that support SELinux labeling are relabeled to be accessible
  by the label specified under `seLinuxOptions`. Usually you only
  need to set the `level` section. This sets the
  [Multi-Category Security (MCS)](httpsselinuxproject.orgpageNB_MLS)
  label given to all Containers in the Pod as well as the Volumes.

After you specify an MCS label for a Pod, all Pods with the same label can access the Volume.
If you need inter-Pod protection, you must assign a unique MCS label to each Pod.

# # Clean up

Delete the Pod

```shell
kubectl delete pod security-context-demo
kubectl delete pod security-context-demo-2
kubectl delete pod security-context-demo-3
kubectl delete pod security-context-demo-4
```

# #  heading whatsnext

* [PodSecurityContext](docsreferencegeneratedkubernetes-api#podsecuritycontext-v1-core)
* [SecurityContext](docsreferencegeneratedkubernetes-api#securitycontext-v1-core)
* [CRI Plugin Config Guide](httpsgithub.comcontainerdcontainerdblobmaindocscriconfig.md)
* [Security Contexts design document](httpsgit.k8s.iodesign-proposals-archiveauthsecurity_context.md)
* [Ownership Management design document](httpsgit.k8s.iodesign-proposals-archivestoragevolume-ownership-management.md)
* [PodSecurity Admission](docsconceptssecuritypod-security-admission)
* [AllowPrivilegeEscalation design
  document](httpsgit.k8s.iodesign-proposals-archiveauthno-new-privs.md)
* For more information about security mechanisms in Linux, see
  [Overview of Linux Kernel Security Features](httpswww.linux.comlearnoverview-linux-kernel-security-features)
  (Note Some information is out of date)
* Read about [User Namespaces](docsconceptsworkloadspodsuser-namespaces)
  for Linux pods.
* [Masked Paths in the OCI Runtime
  Specification](httpsgithub.comopencontainersruntime-specblobf66aad47309config-linux.md#masked-paths)
