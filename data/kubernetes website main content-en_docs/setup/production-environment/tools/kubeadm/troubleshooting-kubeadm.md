---
title Troubleshooting kubeadm
content_type concept
weight 20
---

As with any program, you might run into an error installing or running kubeadm.
This page lists some common failure scenarios and have provided steps that can help you understand and fix the problem.

If your problem is not listed below, please follow the following steps

- If you think your problem is a bug with kubeadm
  - Go to [github.comkuberneteskubeadm](httpsgithub.comkuberneteskubeadmissues) and search for existing issues.
  - If no issue exists, please [open one](httpsgithub.comkuberneteskubeadmissuesnew) and follow the issue template.

- If you are unsure about how kubeadm works, you can ask on [Slack](httpsslack.k8s.io) in `#kubeadm`,
  or open a question on [StackOverflow](httpsstackoverflow.comquestionstaggedkubernetes). Please include
  relevant tags like `#kubernetes` and `#kubeadm` so folks can help you.

# # Not possible to join a v1.18 Node to a v1.17 cluster due to missing RBAC

In v1.18 kubeadm added prevention for joining a Node in the cluster if a Node with the same name already exists.
This required adding RBAC for the bootstrap-token user to be able to GET a Node object.

However this causes an issue where `kubeadm join` from v1.18 cannot join a cluster created by kubeadm v1.17.

To workaround the issue you have two options

Execute `kubeadm init phase bootstrap-token` on a control-plane node using kubeadm v1.18.
Note that this enables the rest of the bootstrap-token permissions as well.

or

Apply the following RBAC manually using `kubectl apply -f ...`

```yaml
apiVersion rbac.authorization.k8s.iov1
kind ClusterRole
metadata
  name kubeadmget-nodes
rules
  - apiGroups
      -
    resources
      - nodes
    verbs
      - get
---
apiVersion rbac.authorization.k8s.iov1
kind ClusterRoleBinding
metadata
  name kubeadmget-nodes
roleRef
  apiGroup rbac.authorization.k8s.io
  kind ClusterRole
  name kubeadmget-nodes
subjects
  - apiGroup rbac.authorization.k8s.io
    kind Group
    name systembootstrapperskubeadmdefault-node-token
```

# # `ebtables` or some similar executable not found during installation

If you see the following warnings while running `kubeadm init`

```console
[preflight] WARNING ebtables not found in system path
[preflight] WARNING ethtool not found in system path
```

Then you may be missing `ebtables`, `ethtool` or a similar executable on your node.
You can install them with the following commands

- For UbuntuDebian users, run `apt install ebtables ethtool`.
- For CentOSFedora users, run `yum install ebtables ethtool`.

# # kubeadm blocks waiting for control plane during installation

If you notice that `kubeadm init` hangs after printing out the following line

```console
[apiclient] Created API client, waiting for the control plane to become ready
```

This may be caused by a number of problems. The most common are

- network connection problems. Check that your machine has full network connectivity before continuing.
- the cgroup driver of the container runtime differs from that of the kubelet. To understand how to
  configure it properly, see [Configuring a cgroup driver](docstasksadminister-clusterkubeadmconfigure-cgroup-driver).
- control plane containers are crashlooping or hanging. You can check this by running `docker ps`
  and investigating each container by running `docker logs`. For other container runtime, see
  [Debugging Kubernetes nodes with crictl](docstasksdebugdebug-clustercrictl).

# # kubeadm blocks when removing managed containers

The following could happen if the container runtime halts and does not remove
any Kubernetes-managed containers

```shell
sudo kubeadm reset
```

```console
[preflight] Running pre-flight checks
[reset] Stopping the kubelet service
[reset] Unmounting mounted directories in varlibkubelet
[reset] Removing kubernetes-managed containers
(block)
```

A possible solution is to restart the container runtime and then re-run `kubeadm reset`.
You can also use `crictl` to debug the state of the container runtime. See
[Debugging Kubernetes nodes with crictl](docstasksdebugdebug-clustercrictl).

# # Pods in `RunContainerError`, `CrashLoopBackOff` or `Error` state

Right after `kubeadm init` there should not be any pods in these states.

- If there are pods in one of these states _right after_ `kubeadm init`, please open an
  issue in the kubeadm repo. `coredns` (or `kube-dns`) should be in the `Pending` state
  until you have deployed the network add-on.
- If you see Pods in the `RunContainerError`, `CrashLoopBackOff` or `Error` state
  after deploying the network add-on and nothing happens to `coredns` (or `kube-dns`),
  its very likely that the Pod Network add-on that you installed is somehow broken.
  You might have to grant it more RBAC privileges or use a newer version. Please file
  an issue in the Pod Network providers issue tracker and get the issue triaged there.

# # `coredns` is stuck in the `Pending` state

This is **expected** and part of the design. kubeadm is network provider-agnostic, so the admin
should [install the pod network add-on](docsconceptscluster-administrationaddons)
of choice. You have to install a Pod Network
before CoreDNS may be deployed fully. Hence the `Pending` state before the network is set up.

# # `HostPort` services do not work

The `HostPort` and `HostIP` functionality is available depending on your Pod Network
provider. Please contact the author of the Pod Network add-on to find out whether
`HostPort` and `HostIP` functionality are available.

Calico, Canal, and Flannel CNI providers are verified to support HostPort.

For more information, see the
[CNI portmap documentation](httpsgithub.comcontainernetworkingpluginsblobmasterpluginsmetaportmapREADME.md).

If your network provider does not support the portmap CNI plugin, you may need to use the
[NodePort feature of services](docsconceptsservices-networkingservice#type-nodeport)
or use `HostNetworktrue`.

# # Pods are not accessible via their Service IP

- Many network add-ons do not yet enable [hairpin mode](docstasksdebugdebug-applicationdebug-service#a-pod-fails-to-reach-itself-via-the-service-ip)
  which allows pods to access themselves via their Service IP. This is an issue related to
  [CNI](httpsgithub.comcontainernetworkingcniissues476). Please contact the network
  add-on provider to get the latest status of their support for hairpin mode.

- If you are using VirtualBox (directly or via Vagrant), you will need to
  ensure that `hostname -i` returns a routable IP address. By default, the first
  interface is connected to a non-routable host-only network. A work around
  is to modify `etchosts`, see this
  [Vagrantfile](httpsgithub.comerrordeveloperk8s-playgroundblob22dd39dfc06111235620e6c4404a96ae146f26fdVagrantfile#L11)
  for an example.

# # TLS certificate errors

The following error indicates a possible certificate mismatch.

```none
# kubectl get pods
Unable to connect to the server x509 certificate signed by unknown authority (possibly because of cryptorsa verification error while trying to verify candidate authority certificate kubernetes)
```

- Verify that the `HOME.kubeconfig` file contains a valid certificate, and
  regenerate a certificate if necessary. The certificates in a kubeconfig file
  are base64 encoded. The `base64 --decode` command can be used to decode the certificate
  and `openssl x509 -text -noout` can be used for viewing the certificate information.

- Unset the `KUBECONFIG` environment variable using

  ```sh
  unset KUBECONFIG
  ```

  Or set it to the default `KUBECONFIG` location

  ```sh
  export KUBECONFIGetckubernetesadmin.conf
  ```

- Another workaround is to overwrite the existing `kubeconfig` for the admin user

  ```sh
  mv HOME.kube HOME.kube.bak
  mkdir HOME.kube
  sudo cp -i etckubernetesadmin.conf HOME.kubeconfig
  sudo chown (id -u)(id -g) HOME.kubeconfig
  ```

# # Kubelet client certificate rotation fails #kubelet-client-cert

By default, kubeadm configures a kubelet with automatic rotation of client certificates by using the
`varlibkubeletpkikubelet-client-current.pem` symlink specified in `etckuberneteskubelet.conf`.
If this rotation process fails you might see errors such as `x509 certificate has expired or is not yet valid`
in kube-apiserver logs. To fix the issue you must follow these steps

1. Backup and delete `etckuberneteskubelet.conf` and `varlibkubeletpkikubelet-client*` from the failed node.
1. From a working control plane node in the cluster that has `etckubernetespkica.key` execute
   `kubeadm kubeconfig user --org systemnodes --client-name systemnodeNODE  kubelet.conf`.
   `NODE` must be set to the name of the existing failed node in the cluster.
   Modify the resulted `kubelet.conf` manually to adjust the cluster name and server endpoint,
   or pass `kubeconfig user --config` (see [Generating kubeconfig files for additional users](docstasksadminister-clusterkubeadmkubeadm-certs#kubeconfig-additional-users)). If your cluster does not have
   the `ca.key` you must sign the embedded certificates in the `kubelet.conf` externally.
1. Copy this resulted `kubelet.conf` to `etckuberneteskubelet.conf` on the failed node.
1. Restart the kubelet (`systemctl restart kubelet`) on the failed node and wait for
   `varlibkubeletpkikubelet-client-current.pem` to be recreated.
1. Manually edit the `kubelet.conf` to point to the rotated kubelet client certificates, by replacing
   `client-certificate-data` and `client-key-data` with

   ```yaml
   client-certificate varlibkubeletpkikubelet-client-current.pem
   client-key varlibkubeletpkikubelet-client-current.pem
   ```

1. Restart the kubelet.
1. Make sure the node becomes `Ready`.

# # Default NIC When using flannel as the pod network in Vagrant

The following error might indicate that something was wrong in the pod network

```sh
Error from server (NotFound) the server could not find the requested resource
```

- If youre using flannel as the pod network inside Vagrant, then you will have to
  specify the default interface name for flannel.

  Vagrant typically assigns two interfaces to all VMs. The first, for which all hosts
  are assigned the IP address `10.0.2.15`, is for external traffic that gets NATed.

  This may lead to problems with flannel, which defaults to the first interface on a host.
  This leads to all hosts thinking they have the same public IP address. To prevent this,
  pass the `--iface eth1` flag to flannel so that the second interface is chosen.

# # Non-public IP used for containers

In some situations `kubectl logs` and `kubectl run` commands may return with the
following errors in an otherwise functional cluster

```console
Error from server Get https10.19.0.4110250containerLogsdefaultmysql-ddc65b868-glc5mmysql dial tcp 10.19.0.4110250 getsockopt no route to host
```

- This may be due to Kubernetes using an IP that can not communicate with other IPs on
  the seemingly same subnet, possibly by policy of the machine provider.
- DigitalOcean assigns a public IP to `eth0` as well as a private one to be used internally
  as anchor for their floating IP feature, yet `kubelet` will pick the latter as the nodes
  `InternalIP` instead of the public one.

  Use `ip addr show` to check for this scenario instead of `ifconfig` because `ifconfig` will
  not display the offending alias IP address. Alternatively an API endpoint specific to
  DigitalOcean allows to query for the anchor IP from the droplet

  ```sh
  curl http169.254.169.254metadatav1interfacespublic0anchor_ipv4address
  ```

  The workaround is to tell `kubelet` which IP to use using `--node-ip`.
  When using DigitalOcean, it can be the public one (assigned to `eth0`) or
  the private one (assigned to `eth1`) should you want to use the optional
  private network. The `kubeletExtraArgs` section of the kubeadm
  [`NodeRegistrationOptions` structure](docsreferenceconfig-apikubeadm-config.v1beta4#kubeadm-k8s-io-v1beta4-NodeRegistrationOptions)
  can be used for this.

  Then restart `kubelet`

  ```sh
  systemctl daemon-reload
  systemctl restart kubelet
  ```

# # `coredns` pods have `CrashLoopBackOff` or `Error` state

If you have nodes that are running SELinux with an older version of Docker, you might experience a scenario
where the `coredns` pods are not starting. To solve that, you can try one of the following options

- Upgrade to a [newer version of Docker](docssetupproduction-environmentcontainer-runtimes#docker).

- [Disable SELinux](httpsaccess.redhat.comdocumentationen-usred_hat_enterprise_linux6htmlsecurity-enhanced_linuxsect-security-enhanced_linux-enabling_and_disabling_selinux-disabling_selinux).

- Modify the `coredns` deployment to set `allowPrivilegeEscalation` to `true`

```bash
kubectl -n kube-system get deployment coredns -o yaml
  sed sallowPrivilegeEscalation falseallowPrivilegeEscalation trueg
  kubectl apply -f -
```

Another cause for CoreDNS to have `CrashLoopBackOff` is when a CoreDNS Pod deployed in Kubernetes detects a loop.
[A number of workarounds](httpsgithub.comcorednscorednstreemasterpluginloop#troubleshooting-loops-in-kubernetes-clusters)
are available to avoid Kubernetes trying to restart the CoreDNS Pod every time CoreDNS detects the loop and exits.

Disabling SELinux or setting `allowPrivilegeEscalation` to `true` can compromise
the security of your cluster.

# # etcd pods restart continually

If you encounter the following error

```
rpc error code  2 desc  oci runtime error exec failed container_linux.go247 starting container process caused process_linux.go110 decoding init error from pipe caused read parent connection reset by peer
```

This issue appears if you run CentOS 7 with Docker 1.13.1.84.
This version of Docker can prevent the kubelet from executing into the etcd container.

To work around the issue, choose one of these options

- Roll back to an earlier version of Docker, such as 1.13.1-75

  ```
  yum downgrade docker-1.13.1-75.git8633870.el7.centos.x86_64 docker-client-1.13.1-75.git8633870.el7.centos.x86_64 docker-common-1.13.1-75.git8633870.el7.centos.x86_64
  ```

- Install one of the more recent recommended versions, such as 18.06

  ```bash
  sudo yum-config-manager --add-repo httpsdownload.docker.comlinuxcentosdocker-ce.repo
  yum install docker-ce-18.06.1.ce-3.el7.x86_64
  ```

# # Not possible to pass a comma separated list of values to arguments inside a `--component-extra-args` flag

`kubeadm init` flags such as `--component-extra-args` allow you to pass custom arguments to a control-plane
component like the kube-apiserver. However, this mechanism is limited due to the underlying type used for parsing
the values (`mapStringString`).

If you decide to pass an argument that supports multiple, comma-separated values such as
`--apiserver-extra-args enable-admission-pluginsLimitRanger,NamespaceExists` this flag will fail with
`flag malformed pair, expect stringstring`. This happens because the list of arguments for
`--apiserver-extra-args` expects `keyvalue` pairs and in this case `NamespacesExists` is considered
as a key that is missing a value.

Alternatively, you can try separating the `keyvalue` pairs like so
`--apiserver-extra-args enable-admission-pluginsLimitRanger,enable-admission-pluginsNamespaceExists`
but this will result in the key `enable-admission-plugins` only having the value of `NamespaceExists`.

A known workaround is to use the kubeadm [configuration file](docsreferenceconfig-apikubeadm-config.v1beta4).

# # kube-proxy scheduled before node is initialized by cloud-controller-manager

In cloud provider scenarios, kube-proxy can end up being scheduled on new worker nodes before
the cloud-controller-manager has initialized the node addresses. This causes kube-proxy to fail
to pick up the nodes IP address properly and has knock-on effects to the proxy function managing
load balancers.

The following error can be seen in kube-proxy Pods

```
server.go610] Failed to retrieve node IP host IP unknown known addresses []
proxier.go340] invalid nodeIP, initializing kube-proxy with 127.0.0.1 as nodeIP
```

A known solution is to patch the kube-proxy DaemonSet to allow scheduling it on control-plane
nodes regardless of their conditions, keeping it off of other nodes until their initial guarding
conditions abate

```
kubectl -n kube-system patch ds kube-proxy -p
  spec
    template
      spec
        tolerations [

            key CriticalAddonsOnly,
            operator Exists
          ,

            effect NoSchedule,
            key node-role.kubernetes.iocontrol-plane

        ]

```

The tracking issue for this problem is [here](httpsgithub.comkuberneteskubeadmissues1027).

# # `usr` is mounted read-only on nodes #usr-mounted-read-only

On Linux distributions such as Fedora CoreOS or Flatcar Container Linux, the directory `usr` is mounted as a read-only filesystem.
For [flex-volume support](httpsgithub.comkubernetescommunityblobab55d85contributorsdevelsig-storageflexvolume.md),
Kubernetes components like the kubelet and kube-controller-manager use the default path of
`usrlibexeckuberneteskubelet-pluginsvolumeexec`, yet the flex-volume directory _must be writeable_
for the feature to work.

FlexVolume was deprecated in the Kubernetes v1.23 release.

To workaround this issue, you can configure the flex-volume directory using the kubeadm
[configuration file](docsreferenceconfig-apikubeadm-config.v1beta4).

On the primary control-plane Node (created using `kubeadm init`), pass the following
file using `--config`

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind InitConfiguration
nodeRegistration
  kubeletExtraArgs
  - name volume-plugin-dir
    value optlibexeckuberneteskubelet-pluginsvolumeexec
---
apiVersion kubeadm.k8s.iov1beta4
kind ClusterConfiguration
controllerManager
  extraArgs
  - name flex-volume-plugin-dir
    value optlibexeckuberneteskubelet-pluginsvolumeexec
```

On joining Nodes

```yaml
apiVersion kubeadm.k8s.iov1beta4
kind JoinConfiguration
nodeRegistration
  kubeletExtraArgs
  - name volume-plugin-dir
    value optlibexeckuberneteskubelet-pluginsvolumeexec
```

Alternatively, you can modify `etcfstab` to make the `usr` mount writeable, but please
be advised that this is modifying a design principle of the Linux distribution.

# # `kubeadm upgrade plan` prints out `context deadline exceeded` error message

This error message is shown when upgrading a Kubernetes cluster with `kubeadm` in
the case of running an external etcd. This is not a critical bug and happens because
older versions of kubeadm perform a version check on the external etcd cluster.
You can proceed with `kubeadm upgrade apply ...`.

This issue is fixed as of version 1.19.

# # `kubeadm reset` unmounts `varlibkubelet`

If `varlibkubelet` is being mounted, performing a `kubeadm reset` will effectively unmount it.

To workaround the issue, re-mount the `varlibkubelet` directory after performing the `kubeadm reset` operation.

This is a regression introduced in kubeadm 1.15. The issue is fixed in 1.20.

# # Cannot use the metrics-server securely in a kubeadm cluster

In a kubeadm cluster, the [metrics-server](httpsgithub.comkubernetes-sigsmetrics-server)
can be used insecurely by passing the `--kubelet-insecure-tls` to it. This is not recommended for production clusters.

If you want to use TLS between the metrics-server and the kubelet there is a problem,
since kubeadm deploys a self-signed serving certificate for the kubelet. This can cause the following errors
on the side of the metrics-server

```
x509 certificate signed by unknown authority
x509 certificate is valid for IP-foo not IP-bar
```

See [Enabling signed kubelet serving certificates](docstasksadminister-clusterkubeadmkubeadm-certs#kubelet-serving-certs)
to understand how to configure the kubelets in a kubeadm cluster to have properly signed serving certificates.

Also see [How to run the metrics-server securely](httpsgithub.comkubernetes-sigsmetrics-serverblobmasterFAQ.md#how-to-run-metrics-server-securely).

# # Upgrade fails due to etcd hash not changing

Only applicable to upgrading a control plane node with a kubeadm binary v1.28.3 or later,
where the node is currently managed by kubeadm versions v1.28.0, v1.28.1 or v1.28.2.

Here is the error message you may encounter

```
[upgradeetcd] Failed to upgrade etcd couldnt upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced static Pod hash for component etcd on Node kinder-upgrade-control-plane-1 did not change after 5m0s timed out waiting for the condition
[upgradeetcd] Waiting for previous etcd to become available
I0907 101009.109104    3704 etcd.go588] [etcd] attempting to see if all cluster endpoints ([https172.17.0.62379 https172.17.0.42379 https172.17.0.32379]) are available 110
[upgradeetcd] Etcd was rolled back and is now available
static Pod hash for component etcd on Node kinder-upgrade-control-plane-1 did not change after 5m0s timed out waiting for the condition
couldnt upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced
k8s.iokubernetescmdkubeadmappphasesupgrade.rollbackOldManifests
	cmdkubeadmappphasesupgradestaticpods.go525
k8s.iokubernetescmdkubeadmappphasesupgrade.upgradeComponent
	cmdkubeadmappphasesupgradestaticpods.go254
k8s.iokubernetescmdkubeadmappphasesupgrade.performEtcdStaticPodUpgrade
	cmdkubeadmappphasesupgradestaticpods.go338
...
```

The reason for this failure is that the affected versions generate an etcd manifest file with
unwanted defaults in the PodSpec. This will result in a diff from the manifest comparison,
and kubeadm will expect a change in the Pod hash, but the kubelet will never update the hash.

There are two way to workaround this issue if you see it in your cluster

- The etcd upgrade can be skipped between the affected versions and v1.28.3 (or later) by using

  ```shell
  kubeadm upgrade applynode [version] --etcd-upgradefalse
  ```

  This is not recommended in case a new etcd version was introduced by a later v1.28 patch version.

- Before upgrade, patch the manifest for the etcd static pod, to remove the problematic defaulted attributes

  ```patch
  diff --git aetckubernetesmanifestsetcd_defaults.yaml betckubernetesmanifestsetcd_origin.yaml
  index d807ccbe0aa..46b35f00e15 100644
  --- aetckubernetesmanifestsetcd_defaults.yaml
   betckubernetesmanifestsetcd_origin.yaml
   -43,7 43,6  spec
          scheme HTTP
        initialDelaySeconds 10
        periodSeconds 10
  -      successThreshold 1
        timeoutSeconds 15
      name etcd
      resources
   -59,26 58,18  spec
          scheme HTTP
        initialDelaySeconds 10
        periodSeconds 10
  -      successThreshold 1
        timeoutSeconds 15
  -    terminationMessagePath devtermination-log
  -    terminationMessagePolicy File
      volumeMounts
      - mountPath varlibetcd
        name etcd-data
      - mountPath etckubernetespkietcd
        name etcd-certs
  -  dnsPolicy ClusterFirst
  -  enableServiceLinks true
    hostNetwork true
    priority 2000001000
    priorityClassName system-node-critical
  -  restartPolicy Always
  -  schedulerName default-scheduler
    securityContext
      seccompProfile
        type RuntimeDefault
  -  terminationGracePeriodSeconds 30
    volumes
    - hostPath
        path etckubernetespkietcd
  ```

More information can be found in the
[tracking issue](httpsgithub.comkuberneteskubeadmissues2927) for this bug.
