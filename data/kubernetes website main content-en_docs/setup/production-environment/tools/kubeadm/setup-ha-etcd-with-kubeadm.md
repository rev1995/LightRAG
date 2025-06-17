---
reviewers
- sig-cluster-lifecycle
title Set up a High Availability etcd Cluster with kubeadm
content_type task
weight 70
---

By default, kubeadm runs a local etcd instance on each control plane node.
It is also possible to treat the etcd cluster as external and provision
etcd instances on separate hosts. The differences between the two approaches are covered in the
[Options for Highly Available topology](docssetupproduction-environmenttoolskubeadmha-topology) page.

This task walks through the process of creating a high availability external
etcd cluster of three members that can be used by kubeadm during cluster creation.

# #  heading prerequisites

- Three hosts that can talk to each other over TCP ports 2379 and 2380. This
  document assumes these default ports. However, they are configurable through
  the kubeadm config file.
- Each host must have systemd and a bash compatible shell installed.
- Each host must [have a container runtime, kubelet, and kubeadm installed](docssetupproduction-environmenttoolskubeadminstall-kubeadm).
- Each host should have access to the Kubernetes container image registry (`registry.k8s.io`) or listpull the required etcd image using
  `kubeadm config images listpull`. This guide will set up etcd instances as
  [static pods](docstasksconfigure-pod-containerstatic-pod) managed by a kubelet.
- Some infrastructure to copy files between hosts. For example `ssh` and `scp`
  can satisfy this requirement.

# # Setting up the cluster

The general approach is to generate all certs on one node and only distribute
the _necessary_ files to the other nodes.

kubeadm contains all the necessary cryptographic machinery to generate
the certificates described below no other cryptographic tooling is required for
this example.

The examples below use IPv4 addresses but you can also configure kubeadm, the kubelet and etcd
to use IPv6 addresses. Dual-stack is supported by some Kubernetes options, but not by etcd. For more details
on Kubernetes dual-stack support see [Dual-stack support with kubeadm](docssetupproduction-environmenttoolskubeadmdual-stack-support).

1. Configure the kubelet to be a service manager for etcd.

   You must do this on every host where etcd should be running.
   Since etcd was created first, you must override the service priority by creating a new unit file
   that has higher precedence than the kubeadm-provided kubelet unit file.

   ```sh
   cat  etcsystemdsystemkubelet.service.dkubelet.conf
   # Replace systemd with the cgroup driver of your container runtime. The default value in the kubelet is cgroupfs.
   # Replace the value of containerRuntimeEndpoint for a different container runtime if needed.
   #
   apiVersion kubelet.config.k8s.iov1beta1
   kind KubeletConfiguration
   authentication
     anonymous
       enabled false
     webhook
       enabled false
   authorization
     mode AlwaysAllow
   cgroupDriver systemd
   address 127.0.0.1
   containerRuntimeEndpoint unixvarruncontainerdcontainerd.sock
   staticPodPath etckubernetesmanifests
   EOF

   cat  etcsystemdsystemkubelet.service.d20-etcd-service-manager.conf
   [Service]
   ExecStart
   ExecStartusrbinkubelet --configetcsystemdsystemkubelet.service.dkubelet.conf
   Restartalways
   EOF

   systemctl daemon-reload
   systemctl restart kubelet
   ```

   Check the kubelet status to ensure it is running.

   ```sh
   systemctl status kubelet
   ```

1. Create configuration files for kubeadm.

   Generate one kubeadm configuration file for each host that will have an etcd
   member running on it using the following script.

   ```sh
   # Update HOST0, HOST1 and HOST2 with the IPs of your hosts
   export HOST010.0.0.6
   export HOST110.0.0.7
   export HOST210.0.0.8

   # Update NAME0, NAME1 and NAME2 with the hostnames of your hosts
   export NAME0infra0
   export NAME1infra1
   export NAME2infra2

   # Create temp directories to store files that will end up on other hosts
   mkdir -p tmpHOST0 tmpHOST1 tmpHOST2

   HOSTS(HOST0 HOST1 HOST2)
   NAMES(NAME0 NAME1 NAME2)

   for i in !HOSTS[] do
   HOSTHOSTS[i]
   NAMENAMES[i]
   cat  tmpHOSTkubeadmcfg.yaml
   ---
   apiVersion kubeadm.k8s.iov1beta4
   kind InitConfiguration
   nodeRegistration
       name NAME
   localAPIEndpoint
       advertiseAddress HOST
   ---
   apiVersion kubeadm.k8s.iov1beta4
   kind ClusterConfiguration
   etcd
       local
           serverCertSANs
           - HOST
           peerCertSANs
           - HOST
           extraArgs
           - name initial-cluster
             value NAMES[0]httpsHOSTS[0]2380,NAMES[1]httpsHOSTS[1]2380,NAMES[2]httpsHOSTS[2]2380
           - name initial-cluster-state
             value new
           - name name
             value NAME
           - name listen-peer-urls
             value httpsHOST2380
           - name listen-client-urls
             value httpsHOST2379
           - name advertise-client-urls
             value httpsHOST2379
           - name initial-advertise-peer-urls
             value httpsHOST2380
   EOF
   done
   ```

1. Generate the certificate authority.

   If you already have a CA then the only action that is copying the CAs `crt` and
   `key` file to `etckubernetespkietcdca.crt` and
   `etckubernetespkietcdca.key`. After those files have been copied,
   proceed to the next step, Create certificates for each member.

   If you do not already have a CA then run this command on `HOST0` (where you
   generated the configuration files for kubeadm).

   ```
   kubeadm init phase certs etcd-ca
   ```

   This creates two files

   - `etckubernetespkietcdca.crt`
   - `etckubernetespkietcdca.key`

1. Create certificates for each member.

   ```sh
   kubeadm init phase certs etcd-server --configtmpHOST2kubeadmcfg.yaml
   kubeadm init phase certs etcd-peer --configtmpHOST2kubeadmcfg.yaml
   kubeadm init phase certs etcd-healthcheck-client --configtmpHOST2kubeadmcfg.yaml
   kubeadm init phase certs apiserver-etcd-client --configtmpHOST2kubeadmcfg.yaml
   cp -R etckubernetespki tmpHOST2
   # cleanup non-reusable certificates
   find etckubernetespki -not -name ca.crt -not -name ca.key -type f -delete

   kubeadm init phase certs etcd-server --configtmpHOST1kubeadmcfg.yaml
   kubeadm init phase certs etcd-peer --configtmpHOST1kubeadmcfg.yaml
   kubeadm init phase certs etcd-healthcheck-client --configtmpHOST1kubeadmcfg.yaml
   kubeadm init phase certs apiserver-etcd-client --configtmpHOST1kubeadmcfg.yaml
   cp -R etckubernetespki tmpHOST1
   find etckubernetespki -not -name ca.crt -not -name ca.key -type f -delete

   kubeadm init phase certs etcd-server --configtmpHOST0kubeadmcfg.yaml
   kubeadm init phase certs etcd-peer --configtmpHOST0kubeadmcfg.yaml
   kubeadm init phase certs etcd-healthcheck-client --configtmpHOST0kubeadmcfg.yaml
   kubeadm init phase certs apiserver-etcd-client --configtmpHOST0kubeadmcfg.yaml
   # No need to move the certs because they are for HOST0

   # clean up certs that should not be copied off this host
   find tmpHOST2 -name ca.key -type f -delete
   find tmpHOST1 -name ca.key -type f -delete
   ```

1. Copy certificates and kubeadm configs.

   The certificates have been generated and now they must be moved to their
   respective hosts.

   ```sh
   USERubuntu
   HOSTHOST1
   scp -r tmpHOST* USERHOST
   ssh USERHOST
   USERHOST  sudo -Es
   rootHOST  chown -R rootroot pki
   rootHOST  mv pki etckubernetes
   ```

1. Ensure all expected files exist.

   The complete list of required files on `HOST0` is

   ```
   tmpHOST0
    kubeadmcfg.yaml
   ---
   etckubernetespki
    apiserver-etcd-client.crt
    apiserver-etcd-client.key
    etcd
        ca.crt
        ca.key
        healthcheck-client.crt
        healthcheck-client.key
        peer.crt
        peer.key
        server.crt
        server.key
   ```

   On `HOST1`

   ```
   HOME
    kubeadmcfg.yaml
   ---
   etckubernetespki
    apiserver-etcd-client.crt
    apiserver-etcd-client.key
    etcd
        ca.crt
        healthcheck-client.crt
        healthcheck-client.key
        peer.crt
        peer.key
        server.crt
        server.key
   ```

   On `HOST2`

   ```
   HOME
    kubeadmcfg.yaml
   ---
   etckubernetespki
    apiserver-etcd-client.crt
    apiserver-etcd-client.key
    etcd
        ca.crt
        healthcheck-client.crt
        healthcheck-client.key
        peer.crt
        peer.key
        server.crt
        server.key
   ```

1. Create the static pod manifests.

   Now that the certificates and configs are in place its time to create the
   manifests. On each host run the `kubeadm` command to generate a static manifest
   for etcd.

   ```sh
   rootHOST0  kubeadm init phase etcd local --configtmpHOST0kubeadmcfg.yaml
   rootHOST1  kubeadm init phase etcd local --configHOMEkubeadmcfg.yaml
   rootHOST2  kubeadm init phase etcd local --configHOMEkubeadmcfg.yaml
   ```

1. Optional Check the cluster health.

    If `etcdctl` isnt available, you can run this tool inside a container image.
    You would do that directly with your container runtime using a tool such as
    `crictl run` and not through Kubernetes

    ```sh
    ETCDCTL_API3 etcdctl
    --cert etckubernetespkietcdpeer.crt
    --key etckubernetespkietcdpeer.key
    --cacert etckubernetespkietcdca.crt
    --endpoints httpsHOST02379 endpoint health
    ...
    https[HOST0 IP]2379 is healthy successfully committed proposal took  16.283339ms
    https[HOST1 IP]2379 is healthy successfully committed proposal took  19.44402ms
    https[HOST2 IP]2379 is healthy successfully committed proposal took  35.926451ms
    ```

    - Set `HOST0`to the IP address of the host you are testing.

# #  heading whatsnext

Once you have an etcd cluster with 3 working members, you can continue setting up a
highly available control plane using the
[external etcd method with kubeadm](docssetupproduction-environmenttoolskubeadmhigh-availability).
