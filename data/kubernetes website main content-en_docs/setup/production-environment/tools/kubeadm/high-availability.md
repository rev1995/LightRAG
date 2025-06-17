---
reviewers
- sig-cluster-lifecycle
title Creating Highly Available Clusters with kubeadm
content_type task
weight 60
---

This page explains two different approaches to setting up a highly available Kubernetes
cluster using kubeadm

- With stacked control plane nodes. This approach requires less infrastructure. The etcd members
  and control plane nodes are co-located.
- With an external etcd cluster. This approach requires more infrastructure. The
  control plane nodes and etcd members are separated.

Before proceeding, you should carefully consider which approach best meets the needs of your applications
and environment. [Options for Highly Available topology](docssetupproduction-environmenttoolskubeadmha-topology)
outlines the advantages and disadvantages of each.

If you encounter issues with setting up the HA cluster, please report these
in the kubeadm [issue tracker](httpsgithub.comkuberneteskubeadmissuesnew).

See also the [upgrade documentation](docstasksadminister-clusterkubeadmkubeadm-upgrade).

This page does not address running your cluster on a cloud provider. In a cloud
environment, neither approach documented here works with Service objects of type
LoadBalancer, or with dynamic PersistentVolumes.

# #  heading prerequisites

The prerequisites depend on which topology you have selected for your clusters
control plane

 tab nameStacked etcd

You need

- Three or more machines that meet [kubeadms minimum requirements](docssetupproduction-environmenttoolskubeadminstall-kubeadm#before-you-begin) for
  the control-plane nodes. Having an odd number of control plane nodes can help
  with leader selection in the case of machine or zone failure.
  - including a , already set up and working
- Three or more machines that meet [kubeadms minimum
  requirements](docssetupproduction-environmenttoolskubeadminstall-kubeadm#before-you-begin) for the workers
  - including a container runtime, already set up and working
- Full network connectivity between all machines in the cluster (public or
  private network)
- Superuser privileges on all machines using `sudo`
  - You can use a different tool this guide uses `sudo` in the examples.
- SSH access from one device to all nodes in the system
- `kubeadm` and `kubelet` already installed on all machines.

_See [Stacked etcd topology](docssetupproduction-environmenttoolskubeadmha-topology#stacked-etcd-topology) for context._

 tab
 tab nameExternal etcd

You need

- Three or more machines that meet [kubeadms minimum requirements](docssetupproduction-environmenttoolskubeadminstall-kubeadm#before-you-begin) for
  the control-plane nodes. Having an odd number of control plane nodes can help
  with leader selection in the case of machine or zone failure.
  - including a , already set up and working
- Three or more machines that meet [kubeadms minimum
  requirements](docssetupproduction-environmenttoolskubeadminstall-kubeadm#before-you-begin) for the workers
  - including a container runtime, already set up and working
- Full network connectivity between all machines in the cluster (public or
  private network)
- Superuser privileges on all machines using `sudo`
  - You can use a different tool this guide uses `sudo` in the examples.
- SSH access from one device to all nodes in the system
- `kubeadm` and `kubelet` already installed on all machines.

And you also need

- Three or more additional machines, that will become etcd cluster members.
  Having an odd number of members in the etcd cluster is a requirement for achieving
  optimal voting quorum.
  - These machines again need to have `kubeadm` and `kubelet` installed.
  - These machines also require a container runtime, that is already set up and working.

_See [External etcd topology](docssetupproduction-environmenttoolskubeadmha-topology#external-etcd-topology) for context._
 tab

# # # Container images

Each host should have access read and fetch images from the Kubernetes container image registry,
`registry.k8s.io`. If you want to deploy a highly-available cluster where the hosts do not have
access to pull images, this is possible. You must ensure by some other means that the correct
container images are already available on the relevant hosts.

# # # Command line interface #kubectl

To manage Kubernetes once your cluster is set up, you should
[install kubectl](docstaskstools#kubectl) on your PC. It is also useful
to install the `kubectl` tool on each control plane node, as this can be
helpful for troubleshooting.

# # First steps for both methods

# # # Create load balancer for kube-apiserver

There are many configurations for load balancers. The following example is only one
option. Your cluster requirements may need a different configuration.

1. Create a kube-apiserver load balancer with a name that resolves to DNS.

   - In a cloud environment you should place your control plane nodes behind a TCP
     forwarding load balancer. This load balancer distributes traffic to all
     healthy control plane nodes in its target list. The health check for
     an apiserver is a TCP check on the port the kube-apiserver listens on
     (default value `6443`).

   - It is not recommended to use an IP address directly in a cloud environment.

   - The load balancer must be able to communicate with all control plane nodes
     on the apiserver port. It must also allow incoming traffic on its
     listening port.

   - Make sure the address of the load balancer always matches
     the address of kubeadms `ControlPlaneEndpoint`.

   - Read the [Options for Software Load Balancing](httpsgit.k8s.iokubeadmdocsha-considerations.md#options-for-software-load-balancing)
     guide for more details.

1. Add the first control plane node to the load balancer, and test the
   connection

   ```shell
   nc -zv -w 2
   ```

   A connection refused error is expected because the API server is not yet
   running. A timeout, however, means the load balancer cannot communicate
   with the control plane node. If a timeout occurs, reconfigure the load
   balancer to communicate with the control plane node.

1. Add the remaining control plane nodes to the load balancer target group.

# # Stacked control plane and etcd nodes

# # # Steps for the first control plane node

1. Initialize the control plane

   ```sh
   sudo kubeadm init --control-plane-endpoint LOAD_BALANCER_DNSLOAD_BALANCER_PORT --upload-certs
   ```

   - You can use the `--kubernetes-version` flag to set the Kubernetes version to use.
     It is recommended that the versions of kubeadm, kubelet, kubectl and Kubernetes match.
   - The `--control-plane-endpoint` flag should be set to the address or DNS and port of the load balancer.

   - The `--upload-certs` flag is used to upload the certificates that should be shared
     across all the control-plane instances to the cluster. If instead, you prefer to copy certs across
     control-plane nodes manually or using automation tools, please remove this flag and refer to [Manual
     certificate distribution](#manual-certs) section below.

   The `kubeadm init` flags `--config` and `--certificate-key` cannot be mixed, therefore if you want
   to use the [kubeadm configuration](docsreferenceconfig-apikubeadm-config.v1beta4)
   you must add the `certificateKey` field in the appropriate config locations
   (under `InitConfiguration` and `JoinConfiguration controlPlane`).

   Some CNI network plugins require additional configuration, for example specifying the pod IP CIDR, while others do not.
   See the [CNI network documentation](docssetupproduction-environmenttoolskubeadmcreate-cluster-kubeadm#pod-network).
   To add a pod CIDR pass the flag `--pod-network-cidr`, or if you are using a kubeadm configuration file
   set the `podSubnet` field under the `networking` object of `ClusterConfiguration`.

   The output looks similar to

   ```sh
   ...
   You can now join any number of control-plane node by running the following command on each as a root
       kubeadm join 192.168.0.2006443 --token 9vr73a.a8uxyaju799qwdjv --discovery-token-ca-cert-hash sha2567c2e69131a36ae2a042a339b33381c6d0d43887e2de83720eff5359e26aec866 --control-plane --certificate-key f8902e114ef118304e561c3ecd4d0b543adc226b7a07f675f56564185ffe0c07

   Please note that the certificate-key gives access to cluster sensitive data, keep it secret!
   As a safeguard, uploaded-certs will be deleted in two hours If necessary, you can use kubeadm init phase upload-certs to reload certs afterward.

   Then you can join any number of worker nodes by running the following on each as root
       kubeadm join 192.168.0.2006443 --token 9vr73a.a8uxyaju799qwdjv --discovery-token-ca-cert-hash sha2567c2e69131a36ae2a042a339b33381c6d0d43887e2de83720eff5359e26aec866
   ```

   - Copy this output to a text file. You will need it later to join control plane and worker nodes to
     the cluster.
   - When `--upload-certs` is used with `kubeadm init`, the certificates of the primary control plane
     are encrypted and uploaded in the `kubeadm-certs` Secret.
   - To re-upload the certificates and generate a new decryption key, use the following command on a
     control plane
     node that is already joined to the cluster

     ```sh
     sudo kubeadm init phase upload-certs --upload-certs
     ```

   - You can also specify a custom `--certificate-key` during `init` that can later be used by `join`.
     To generate such a key you can use the following command

     ```sh
     kubeadm certs certificate-key
     ```

   The certificate key is a hex encoded string that is an AES key of size 32 bytes.

   The `kubeadm-certs` Secret and the decryption key expire after two hours.

   As stated in the command output, the certificate key gives access to cluster sensitive data, keep it secret!

1. Apply the CNI plugin of your choice
   [Follow these instructions](docssetupproduction-environmenttoolskubeadmcreate-cluster-kubeadm#pod-network)
   to install the CNI provider. Make sure the configuration corresponds to the Pod CIDR specified in the
   kubeadm configuration file (if applicable).

   You must pick a network plugin that suits your use case and deploy it before you move on to next step.
   If you dont do this, you will not be able to launch your cluster properly.

1. Type the following and watch the pods of the control plane components get started

   ```sh
   kubectl get pod -n kube-system -w
   ```

# # # Steps for the rest of the control plane nodes

For each additional control plane node you should

1. Execute the join command that was previously given to you by the `kubeadm init` output on the first node.
   It should look something like this

   ```sh
   sudo kubeadm join 192.168.0.2006443 --token 9vr73a.a8uxyaju799qwdjv --discovery-token-ca-cert-hash sha2567c2e69131a36ae2a042a339b33381c6d0d43887e2de83720eff5359e26aec866 --control-plane --certificate-key f8902e114ef118304e561c3ecd4d0b543adc226b7a07f675f56564185ffe0c07
   ```

   - The `--control-plane` flag tells `kubeadm join` to create a new control plane.
   - The `--certificate-key ...` will cause the control plane certificates to be downloaded
     from the `kubeadm-certs` Secret in the cluster and be decrypted using the given key.

You can join multiple control-plane nodes in parallel.

# # External etcd nodes

Setting up a cluster with external etcd nodes is similar to the procedure used for stacked etcd
with the exception that you should setup etcd first, and you should pass the etcd information
in the kubeadm config file.

# # # Set up the etcd cluster

1. Follow these [instructions](docssetupproduction-environmenttoolskubeadmsetup-ha-etcd-with-kubeadm) to set up the etcd cluster.

1. Set up SSH as described [here](#manual-certs).

1. Copy the following files from any etcd node in the cluster to the first control plane node

   ```sh
   export CONTROL_PLANEubuntu10.0.0.7
   scp etckubernetespkietcdca.crt CONTROL_PLANE
   scp etckubernetespkiapiserver-etcd-client.crt CONTROL_PLANE
   scp etckubernetespkiapiserver-etcd-client.key CONTROL_PLANE
   ```

   - Replace the value of `CONTROL_PLANE` with the `userhost` of the first control-plane node.

# # # Set up the first control plane node

1. Create a file called `kubeadm-config.yaml` with the following contents

   ```yaml
   ---
   apiVersion kubeadm.k8s.iov1beta4
   kind ClusterConfiguration
   kubernetesVersion stable
   controlPlaneEndpoint LOAD_BALANCER_DNSLOAD_BALANCER_PORT # change this (see below)
   etcd
     external
       endpoints
         - httpsETCD_0_IP2379 # change ETCD_0_IP appropriately
         - httpsETCD_1_IP2379 # change ETCD_1_IP appropriately
         - httpsETCD_2_IP2379 # change ETCD_2_IP appropriately
       caFile etckubernetespkietcdca.crt
       certFile etckubernetespkiapiserver-etcd-client.crt
       keyFile etckubernetespkiapiserver-etcd-client.key
   ```

   The difference between stacked etcd and external etcd here is that the external etcd setup requires
   a configuration file with the etcd endpoints under the `external` object for `etcd`.
   In the case of the stacked etcd topology, this is managed automatically.

   - Replace the following variables in the config template with the appropriate values for your cluster

     - `LOAD_BALANCER_DNS`
     - `LOAD_BALANCER_PORT`
     - `ETCD_0_IP`
     - `ETCD_1_IP`
     - `ETCD_2_IP`

The following steps are similar to the stacked etcd setup

1. Run `sudo kubeadm init --config kubeadm-config.yaml --upload-certs` on this node.

1. Write the output join commands that are returned to a text file for later use.

1. Apply the CNI plugin of your choice.

   You must pick a network plugin that suits your use case and deploy it before you move on to next step.
   If you dont do this, you will not be able to launch your cluster properly.

# # # Steps for the rest of the control plane nodes

The steps are the same as for the stacked etcd setup

- Make sure the first control plane node is fully initialized.
- Join each control plane node with the join command you saved to a text file. Its recommended
  to join the control plane nodes one at a time.
- Dont forget that the decryption key from `--certificate-key` expires after two hours, by default.

# # Common tasks after bootstrapping control plane

# # # Install workers

Worker nodes can be joined to the cluster with the command you stored previously
as the output from the `kubeadm init` command

```sh
sudo kubeadm join 192.168.0.2006443 --token 9vr73a.a8uxyaju799qwdjv --discovery-token-ca-cert-hash sha2567c2e69131a36ae2a042a339b33381c6d0d43887e2de83720eff5359e26aec866
```

# # Manual certificate distribution #manual-certs

If you choose to not use `kubeadm init` with the `--upload-certs` flag this means that
you are going to have to manually copy the certificates from the primary control plane node to the
joining control plane nodes.

There are many ways to do this. The following example uses `ssh` and `scp`

SSH is required if you want to control all nodes from a single machine.

1. Enable ssh-agent on your main device that has access to all other nodes in
   the system

   ```shell
   eval (ssh-agent)
   ```

1. Add your SSH identity to the session

   ```shell
   ssh-add .sshpath_to_private_key
   ```

1. SSH between nodes to check that the connection is working correctly.

   - When you SSH to any node, add the `-A` flag. This flag allows the node that you
     have logged into via SSH to access the SSH agent on your PC. Consider alternative
     methods if you do not fully trust the security of your user session on the node.

     ```shell
     ssh -A 10.0.0.7
     ```

   - When using sudo on any node, make sure to preserve the environment so SSH
     forwarding works

     ```shell
     sudo -E -s
     ```

1. After configuring SSH on all the nodes you should run the following script on the first
   control plane node after running `kubeadm init`. This script will copy the certificates from
   the first control plane node to the other control plane nodes

   In the following example, replace `CONTROL_PLANE_IPS` with the IP addresses of the
   other control plane nodes.

   ```sh
   USERubuntu # customizable
   CONTROL_PLANE_IPS10.0.0.7 10.0.0.8
   for host in CONTROL_PLANE_IPS do
       scp etckubernetespkica.crt USERhost
       scp etckubernetespkica.key USERhost
       scp etckubernetespkisa.key USERhost
       scp etckubernetespkisa.pub USERhost
       scp etckubernetespkifront-proxy-ca.crt USERhost
       scp etckubernetespkifront-proxy-ca.key USERhost
       scp etckubernetespkietcdca.crt USERhostetcd-ca.crt
       # Skip the next line if you are using external etcd
       scp etckubernetespkietcdca.key USERhostetcd-ca.key
   done
   ```

   Copy only the certificates in the above list. kubeadm will take care of generating the rest of the certificates
   with the required SANs for the joining control-plane instances. If you copy all the certificates by mistake,
   the creation of additional nodes could fail due to a lack of required SANs.

1. Then on each joining control plane node you have to run the following script before running `kubeadm join`.
   This script will move the previously copied certificates from the home directory to `etckubernetespki`

   ```sh
   USERubuntu # customizable
   mkdir -p etckubernetespkietcd
   mv homeUSERca.crt etckubernetespki
   mv homeUSERca.key etckubernetespki
   mv homeUSERsa.pub etckubernetespki
   mv homeUSERsa.key etckubernetespki
   mv homeUSERfront-proxy-ca.crt etckubernetespki
   mv homeUSERfront-proxy-ca.key etckubernetespki
   mv homeUSERetcd-ca.crt etckubernetespkietcdca.crt
   # Skip the next line if you are using external etcd
   mv homeUSERetcd-ca.key etckubernetespkietcdca.key
   ```
