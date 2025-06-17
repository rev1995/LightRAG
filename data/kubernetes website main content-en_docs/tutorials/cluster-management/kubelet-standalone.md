---
title Running Kubelet in Standalone Mode
content_type tutorial
weight 10
---

This tutorial shows you how to run a standalone kubelet instance.

You may have different motivations for running a standalone kubelet.
This tutorial is aimed at introducing you to Kubernetes, even if you dont have
much experience with it. You can follow this tutorial and learn about node setup,
basic (static) Pods, and how Kubernetes manages containers.

Once you have followed this tutorial, you could try using a cluster that has a
 to manage pods
and nodes, and other types of objects. For example,
[Hello, minikube](docstutorialshello-minikube).

You can also run the kubelet in standalone mode to suit production use cases, such as
to run the control plane for a highly available, resiliently deployed cluster. This
tutorial does not cover the details you need for running a resilient control plane.

# #  heading objectives

* Install `cri-o`, and `kubelet` on a Linux system and run them as `systemd` services.
* Launch a Pod running `nginx` that listens to requests on TCP port 80 on the Pods IP address.
* Learn how the different components of the solution interact among themselves.

The kubelet configuration used for this tutorial is insecure by design and should
_not_ be used in a production environment.

# #  heading prerequisites

* Admin (`root`) access to a Linux system that uses `systemd` and `iptables`
  (or nftables with `iptables` emulation).
* Access to the Internet to download the components needed for the tutorial, such as
  * A
    that implements the Kubernetes .
  * Network plugins (these are often known as
    )
  * Required CLI tools `curl`, `tar`, `jq`.

# # Prepare the system

# # # Swap configuration

By default, kubelet fails to start if swap memory is detected on a node.
This means that swap should either be disabled or tolerated by kubelet.

If you configure the kubelet to tolerate swap, the kubelet still configures Pods (and the
containers in those Pods) not to use swap space. To find out how Pods can actually
use the available swap, you can read more about
[swap memory management](docsconceptsarchitecturenodes#swap-memory) on Linux nodes.

If you have swap memory enabled, either disable it or add `failSwapOn false` to the
kubelet configuration file.

To check if swap is enabled

```shell
sudo swapon --show
```

If there is no output from the command, then swap memory is already disabled.

To disable swap temporarily

```shell
sudo swapoff -a
```

To make this change persistent across reboots

Make sure swap is disabled in either `etcfstab` or `systemd.swap`, depending on how it was
configured on your system.

# # # Enable IPv4 packet forwarding

To check if IPv4 packet forwarding is enabled

```shell
cat procsysnetipv4ip_forward
```

If the output is `1`, it is already enabled. If the output is `0`, then follow next steps.

To enable IPv4 packet forwarding, create a configuration file that sets the
`net.ipv4.ip_forward` parameter to `1`

```shell
sudo tee etcsysctl.dk8s.conf  crio-install
```

Run the installer script

```shell
sudo bash crio-install
```

Enable and start the `crio` service

```shell
sudo systemctl daemon-reload
sudo systemctl enable --now crio.service
```

Quick test

```shell
sudo systemctl is-active crio.service
```

The output is similar to

```
active
```

Detailed service check

```shell
sudo journalctl -f -u crio.service
```

# # # Install network plugins

The `cri-o` installer installs and configures the `cni-plugins` package. You can
verify the installation running the following command

```shell
optcnibinbridge --version
```

The output is similar to

```
CNI bridge plugin v1.5.1
CNI protocol versions supported 0.1.0, 0.2.0, 0.3.0, 0.3.1, 0.4.0, 1.0.0
```

To check the default configuration

```shell
cat etccninet.d11-crio-ipv4-bridge.conflist
```

The output is similar to

```json

  cniVersion 1.0.0,
  name crio,
  plugins [

      type bridge,
      bridge cni0,
      isGateway true,
      ipMasq true,
      hairpinMode true,
      ipam
        type host-local,
        routes [
             dst 0.0.0.00
        ],
        ranges [
            [ subnet 10.85.0.016 ]
        ]

  ]

```

Make sure that the default `subnet` range (`10.85.0.016`) does not overlap with
any of your active networks. If there is an overlap, you can edit the file and change it
accordingly. Restart the service after the change.

# # # Download and set up the kubelet

Download the [latest stable release](releasesdownload) of the kubelet.

curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubelet

curl -LO httpsdl.k8s.iorelease(curl -L -s httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubelet

Configure

```shell
sudo mkdir -p etckubernetesmanifests
```

```shell
sudo tee etckuberneteskubelet.yaml
Because you are not setting up a production cluster, you are using plain HTTP
(`readOnlyPort 10255`) for unauthenticated queries to the kubelets API.

The _authentication webhook_ is disabled and _authorization mode_ is set to `AlwaysAllow`
for the purpose of this tutorial. You can learn more about
[authorization modes](docsreferenceaccess-authn-authzauthorization#authorization-modules)
and [webhook authentication](docsreferenceaccess-authn-authzwebhook) to properly
configure kubelet in standalone mode in your environment.

See [Ports and Protocols](docsreferencenetworkingports-and-protocols) to
understand which ports Kubernetes components use.

Install

```shell
chmod x kubelet
sudo cp kubelet usrbin
```

Create a `systemd` service unit file

```shell
sudo tee etcsystemdsystemkubelet.service  static-web.yaml
apiVersion v1
kind Pod
metadata
  name static-web
spec
  containers
    - name web
      image nginx
      ports
        - name web
          containerPort 80
          protocol TCP
EOF
```

Copy the `static-web.yaml` manifest file to the `etckubernetesmanifests` directory.

```shell
sudo cp static-web.yaml etckubernetesmanifests
```

# # # Find out information about the kubelet and the Pod #find-out-information

The Pod networking plugin creates a network bridge (`cni0`) and a pair of `veth` interfaces
for each Pod (one of the pair is inside the newly made Pod, and the other is at the host level).

Query the kubelets API endpoint at `httplocalhost10255pods`

```shell
curl httplocalhost10255pods  jq .
```

To obtain the IP address of the `static-web` Pod

```shell
curl httplocalhost10255pods  jq .items[].status.podIP
```

The output is similar to

```
10.85.0.4
```

Connect to the `nginx` server Pod on `http` (port 80 is the default), in this case

```shell
curl http10.85.0.4
```

The output is similar to

```html

Welcome to nginx!
...
```

# # Where to look for more details

If you need to diagnose a problem getting this tutorial to work, you can look
within the following directories for monitoring and troubleshooting

```
varlibcni
varlibcontainers
varlibkubelet

varlogcontainers
varlogpods
```

# # Clean up

# # # kubelet

```shell
sudo systemctl disable --now kubelet.service
sudo systemctl daemon-reload
sudo rm etcsystemdsystemkubelet.service
sudo rm usrbinkubelet
sudo rm -rf etckubernetes
sudo rm -rf varlibkubelet
sudo rm -rf varlogcontainers
sudo rm -rf varlogpods
```

# # # Container Runtime

```shell
sudo systemctl disable --now crio.service
sudo systemctl daemon-reload
sudo rm -rf usrlocalbin
sudo rm -rf usrlocallib
sudo rm -rf usrlocalshare
sudo rm -rf usrlibexeccrio
sudo rm -rf etccrio
sudo rm -rf etccontainers
```

# # # Network Plugins

```shell
sudo rm -rf optcni
sudo rm -rf etccni
sudo rm -rf varlibcni
```

# # Conclusion

This page covered the basic aspects of deploying a kubelet in standalone mode.
You are now ready to deploy Pods and test additional functionality.

Notice that in standalone mode the kubelet does *not* support fetching Pod
configurations from the control plane (because there is no control plane connection).

You also cannot use a  or a
 to configure the containers
in a static Pod.

# #  heading whatsnext

* Follow [Hello, minikube](docstutorialshello-minikube) to learn about running Kubernetes
  _with_ a control plane. The minikube tool helps you set up a practice cluster on your own computer.
* Learn more about [Network Plugins](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins)
* Learn more about [Container Runtimes](docssetupproduction-environmentcontainer-runtimes)
* Learn more about [kubelet](docsreferencecommand-line-tools-referencekubelet)
* Learn more about [static Pods](docstasksconfigure-pod-containerstatic-pod)
