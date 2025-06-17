---
reviewers
- danwent
- aanm
title Use Cilium for NetworkPolicy
content_type task
weight 30
---

This page shows how to use Cilium for NetworkPolicy.

For background on Cilium, read the [Introduction to Cilium](httpsdocs.cilium.ioenstableoverviewintro).

# #  heading prerequisites

# # Deploying Cilium on Minikube for Basic Testing

To get familiar with Cilium easily you can follow the
[Cilium Kubernetes Getting Started Guide](httpsdocs.cilium.ioenstablegettingstartedk8s-install-default)
to perform a basic DaemonSet installation of Cilium in minikube.

To start minikube, minimal version required is  v1.5.2, run the with the
following arguments

```shell
minikube version
```
```
minikube version v1.5.2
```

```shell
minikube start --network-plugincni
```

For minikube you can install Cilium using its CLI tool. To do so, first download the latest
version of the CLI with the following command

```shell
curl -LO httpsgithub.comciliumcilium-clireleaseslatestdownloadcilium-linux-amd64.tar.gz
```

Then extract the downloaded file to your `usrlocalbin` directory with the following command

```shell
sudo tar xzvfC cilium-linux-amd64.tar.gz usrlocalbin
rm cilium-linux-amd64.tar.gz
```

After running the above commands, you can now install Cilium with the following command

```shell
cilium install
```

Cilium will then automatically detect the cluster configuration and create and
install the appropriate components for a successful installation.
The components are

- Certificate Authority (CA) in Secret `cilium-ca` and certificates for Hubble (Ciliums observability layer).
- Service accounts.
- Cluster roles.
- ConfigMap.
- Agent DaemonSet and an Operator Deployment.

After the installation, you can view the overall status of the Cilium deployment with the `cilium status` command.
See the expected output of the `status` command
[here](httpsdocs.cilium.ioenstablegettingstartedk8s-install-default#validate-the-installation).

The remainder of the Getting Started Guide explains how to enforce both L3L4
(i.e., IP address  port) security policies, as well as L7 (e.g., HTTP) security
policies using an example application.

# # Deploying Cilium for Production Use

For detailed instructions around deploying Cilium for production, see
[Cilium Kubernetes Installation Guide](httpsdocs.cilium.ioenstablenetworkkubernetesconcepts)
This documentation includes detailed requirements, instructions and example
production DaemonSet files.

# #  Understanding Cilium components

Deploying a cluster with Cilium adds Pods to the `kube-system` namespace. To see
this list of Pods run

```shell
kubectl get pods --namespacekube-system -l k8s-appcilium
```

Youll see a list of Pods similar to this

```console
NAME           READY   STATUS    RESTARTS   AGE
cilium-kkdhz   11     Running   0          3m23s
...
```

A `cilium` Pod runs on each node in your cluster and enforces network policy
on the traffic tofrom Pods on that node using Linux BPF.

# #  heading whatsnext

Once your cluster is running, you can follow the
[Declare Network Policy](docstasksadminister-clusterdeclare-network-policy)
to try out Kubernetes NetworkPolicy with Cilium.
Have fun, and if you have questions, contact us using the
[Cilium Slack Channel](httpscilium.herokuapp.com).
