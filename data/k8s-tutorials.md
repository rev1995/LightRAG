Directory structure  tutorials  _index.md  hello-minikube.md
 cluster-management   _index.md   kubelet-standalone.md 
 namespaces-walkthrough.md  configuration   _index.md  
configure-redis-using-configmap.md   pod-sidecar-containers.md  
updating-configuration-via-a-configmap.md  kubernetes-basics  
_index.md   create-cluster    _index.md   
cluster-interactive-gone.html    cluster-intro.md   deploy-app
   _index.md    deploy-interactive-gone.html   
deploy-intro.md   explore    _index.md   
explore-interactive-gone.html    explore-intro.md   expose  
 _index.md    expose-interactive-gone.html   
expose-intro.md   scale    _index.md   
scale-interactive-gone.html    scale-intro.md   update  
_index.md   update-interactive-gone.html   update-intro.md 
security   _index.md   apparmor.md   cluster-level-pss.md
  ns-level-pss.md   seccomp.md  services   _index.md 
 connect-applications-service.md  
pods-and-endpoint-termination-flow.md   source-ip.md 
stateful-application   _index.md   basic-stateful-set.md  
cassandra.md   mysql-wordpress-persistent-volume.md  
zookeeper.md  stateless-application  _index.md 
expose-external-ip-address.md  guestbook.md

 FILE datakubernetes
website main content-en_docstutorials_index.md
 --- title Tutorials
main_menu true no_list true weight 60 content_type concept ---

This section of the Kubernetes documentation contains tutorials. A
tutorial shows how to accomplish a goal that is larger than a single
[task](docstasks). Typically a tutorial has several sections, each of
which has a sequence of steps. Before walking through each tutorial, you
may want to bookmark the [Standardized
Glossary](docsreferenceglossary) page for later references.

# # Basics

* [Kubernetes Basics](docstutorialskubernetes-basics) is an in-depth
interactive tutorial that helps you understand the Kubernetes system and
try out some basic Kubernetes features. * [Introduction to Kubernetes
(edX)](httpswww.edx.orgcourseintroduction-kubernetes-linuxfoundationx-lfs158x#)
* [Hello Minikube](docstutorialshello-minikube)

# # Configuration

* [Configuring Redis Using a
ConfigMap](docstutorialsconfigurationconfigure-redis-using-configmap)

# # Authoring Pods

* [Adopting Sidecar
Containers](docstutorialsconfigurationpod-sidecar-containers)

# # Stateless Applications

* [Exposing an External IP Address to Access an Application in a
Cluster](docstutorialsstateless-applicationexpose-external-ip-address)
* [Example Deploying PHP Guestbook application with
Redis](docstutorialsstateless-applicationguestbook)

# # Stateful Applications

* [StatefulSet
Basics](docstutorialsstateful-applicationbasic-stateful-set) *
[Example WordPress and MySQL with Persistent
Volumes](docstutorialsstateful-applicationmysql-wordpress-persistent-volume)
* [Example Deploying Cassandra with Stateful
Sets](docstutorialsstateful-applicationcassandra) * [Running
ZooKeeper, A CP Distributed
System](docstutorialsstateful-applicationzookeeper)

# # Services

* [Connecting Applications with
Services](docstutorialsservicesconnect-applications-service) * [Using
Source IP](docstutorialsservicessource-ip)

# # Security

* [Apply Pod Security Standards at Cluster
level](docstutorialssecuritycluster-level-pss) * [Apply Pod Security
Standards at Namespace level](docstutorialssecurityns-level-pss) *
[Restrict a Containers Access to Resources with
AppArmor](docstutorialssecurityapparmor) *
[Seccomp](docstutorialssecurityseccomp)

# # Cluster Management

* [Running Kubelet in Standalone
Mode](docstutorialscluster-managementkubelet-standalone)

# # heading whatsnext

If you would like to write a tutorial, see [Content Page
Types](docscontributestylepage-content-types) for information about the
tutorial page type.

 FILE datakubernetes
website main content-en_docstutorialshello-minikube.md
 --- title Hello
Minikube content_type tutorial weight 5 card name tutorials weight 10
---

This tutorial shows you how to run a sample app on Kubernetes using
minikube. The tutorial provides a container image that uses NGINX to
echo back all the requests.

# # heading objectives

* Deploy a sample application to minikube. * Run the app. * View
application logs.

# # heading prerequisites

This tutorial assumes that you have already set up `minikube`. See
__Step 1__ in [minikube start](httpsminikube.sigs.k8s.iodocsstart)
for installation instructions.

Only execute the instructions in __Step 1, Installation__. The rest
is covered on this page.

You also need to install `kubectl`. See [Install
tools](docstaskstools#kubectl) for installation instructions.

# # Create a minikube cluster

```shell minikube start ```

# # Open the Dashboard

Open the Kubernetes dashboard. You can do this two different ways

tab nameLaunch a browser Open a **new** terminal, and run
```shell # Start a new terminal, and leave this running. minikube
dashboard ```

Now, switch back to the terminal where you ran `minikube start`.

The `dashboard` command enables the dashboard add-on and opens the
proxy in the default web browser. You can create Kubernetes resources on
the dashboard such as Deployment and Service.

To find out how to avoid directly invoking the browser from the terminal
and get a URL for the web dashboard, see the URL copy and paste tab.

By default, the dashboard is only accessible from within the internal
Kubernetes virtual network. The `dashboard` command creates a
temporary proxy to make the dashboard accessible from outside the
Kubernetes virtual network.

To stop the proxy, run `CtrlC` to exit the process. After the command
exits, the dashboard remains running in the Kubernetes cluster. You can
run the `dashboard` command again to create another proxy to access
the dashboard.

tab tab nameURL copy and paste

If you dont want minikube to open a web browser for you, run the
`dashboard` subcommand with the `--url` flag. `minikube` outputs
a URL that you can open in the browser you prefer.

Open a **new** terminal, and run ```shell # Start a new
terminal, and leave this running. minikube dashboard --url ```

Now, you can use this URL and switch back to the terminal where you ran
`minikube start`.

tab

# # Create a Deployment

A Kubernetes [*Pod*](docsconceptsworkloadspods) is a group of one or
more Containers, tied together for the purposes of administration and
networking. The Pod in this tutorial has only one Container. A
Kubernetes
[*Deployment*](docsconceptsworkloadscontrollersdeployment) checks on
the health of your Pod and restarts the Pods Container if it terminates.
Deployments are the recommended way to manage the creation and scaling
of Pods.

1. Use the `kubectl create` command to create a Deployment that
manages a Pod. The Pod runs a Container based on the provided Docker
image.

```shell # Run a test container image that includes a webserver
kubectl create deployment hello-node
--imageregistry.k8s.ioe2e-test-imagesagnhost2.39 -- agnhost netexec
--http-port8080 ```

1. View the Deployment

```shell kubectl get deployments ```

The output is similar to

``` NAME READY UP-TO-DATE AVAILABLE AGE hello-node 11 1 1 1m ```

(It may take some time for the pod to become available. If you see 01,
try again in a few seconds.)

1. View the Pod

```shell kubectl get pods ```

The output is similar to

``` NAME READY STATUS RESTARTS AGE hello-node-5f76cf6ccf-br9b5 11
Running 0 1m ```

1. View cluster events

```shell kubectl get events ```

1. View the `kubectl` configuration

```shell kubectl config view ```

1. View application logs for a container in a pod (replace pod name
with the one you got from `kubectl get pods`).

Replace `hello-node-5f76cf6ccf-br9b5` in the `kubectl logs` command
with the name of the pod from the `kubectl get pods` command output.

```shell kubectl logs hello-node-5f76cf6ccf-br9b5 ```

The output is similar to

``` I0911 091926.677397 1 log.go195] Started HTTP server on port
8080 I0911 091926.677586 1 log.go195] Started UDP server on port 8081
```

For more information about `kubectl` commands, see the [kubectl
overview](docsreferencekubectl).

# # Create a Service

By default, the Pod is only accessible by its internal IP address within
the Kubernetes cluster. To make the `hello-node` Container accessible
from outside the Kubernetes virtual network, you have to expose the Pod
as a Kubernetes [*Service*](docsconceptsservices-networkingservice).

The agnhost container has a `shell` endpoint, which is useful for
debugging, but dangerous to expose to the public internet. Do not run
this on an internet-facing cluster, or a production cluster.

1. Expose the Pod to the public internet using the `kubectl expose`
command

```shell kubectl expose deployment hello-node --typeLoadBalancer
--port8080 ```

The `--typeLoadBalancer` flag indicates that you want to expose your
Service outside of the cluster.

The application code inside the test image only listens on TCP port
8080. If you used `kubectl expose` to expose a different port, clients
could not connect to that other port.

2. View the Service you created

```shell kubectl get services ```

The output is similar to

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE hello-node
LoadBalancer 10.108.144.78 808030369TCP 21s kubernetes ClusterIP
10.96.0.1 443TCP 23m ```

On cloud providers that support load balancers, an external IP address
would be provisioned to access the Service. On minikube, the
`LoadBalancer` type makes the Service accessible through the
`minikube service` command.

3. Run the following command

```shell minikube service hello-node ```

This opens up a browser window that serves your app and shows the apps
response.

# # Enable addons

The minikube tool includes a set of built-in that can be enabled,
disabled and opened in the local Kubernetes environment.

1. List the currently supported addons

```shell minikube addons list ```

The output is similar to

``` addon-manager enabled dashboard enabled default-storageclass
enabled efk disabled freshpod disabled gvisor disabled helm-tiller
disabled ingress disabled ingress-dns disabled logviewer disabled
metrics-server disabled nvidia-driver-installer disabled
nvidia-gpu-device-plugin disabled registry disabled registry-creds
disabled storage-provisioner enabled storage-provisioner-gluster
disabled ```

1. Enable an addon, for example, `metrics-server`

```shell minikube addons enable metrics-server ```

The output is similar to

``` The metrics-server addon is enabled ```

1. View the Pod and Service you created by installing that addon

```shell kubectl get pod,svc -n kube-system ```

The output is similar to

``` NAME READY STATUS RESTARTS AGE podcoredns-5644d7b6d9-mh9ll 11
Running 0 34m podcoredns-5644d7b6d9-pqd2t 11 Running 0 34m
podmetrics-server-67fb648c5 11 Running 0 26s podetcd-minikube 11 Running
0 34m podinfluxdb-grafana-b29w8 22 Running 0 26s
podkube-addon-manager-minikube 11 Running 0 34m
podkube-apiserver-minikube 11 Running 0 34m
podkube-controller-manager-minikube 11 Running 0 34m podkube-proxy-rnlps
11 Running 0 34m podkube-scheduler-minikube 11 Running 0 34m
podstorage-provisioner 11 Running 0 34m

NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE servicemetrics-server
ClusterIP 10.96.241.45 80TCP 26s servicekube-dns ClusterIP 10.96.0.10
53UDP,53TCP 34m servicemonitoring-grafana NodePort 10.99.24.54
8030002TCP 26s servicemonitoring-influxdb ClusterIP 10.111.169.94
8083TCP,8086TCP 26s ```

1. Check the output from `metrics-server`

```shell kubectl top pods ```

The output is similar to

``` NAME CPU(cores) MEMORY(bytes) hello-node-ccf4b9788-4jn97 1m 6Mi
```

If you see the following message, wait, and try again

``` error Metrics API not available ```

1. Disable `metrics-server`

```shell minikube addons disable metrics-server ```

The output is similar to

``` metrics-server was successfully disabled ```

# # Clean up

Now you can clean up the resources you created in your cluster

```shell kubectl delete service hello-node kubectl delete deployment
hello-node ```

Stop the Minikube cluster

```shell minikube stop ```

Optionally, delete the Minikube VM

```shell # Optional minikube delete ```

If you want to use minikube again to learn more about Kubernetes, you
dont need to delete it.

# # Conclusion

This page covered the basic aspects to get a minikube cluster up and
running. You are now ready to deploy applications.

# # heading whatsnext

* Tutorial to _[deploy your first app on Kubernetes with
kubectl](docstutorialskubernetes-basicsdeploy-appdeploy-intro)_. *
Learn more about [Deployment
objects](docsconceptsworkloadscontrollersdeployment). * Learn more
about [Deploying
applications](docstasksrun-applicationrun-stateless-application-deployment).
* Learn more about [Service
objects](docsconceptsservices-networkingservice).

 FILE datakubernetes
website main content-en_docstutorialscluster-management_index.md
 --- title Cluster
Management weight 60 ---

 FILE datakubernetes
website main
content-en_docstutorialscluster-managementkubelet-standalone.md
 --- title Running
Kubelet in Standalone Mode content_type tutorial weight 10 ---

This tutorial shows you how to run a standalone kubelet instance.

You may have different motivations for running a standalone kubelet.
This tutorial is aimed at introducing you to Kubernetes, even if you
dont have much experience with it. You can follow this tutorial and
learn about node setup, basic (static) Pods, and how Kubernetes manages
containers.

Once you have followed this tutorial, you could try using a cluster that
has a to manage pods and nodes, and other types of objects. For example,
[Hello, minikube](docstutorialshello-minikube).

You can also run the kubelet in standalone mode to suit production use
cases, such as to run the control plane for a highly available,
resiliently deployed cluster. This tutorial does not cover the details
you need for running a resilient control plane.

# # heading objectives

* Install `cri-o`, and `kubelet` on a Linux system and run them as
`systemd` services. * Launch a Pod running `nginx` that listens to
requests on TCP port 80 on the Pods IP address. * Learn how the
different components of the solution interact among themselves.

The kubelet configuration used for this tutorial is insecure by design
and should _not_ be used in a production environment.

# # heading prerequisites

* Admin (`root`) access to a Linux system that uses `systemd` and
`iptables` (or nftables with `iptables` emulation). * Access to the
Internet to download the components needed for the tutorial, such as *
A that implements the Kubernetes . * Network plugins (these are often
known as ) * Required CLI tools `curl`, `tar`, `jq`.

# # Prepare the system

# # # Swap configuration

By default, kubelet fails to start if swap memory is detected on a node.
This means that swap should either be disabled or tolerated by kubelet.

If you configure the kubelet to tolerate swap, the kubelet still
configures Pods (and the containers in those Pods) not to use swap
space. To find out how Pods can actually use the available swap, you can
read more about [swap memory
management](docsconceptsarchitecturenodes#swap-memory) on Linux nodes.

If you have swap memory enabled, either disable it or add `failSwapOn
false` to the kubelet configuration file.

To check if swap is enabled

```shell sudo swapon --show ```

If there is no output from the command, then swap memory is already
disabled.

To disable swap temporarily

```shell sudo swapoff -a ```

To make this change persistent across reboots

Make sure swap is disabled in either `etcfstab` or `systemd.swap`,
depending on how it was configured on your system.

# # # Enable IPv4 packet forwarding

To check if IPv4 packet forwarding is enabled

```shell cat procsysnetipv4ip_forward ```

If the output is `1`, it is already enabled. If the output is `0`,
then follow next steps.

To enable IPv4 packet forwarding, create a configuration file that sets
the `net.ipv4.ip_forward` parameter to `1`

```shell sudo tee etcsysctl.dk8s.conf crio-install ```

Run the installer script

```shell sudo bash crio-install ```

Enable and start the `crio` service

```shell sudo systemctl daemon-reload sudo systemctl enable --now
crio.service ```

Quick test

```shell sudo systemctl is-active crio.service ```

The output is similar to

``` active ```

Detailed service check

```shell sudo journalctl -f -u crio.service ```

# # # Install network plugins

The `cri-o` installer installs and configures the `cni-plugins`
package. You can verify the installation running the following command

```shell optcnibinbridge --version ```

The output is similar to

``` CNI bridge plugin v1.5.1 CNI protocol versions supported 0.1.0,
0.2.0, 0.3.0, 0.3.1, 0.4.0, 1.0.0 ```

To check the default configuration

```shell cat etccninet.d11-crio-ipv4-bridge.conflist ```

The output is similar to

```json

cniVersion 1.0.0, name crio, plugins [

type bridge, bridge cni0, isGateway true, ipMasq true, hairpinMode true,
ipam type host-local, routes [ dst 0.0.0.00 ], ranges [ [ subnet
10.85.0.016 ] ]

]

```

Make sure that the default `subnet` range (`10.85.0.016`) does not
overlap with any of your active networks. If there is an overlap, you
can edit the file and change it accordingly. Restart the service after
the change.

# # # Download and set up the kubelet

Download the [latest stable release](releasesdownload) of the kubelet.

curl -LO httpsdl.k8s.iorelease(curl -L -s
httpsdl.k8s.ioreleasestable.txt)binlinuxamd64kubelet

curl -LO httpsdl.k8s.iorelease(curl -L -s
httpsdl.k8s.ioreleasestable.txt)binlinuxarm64kubelet

Configure

```shell sudo mkdir -p etckubernetesmanifests ```

```shell sudo tee etckuberneteskubelet.yaml Because you are not
setting up a production cluster, you are using plain HTTP
(`readOnlyPort 10255`) for unauthenticated queries to the kubelets
API.

The _authentication webhook_ is disabled and _authorization mode_ is
set to `AlwaysAllow` for the purpose of this tutorial. You can learn
more about [authorization
modes](docsreferenceaccess-authn-authzauthorization#authorization-modules)
and [webhook authentication](docsreferenceaccess-authn-authzwebhook)
to properly configure kubelet in standalone mode in your environment.

See [Ports and Protocols](docsreferencenetworkingports-and-protocols)
to understand which ports Kubernetes components use.

Install

```shell chmod x kubelet sudo cp kubelet usrbin ```

Create a `systemd` service unit file

```shell sudo tee etcsystemdsystemkubelet.service static-web.yaml
apiVersion v1 kind Pod metadata name static-web spec containers  - name
web image nginx ports  - name web containerPort 80 protocol TCP EOF
```

Copy the `static-web.yaml` manifest file to the
`etckubernetesmanifests` directory.

```shell sudo cp static-web.yaml etckubernetesmanifests ```

# # # Find out information about the kubelet and the Pod
# find-out-information

The Pod networking plugin creates a network bridge (`cni0`) and a pair
of `veth` interfaces for each Pod (one of the pair is inside the newly
made Pod, and the other is at the host level).

Query the kubelets API endpoint at `httplocalhost10255pods`

```shell curl httplocalhost10255pods jq . ```

To obtain the IP address of the `static-web` Pod

```shell curl httplocalhost10255pods jq .items[].status.podIP
```

The output is similar to

``` 10.85.0.4 ```

Connect to the `nginx` server Pod on `http` (port 80 is the
default), in this case

```shell curl http10.85.0.4 ```

The output is similar to

```html

Welcome to nginx! ... ```

# # Where to look for more details

If you need to diagnose a problem getting this tutorial to work, you can
look within the following directories for monitoring and troubleshooting

``` varlibcni varlibcontainers varlibkubelet

varlogcontainers varlogpods ```

# # Clean up

# # # kubelet

```shell sudo systemctl disable --now kubelet.service sudo systemctl
daemon-reload sudo rm etcsystemdsystemkubelet.service sudo rm
usrbinkubelet sudo rm -rf etckubernetes sudo rm -rf varlibkubelet sudo
rm -rf varlogcontainers sudo rm -rf varlogpods ```

# # # Container Runtime

```shell sudo systemctl disable --now crio.service sudo systemctl
daemon-reload sudo rm -rf usrlocalbin sudo rm -rf usrlocallib sudo rm
-rf usrlocalshare sudo rm -rf usrlibexeccrio sudo rm -rf etccrio sudo rm
-rf etccontainers ```

# # # Network Plugins

```shell sudo rm -rf optcni sudo rm -rf etccni sudo rm -rf varlibcni
```

# # Conclusion

This page covered the basic aspects of deploying a kubelet in standalone
mode. You are now ready to deploy Pods and test additional
functionality.

Notice that in standalone mode the kubelet does *not* support fetching
Pod configurations from the control plane (because there is no control
plane connection).

You also cannot use a or a to configure the containers in a static Pod.

# # heading whatsnext

* Follow [Hello, minikube](docstutorialshello-minikube) to learn
about running Kubernetes _with_ a control plane. The minikube tool
helps you set up a practice cluster on your own computer. * Learn more
about [Network
Plugins](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins)
* Learn more about [Container
Runtimes](docssetupproduction-environmentcontainer-runtimes) * Learn
more about [kubelet](docsreferencecommand-line-tools-referencekubelet)
* Learn more about [static
Pods](docstasksconfigure-pod-containerstatic-pod)

 FILE datakubernetes
website main
content-en_docstutorialscluster-managementnamespaces-walkthrough.md
 --- reviewers -
derekwaynecarr - janetkuo title Namespaces Walkthrough content_type task
weight 260 ---

Kubernetes help different projects, teams, or customers to share a
Kubernetes cluster.

It does this by providing the following

1. A scope for
[Names](docsconceptsoverviewworking-with-objectsnames). 2. A mechanism
to attach authorization and policy to a subsection of the cluster.

Use of multiple namespaces is optional.

This example demonstrates how to use Kubernetes namespaces to subdivide
your cluster.

# # heading prerequisites

# # Prerequisites

This example assumes the following

1. You have an [existing Kubernetes cluster](docssetup). 2. You have
a basic understanding of Kubernetes , , and .

# # Understand the default namespace

By default, a Kubernetes cluster will instantiate a default namespace
when provisioning the cluster to hold the default set of Pods, Services,
and Deployments used by the cluster.

Assuming you have a fresh cluster, you can inspect the available
namespaces by doing the following

```shell kubectl get namespaces ``` ``` NAME STATUS AGE default
Active 13m ```

# # Create new namespaces

For this exercise, we will create two additional Kubernetes namespaces
to hold our content.

Lets imagine a scenario where an organization is using a shared
Kubernetes cluster for development and production use cases.

The development team would like to maintain a space in the cluster where
they can get a view on the list of Pods, Services, and Deployments they
use to build and run their application. In this space, Kubernetes
resources come and go, and the restrictions on who can or cannot modify
resources are relaxed to enable agile development.

The operations team would like to maintain a space in the cluster where
they can enforce strict procedures on who can or cannot manipulate the
set of Pods, Services, and Deployments that run the production site.

One pattern this organization could follow is to partition the
Kubernetes cluster into two namespaces `development` and
`production`.

Lets create two new namespaces to hold our work.

Use the file [`namespace-dev.yaml`](examplesadminnamespace-dev.yaml)
which describes a `development` namespace

code_sample languageyaml fileadminnamespace-dev.yaml

Create the `development` namespace using kubectl.

```shell kubectl create -f httpsk8s.ioexamplesadminnamespace-dev.yaml
```

Save the following contents into file
[`namespace-prod.yaml`](examplesadminnamespace-prod.yaml) which
describes a `production` namespace

code_sample languageyaml fileadminnamespace-prod.yaml

And then lets create the `production` namespace using kubectl.

```shell kubectl create -f
httpsk8s.ioexamplesadminnamespace-prod.yaml ```

To be sure things are right, lets list all of the namespaces in our
cluster.

```shell kubectl get namespaces --show-labels ``` ``` NAME
STATUS AGE LABELS default Active 32m development Active 29s
namedevelopment production Active 23s nameproduction ```

# # Create pods in each namespace

A Kubernetes namespace provides the scope for Pods, Services, and
Deployments in the cluster.

Users interacting with one namespace do not see the content in another
namespace.

To demonstrate this, lets spin up a simple Deployment and Pods in the
`development` namespace.

We first check what is the current context

```shell kubectl config view ``` ```yaml apiVersion v1
clusters - cluster certificate-authority-data REDACTED server
https130.211.122.180 name lithe-cocoa-92103_kubernetes contexts -
context cluster lithe-cocoa-92103_kubernetes user
lithe-cocoa-92103_kubernetes name lithe-cocoa-92103_kubernetes
current-context lithe-cocoa-92103_kubernetes kind Config preferences
users - name lithe-cocoa-92103_kubernetes user client-certificate-data
REDACTED client-key-data REDACTED token
65rZW78y8HbwXXtSXuUw9DbP4FLjHi4b - name
lithe-cocoa-92103_kubernetes-basic-auth user password h5M0FtUUIflBSdI7
username admin ``` ```shell kubectl config current-context ```
``` lithe-cocoa-92103_kubernetes ```

The next step is to define a context for the kubectl client to work in
each namespace. The value of cluster and user fields are copied from the
current context.

```shell kubectl config set-context dev --namespacedevelopment
--clusterlithe-cocoa-92103_kubernetes
--userlithe-cocoa-92103_kubernetes

kubectl config set-context prod --namespaceproduction
--clusterlithe-cocoa-92103_kubernetes
--userlithe-cocoa-92103_kubernetes ```

By default, the above commands add two contexts that are saved into file
`.kubeconfig`. You can now view the contexts and alternate against the
two new request contexts depending on which namespace you wish to work
against.

To view the new contexts

```shell kubectl config view ``` ```yaml apiVersion v1
clusters - cluster certificate-authority-data REDACTED server
https130.211.122.180 name lithe-cocoa-92103_kubernetes contexts -
context cluster lithe-cocoa-92103_kubernetes user
lithe-cocoa-92103_kubernetes name lithe-cocoa-92103_kubernetes - context
cluster lithe-cocoa-92103_kubernetes namespace development user
lithe-cocoa-92103_kubernetes name dev - context cluster
lithe-cocoa-92103_kubernetes namespace production user
lithe-cocoa-92103_kubernetes name prod current-context
lithe-cocoa-92103_kubernetes kind Config preferences users - name
lithe-cocoa-92103_kubernetes user client-certificate-data REDACTED
client-key-data REDACTED token 65rZW78y8HbwXXtSXuUw9DbP4FLjHi4b - name
lithe-cocoa-92103_kubernetes-basic-auth user password h5M0FtUUIflBSdI7
username admin ```

Lets switch to operate in the `development` namespace.

```shell kubectl config use-context dev ```

You can verify your current context by doing the following

```shell kubectl config current-context ``` ``` dev ```

At this point, all requests we make to the Kubernetes cluster from the
command line are scoped to the `development` namespace.

Lets create some contents.

code_sample fileadminsnowflake-deployment.yaml

Apply the manifest to create a Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesadminsnowflake-deployment.yaml ``` We have created
a deployment whose replica size is 2 that is running the pod called
`snowflake` with a basic container that serves the hostname.

```shell kubectl get deployment ``` ``` NAME READY UP-TO-DATE
AVAILABLE AGE snowflake 22 2 2 2m ```

```shell kubectl get pods -l appsnowflake ``` ``` NAME READY
STATUS RESTARTS AGE snowflake-3968820950-9dgr8 11 Running 0 2m
snowflake-3968820950-vgc4n 11 Running 0 2m ```

And this is great, developers are able to do what they want, and they do
not have to worry about affecting content in the `production`
namespace.

Lets switch to the `production` namespace and show how resources in
one namespace are hidden from the other.

```shell kubectl config use-context prod ```

The `production` namespace should be empty, and the following commands
should return nothing.

```shell kubectl get deployment kubectl get pods ```

Production likes to run cattle, so lets create some cattle pods.

```shell kubectl create deployment cattle
--imageregistry.k8s.ioserve_hostname --replicas5

kubectl get deployment ``` ``` NAME READY UP-TO-DATE AVAILABLE AGE
cattle 55 5 5 10s ```

```shell kubectl get pods -l appcattle ``` ``` NAME READY
STATUS RESTARTS AGE cattle-2263376956-41xy6 11 Running 0 34s
cattle-2263376956-kw466 11 Running 0 34s cattle-2263376956-n4v97 11
Running 0 34s cattle-2263376956-p5p3i 11 Running 0 34s
cattle-2263376956-sxpth 11 Running 0 34s ```

At this point, it should be clear that the resources users create in one
namespace are hidden from the other namespace.

As the policy support in Kubernetes evolves, we will extend this
scenario to show how you can provide different authorization rules for
each namespace.

 FILE datakubernetes
website main content-en_docstutorialsconfiguration_index.md
 --- title
Configuration weight 30 ---

 FILE datakubernetes
website main
content-en_docstutorialsconfigurationconfigure-redis-using-configmap.md
 --- reviewers -
eparis - pmorie title Configuring Redis using a ConfigMap content_type
tutorial weight 30 ---

This page provides a real world example of how to configure Redis using
a ConfigMap and builds upon the [Configure a Pod to Use a
ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap)
task.

# # heading objectives

* Create a ConfigMap with Redis configuration values * Create a Redis
Pod that mounts and uses the created ConfigMap * Verify that the
configuration was correctly applied.

# # heading prerequisites

* The example shown on this page works with `kubectl` 1.14 and above.
* Understand [Configure a Pod to Use a
ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap).

# # Real World Example Configuring Redis using a ConfigMap

Follow the steps below to configure a Redis cache using data stored in a
ConfigMap.

First create a ConfigMap with an empty configuration block

```shell cat .example-redis-config.yaml apiVersion v1 kind ConfigMap
metadata name example-redis-config data redis-config EOF ```

Apply the ConfigMap created above, along with a Redis pod manifest

```shell kubectl apply -f example-redis-config.yaml kubectl apply -f
httpsraw.githubusercontent.comkuberneteswebsitemaincontentenexamplespodsconfigredis-pod.yaml
```

Examine the contents of the Redis pod manifest and note the following

* A volume named `config` is created by `spec.volumes[1]` * The
`key` and `path` under `spec.volumes[1].configMap.items[0]`
exposes the `redis-config` key from the `example-redis-config`
ConfigMap as a file named `redis.conf` on the `config` volume. *
The `config` volume is then mounted at `redis-master` by
`spec.containers[0].volumeMounts[1]`.

This has the net effect of exposing the data in `data.redis-config`
from the `example-redis-config` ConfigMap above as
`redis-masterredis.conf` inside the Pod.

code_sample filepodsconfigredis-pod.yaml

Examine the created objects

```shell kubectl get podredis configmapexample-redis-config ```

You should see the following output

``` NAME READY STATUS RESTARTS AGE podredis 11 Running 0 8s

NAME DATA AGE configmapexample-redis-config 1 14s ```

Recall that we left `redis-config` key in the `example-redis-config`
ConfigMap blank

```shell kubectl describe configmapexample-redis-config ```

You should see an empty `redis-config` key

```shell Name example-redis-config Namespace default Labels
Annotations

Data

redis-config ```

Use `kubectl exec` to enter the pod and run the `redis-cli` tool to
check the current configuration

```shell kubectl exec -it redis -- redis-cli ```

Check `maxmemory`

```shell 127.0.0.16379 CONFIG GET maxmemory ```

It should show the default value of 0

```shell 1) maxmemory 2) 0 ```

Similarly, check `maxmemory-policy`

```shell 127.0.0.16379 CONFIG GET maxmemory-policy ```

Which should also yield its default value of `noeviction`

```shell 1) maxmemory-policy 2) noeviction ```

Now lets add some configuration values to the `example-redis-config`
ConfigMap

code_sample filepodsconfigexample-redis-config.yaml

Apply the updated ConfigMap

```shell kubectl apply -f example-redis-config.yaml ```

Confirm that the ConfigMap was updated

```shell kubectl describe configmapexample-redis-config ```

You should see the configuration values we just added

```shell Name example-redis-config Namespace default Labels
Annotations

Data

redis-config ---- maxmemory 2mb maxmemory-policy allkeys-lru ```

Check the Redis Pod again using `redis-cli` via `kubectl exec` to
see if the configuration was applied

```shell kubectl exec -it redis -- redis-cli ```

Check `maxmemory`

```shell 127.0.0.16379 CONFIG GET maxmemory ```

It remains at the default value of 0

```shell 1) maxmemory 2) 0 ```

Similarly, `maxmemory-policy` remains at the `noeviction` default
setting

```shell 127.0.0.16379 CONFIG GET maxmemory-policy ```

Returns

```shell 1) maxmemory-policy 2) noeviction ```

The configuration values have not changed because the Pod needs to be
restarted to grab updated values from associated ConfigMaps. Lets delete
and recreate the Pod

```shell kubectl delete pod redis kubectl apply -f
httpsraw.githubusercontent.comkuberneteswebsitemaincontentenexamplespodsconfigredis-pod.yaml
```

Now re-check the configuration values one last time

```shell kubectl exec -it redis -- redis-cli ```

Check `maxmemory`

```shell 127.0.0.16379 CONFIG GET maxmemory ```

It should now return the updated value of 2097152

```shell 1) maxmemory 2) 2097152 ```

Similarly, `maxmemory-policy` has also been updated

```shell 127.0.0.16379 CONFIG GET maxmemory-policy ```

It now reflects the desired value of `allkeys-lru`

```shell 1) maxmemory-policy 2) allkeys-lru ```

Clean up your work by deleting the created resources

```shell kubectl delete podredis configmapexample-redis-config ```

# # heading whatsnext

* Learn more about
[ConfigMaps](docstasksconfigure-pod-containerconfigure-pod-configmap).
* Follow an example of [Updating configuration via a
ConfigMap](docstutorialsconfigurationupdating-configuration-via-a-configmap).

 FILE datakubernetes
website main
content-en_docstutorialsconfigurationpod-sidecar-containers.md
 --- title Adopting
Sidecar Containers content_type tutorial weight 40
min-kubernetes-server-version 1.29 ---

This section is relevant for people adopting a new built-in [sidecar
containers](docsconceptsworkloadspodssidecar-containers) feature for
their workloads.

Sidecar container is not a new concept as posted in the [blog
post](blog201506the-distributed-system-toolkit-patterns). Kubernetes
allows running multiple containers in a Pod to implement this concept.
However, running a sidecar container as a regular container has a lot of
limitations being fixed with the new built-in sidecar containers
support.

# # heading objectives

- Understand the need for sidecar containers - Be able to troubleshoot
issues with the sidecar containers - Understand options to universally
inject sidecar containers to any workload

# # heading prerequisites

# # Sidecar containers overview

Sidecar containers are secondary containers that run along with the main
application container within the same . These containers are used to
enhance or to extend the functionality of the primary _app container_
by providing additional services, or functionalities such as logging,
monitoring, security, or data synchronization, without directly altering
the primary application code. You can read more in the [Sidecar
containers](docsconceptsworkloadspodssidecar-containers) concept page.

The concept of sidecar containers is not new and there are multiple
implementations of this concept. As well as sidecar containers that you,
the person defining the Pod, want to run, you can also find that some
modify Pods - before the Pods start running - so that there are extra
sidecar containers. The mechanisms to _inject_ those extra sidecars
are often [mutating
webhooks](docsreferenceaccess-authn-authzadmission-controllers#mutatingadmissionwebhook).
For example, a service mesh addon might inject a sidecar that configures
mutual TLS and encryption in transit between different Pods.

While the concept of sidecar containers is not new, the native
implementation of this feature in Kubernetes, however, is new. And as
with every new feature, adopting this feature may present certain
challenges.

This tutorial explores challenges and solutions that can be experienced
by end users as well as by authors of sidecar containers.

# # Benefits of a built-in sidecar container

Using Kubernetes native support for sidecar containers provides several
benefits

1. You can configure a native sidecar container to start ahead of . 1.
The built-in sidecar containers can be authored to guarantee that they
are terminated last. Sidecar containers are terminated with a
`SIGTERM` signal once all the regular containers are completed and
terminated. If the sidecar container isnt gracefully shut down, a
`SIGKILL` signal will be used to terminate it. 1. With Jobs, when Pods
`restartPolicy OnFailure` or `restartPolicy Never`, native sidecar
containers do not block Pod completion. With legacy sidecar containers,
special care is needed to handle this situation. 1. Also, with Jobs,
built-in sidecar containers would keep being restarted once they are
done, even if regular containers would not with Pods `restartPolicy
Never`.

See [differences from init
containers](docsconceptsworkloadspodssidecar-containers#differences-from-application-containers)
to learn more about it.

# # Adopting built-in sidecar containers

The `SidecarContainers` [feature
gate](docsreferencecommand-line-tools-referencefeature-gates) is in
beta state starting from Kubernetes version 1.29 and is enabled by
default. Some clusters may have this feature disabled or have software
installed that is incompatible with the feature.

When this happens, the Pod may be rejected or the sidecar containers may
block Pod startup, rendering the Pod useless. This condition is easy to
detect as the Pod simply gets stuck on initialization. However, it is
often unclear what caused the problem.

Here are the considerations and troubleshooting steps that one can take
while adopting sidecar containers for their workload.

# # # Ensure the feature gate is enabled

As a very first step, make sure that both API server and Nodes are at
Kubernetes version v1.29 or later. The feature will break on clusters
where Nodes are running earlier versions where it is not enabled.

The feature can be enabled on nodes with the version 1.28. The behavior
of built-in sidecar container termination was different in version 1.28,
and it is not recommended to adjust the behavior of a sidecar to that
behavior. However, if the only concern is the startup order, the above
statement can be changed to Nodes running version 1.28 with the feature
gate enabled.

You should ensure that the feature gate is enabled for the API server(s)
within the control plane **and** for all nodes.

One of the ways to check the feature gate enablement is to run a command
like this

- For API Server

```shell kubectl get --raw metrics grep kubernetes_feature_enabled
grep SidecarContainers ```

- For the individual node

```shell kubectl get --raw apiv1nodesproxymetrics grep
kubernetes_feature_enabled grep SidecarContainers ```

If you see something like this

``` kubernetes_feature_enablednameSidecarContainers,stageBETA 1
```

it means that the feature is enabled.

# # # Check for 3rd party tooling and mutating webhooks

If you experience issues when validating the feature, it may be an
indication that one of the 3rd party tools or mutating webhooks are
broken.

When the `SidecarContainers` feature gate is enabled, Pods gain a new
field in their API. Some tools or mutating webhooks might have been
built with an earlier version of Kubernetes API.

If tools pass unknown fields as-is using various patching strategies to
mutate a Pod object, this will not be a problem. However, there are
tools that will strip out unknown fields if you have those, they must be
recompiled with the v1.28 version of Kubernetes API client code.

The way to check this is to use the `kubectl describe pod` command
with your Pod that has passed through mutating admission. If any tools
stripped out the new field (`restartPolicyAlways`), you will not see
it in the command output.

If you hit an issue like this, please advise the author of the tools or
the webhooks use one of the patching strategies for modifying objects
instead of a full object update.

Mutating webhook may update Pods based on some conditions. Thus, sidecar
containers may work for some Pods and fail for others.

# # # Automatic injection of sidecars

If you are using software that injects sidecars automatically, there are
a few possible strategies you may follow to ensure that native sidecar
containers can be used. All strategies are generally options you may
choose to decide whether the Pod the sidecar will be injected to will
land on a Node supporting the feature or not.

As an example, you can follow [this conversation in Istio
community](httpsgithub.comistioistioissues48794). The discussion
explores the options listed below.

1. Mark Pods that land to nodes supporting sidecars. You can use node
labels and node affinity to mark nodes supporting sidecar containers and
Pods landing on those nodes. 1. Check Nodes compatibility on injection.
During sidecar injection, you may use the following strategies to check
node compatibility  - query node version and assume the feature gate is
enabled on the version 1.29  - query node prometheus metrics and check
feature enablement status  - assume the nodes are running with a
[supported version
skew](releasesversion-skew-policy#supported-version-skew) from the API
server  - there may be other custom ways to detect nodes
compatibility. 1. Develop a universal sidecar injector. The idea of a
universal sidecar injector is to inject a sidecar container as a regular
container as well as a native sidecar container. And have a runtime
logic to decide which one will work. The universal sidecar injector is
wasteful, as it will account for requests twice, but may be considered
as a workable solution for special cases.  - One way would be on start
of a native sidecar container detect the node version and exit
immediately if the version does not support the sidecar feature.  -
Consider a runtime feature detection design  - Define an empty dir so
containers can communicate with each other  - Inject an init container,
lets call it `NativeSidecar` with `restartPolicyAlways`.  -
`NativeSidecar` must write a file to an empty directory indicating the
first run and exit immediately with exit code `0`.  -
`NativeSidecar` on restart (when native sidecars are supported) checks
that file already exists in the empty dir and changes it - indicating
that the built-in sidecar containers are supported and running.  -
Inject regular container, lets call it `OldWaySidecar`.  -
`OldWaySidecar` on start checks the presence of a file in an empty
dir.  - If the file indicates that the `NativeSidecar` is NOT running,
it assumes that the sidecar feature is not supported and works assuming
it is the sidecar.  - If the file indicates that the `NativeSidecar`
is running, it either does nothing and sleeps forever (in the case when
Pods `restartPolicyAlways`) or exits immediately with exit code `0`
(in the case when Pods `restartPolicy!Always`).

# # heading whatsnext

- Learn more about [sidecar
containers](docsconceptsworkloadspodssidecar-containers).

 FILE datakubernetes
website main
content-en_docstutorialsconfigurationupdating-configuration-via-a-configmap.md
 --- title Updating
Configuration via a ConfigMap content_type tutorial weight 20 ---

This page provides a step-by-step example of updating configuration
within a Pod via a ConfigMap and builds upon the [Configure a Pod to
Use a
ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap)
task. At the end of this tutorial, you will understand how to change the
configuration for a running application. This tutorial uses the
`alpine` and `nginx` images as examples.

# # heading prerequisites

You need to have the [curl](httpscurl.se) command-line tool for making
HTTP requests from the terminal or command prompt. If you do not have
`curl` available, you can install it. Check the documentation for your
local operating system.

# # heading objectives * Update configuration via a ConfigMap mounted
as a Volume * Update environment variables of a Pod via a ConfigMap *
Update configuration via a ConfigMap in a multi-container Pod * Update
configuration via a ConfigMap in a Pod possessing a Sidecar Container

# # Update configuration via a ConfigMap mounted as a Volume
# rollout-configmap-volume

Use the `kubectl create configmap` command to create a ConfigMap from
[literal
values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell kubectl create configmap sport --from-literalsportfootball
```

Below is an example of a Deployment manifest with the ConfigMap
`sport` mounted as a into the Pods only container. code_sample
filedeploymentsdeployment-with-configmap-as-volume.yaml

Create the Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-as-volume.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell kubectl get pods
--selectorapp.kubernetes.ionameconfigmap-volume ```

You should see an output similar to

``` NAME READY STATUS RESTARTS AGE configmap-volume-6b976dfdcf-qxvbm
11 Running 0 72s configmap-volume-6b976dfdcf-skpvm 11 Running 0 72s
configmap-volume-6b976dfdcf-tbc6r 11 Running 0 72s ```

On each node where one of these Pods is running, the kubelet fetches the
data for that ConfigMap and translates it to files in a local volume.
The kubelet then mounts that volume into the container, as specified in
the Pod template. The code running in that container loads the
information from the file and uses it to print a report to stdout. You
can check this report by viewing the logs for one of the Pods in that
Deployment

```shell # Pick one Pod that belongs to the Deployment, and view its
logs kubectl logs deploymentsconfigmap-volume ```

You should see an output similar to

``` Found 3 pods, using podconfigmap-volume-76d9c5678f-x5rgj Thu Jan
4 140646 UTC 2024 My preferred sport is football Thu Jan 4 140656 UTC
2024 My preferred sport is football Thu Jan 4 140706 UTC 2024 My
preferred sport is football Thu Jan 4 140716 UTC 2024 My preferred sport
is football Thu Jan 4 140726 UTC 2024 My preferred sport is football
```

Edit the ConfigMap

```shell kubectl edit configmap sport ```

In the editor that appears, change the value of key `sport` from
`football` to `cricket`. Save your changes. The kubectl tool updates
the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml apiVersion v1 data sport cricket kind ConfigMap # You can
leave the existing metadata as they are. # The values youll see wont
exactly match these. metadata creationTimestamp 2024-01-04T140506Z name
sport namespace default resourceVersion 1743935 uid
024ee001-fe72-487e-872e-34d6464a8a23 ```

You should see the following output

``` configmapsport edited ```

Tail (follow the latest entries in) the logs of one of the pods that
belongs to this Deployment

```shell kubectl logs deploymentsconfigmap-volume --follow ```

After few seconds, you should see the log output change as follows

``` Thu Jan 4 141136 UTC 2024 My preferred sport is football Thu Jan
4 141146 UTC 2024 My preferred sport is football Thu Jan 4 141156 UTC
2024 My preferred sport is football Thu Jan 4 141206 UTC 2024 My
preferred sport is cricket Thu Jan 4 141216 UTC 2024 My preferred sport
is cricket ```

When you have a ConfigMap that is mapped into a running Pod using either
a `configMap` volume or a `projected` volume, and you update that
ConfigMap, the running Pod sees the update almost immediately. However,
your application only sees the change if it is written to either poll
for changes, or watch for file updates. An application that loads its
configuration once at startup will not notice a change.

The total delay from the moment when the ConfigMap is updated to the
moment when new keys are projected to the Pod can be as long as kubelet
sync period. Also check [Mounted ConfigMaps are updated
automatically](docstasksconfigure-pod-containerconfigure-pod-configmap#mounted-configmaps-are-updated-automatically).

# # Update environment variables of a Pod via a ConfigMap
# rollout-configmap-env

Use the `kubectl create configmap` command to create a ConfigMap from
[literal
values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell kubectl create configmap fruits --from-literalfruitsapples
```

Below is an example of a Deployment manifest with an environment
variable configured via the ConfigMap `fruits`.

code_sample filedeploymentsdeployment-with-configmap-as-envvar.yaml

Create the Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-as-envvar.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell kubectl get pods
--selectorapp.kubernetes.ionameconfigmap-env-var ```

You should see an output similar to

``` NAME READY STATUS RESTARTS AGE configmap-env-var-59cfc64f7d-74d7z
11 Running 0 46s configmap-env-var-59cfc64f7d-c4wmj 11 Running 0 46s
configmap-env-var-59cfc64f7d-dpr98 11 Running 0 46s ```

The key-value pair in the ConfigMap is configured as an environment
variable in the container of the Pod. Check this by viewing the logs of
one Pod that belongs to the Deployment.

```shell kubectl logs deploymentconfigmap-env-var ```

You should see an output similar to

``` Found 3 pods, using podconfigmap-env-var-7c994f7769-l74nq Thu Jan
4 160706 UTC 2024 The basket is full of apples Thu Jan 4 160716 UTC 2024
The basket is full of apples Thu Jan 4 160726 UTC 2024 The basket is
full of apples ```

Edit the ConfigMap

```shell kubectl edit configmap fruits ```

In the editor that appears, change the value of key `fruits` from
`apples` to `mangoes`. Save your changes. The kubectl tool updates
the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml apiVersion v1 data fruits mangoes kind ConfigMap # You can
leave the existing metadata as they are. # The values youll see wont
exactly match these. metadata creationTimestamp 2024-01-04T160419Z name
fruits namespace default resourceVersion 1749472 ```

You should see the following output

``` configmapfruits edited ```

Tail the logs of the Deployment and observe the output for few seconds

```shell # As the text explains, the output does NOT change kubectl
logs deploymentsconfigmap-env-var --follow ```

Notice that the output remains **unchanged**, even though you edited
the ConfigMap

``` Thu Jan 4 161256 UTC 2024 The basket is full of apples Thu Jan 4
161306 UTC 2024 The basket is full of apples Thu Jan 4 161316 UTC 2024
The basket is full of apples Thu Jan 4 161326 UTC 2024 The basket is
full of apples ```

Although the value of the key inside the ConfigMap has changed, the
environment variable in the Pod still shows the earlier value. This is
because environment variables for a process running inside a Pod are
**not** updated when the source data changes if you wanted to force
an update, you would need to have Kubernetes replace your existing Pods.
The new Pods would then run with the updated information.

You can trigger that replacement. Perform a rollout for the Deployment,
using [`kubectl
rollout`](docsreferencekubectlgeneratedkubectl_rollout)

```shell # Trigger the rollout kubectl rollout restart deployment
configmap-env-var

# Wait for the rollout to complete kubectl rollout status deployment
configmap-env-var --watchtrue ```

Next, check the Deployment

```shell kubectl get deployment configmap-env-var ```

You should see an output similar to

``` NAME READY UP-TO-DATE AVAILABLE AGE configmap-env-var 33 3 3 12m
```

Check the Pods

```shell kubectl get pods
--selectorapp.kubernetes.ionameconfigmap-env-var ```

The rollout causes Kubernetes to make a new for the Deployment that
means the existing Pods eventually terminate, and new ones are created.
After few seconds, you should see an output similar to

``` NAME READY STATUS RESTARTS AGE configmap-env-var-6d94d89bf5-2ph2l
11 Running 0 13s configmap-env-var-6d94d89bf5-74twx 11 Running 0 8s
configmap-env-var-6d94d89bf5-d5vx8 11 Running 0 11s ```

Please wait for the older Pods to fully terminate before proceeding with
the next steps.

View the logs for a Pod in this Deployment

```shell # Pick one Pod that belongs to the Deployment, and view its
logs kubectl logs deploymentconfigmap-env-var ```

You should see an output similar to the below

``` Found 3 pods, using podconfigmap-env-var-6d9ff89fb6-bzcf6 Thu Jan
4 163035 UTC 2024 The basket is full of mangoes Thu Jan 4 163045 UTC
2024 The basket is full of mangoes Thu Jan 4 163055 UTC 2024 The basket
is full of mangoes ```

This demonstrates the scenario of updating environment variables in a
Pod that are derived from a ConfigMap. Changes to the ConfigMap values
are applied to the Pod during the subsequent rollout. If Pods get
created for another reason, such as scaling up the Deployment, then the
new Pods also use the latest configuration values if you dont trigger a
rollout, then you might find that your app is running with a mix of old
and new environment variable values.

# # Update configuration via a ConfigMap in a multi-container Pod
# rollout-configmap-multiple-containers

Use the `kubectl create configmap` command to create a ConfigMap from
[literal
values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell kubectl create configmap color --from-literalcolorred
```

Below is an example manifest for a Deployment that manages a set of
Pods, each with two containers. The two containers share an `emptyDir`
volume that they use to communicate. The first container runs a web
server (`nginx`). The mount path for the shared volume in the web
server container is `usrsharenginxhtml`. The second helper container
is based on `alpine`, and for this container the `emptyDir` volume
is mounted at `pod-data`. The helper container writes a file in HTML
that has its content based on a ConfigMap. The web server container
serves the HTML via HTTP.

code_sample filedeploymentsdeployment-with-configmap-two-containers.yaml

Create the Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-two-containers.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell kubectl get pods
--selectorapp.kubernetes.ionameconfigmap-two-containers ```

You should see an output similar to

``` NAME READY STATUS RESTARTS AGE
configmap-two-containers-565fb6d4f4-2xhxf 22 Running 0 20s
configmap-two-containers-565fb6d4f4-g5v4j 22 Running 0 20s
configmap-two-containers-565fb6d4f4-mzsmf 22 Running 0 20s ```

Expose the Deployment (the `kubectl` tool creates a for you)

```shell kubectl expose deployment configmap-two-containers
--nameconfigmap-service --port8080 --target-port80 ```

Use `kubectl` to forward the port

```shell # this stays running in the background kubectl port-forward
serviceconfigmap-service 80808080 ```

Access the service.

```shell curl httplocalhost8080 ```

You should see an output similar to

``` Fri Jan 5 080822 UTC 2024 My preferred color is red ```

Edit the ConfigMap

```shell kubectl edit configmap color ```

In the editor that appears, change the value of key `color` from
`red` to `blue`. Save your changes. The kubectl tool updates the
ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml apiVersion v1 data color blue kind ConfigMap # You can leave
the existing metadata as they are. # The values youll see wont exactly
match these. metadata creationTimestamp 2024-01-05T081205Z name color
namespace configmap resourceVersion 1801272 uid
80d33e4a-cbb4-4bc9-ba8c-544c68e425d6 ```

Loop over the service URL for few seconds.

```shell # Cancel this when youre happy with it (Ctrl-C) while true
do curl --connect-timeout 7.5 httplocalhost8080 sleep 10 done ```

You should see the output change as follows

``` Fri Jan 5 081400 UTC 2024 My preferred color is red Fri Jan 5
081402 UTC 2024 My preferred color is red Fri Jan 5 081420 UTC 2024 My
preferred color is red Fri Jan 5 081422 UTC 2024 My preferred color is
red Fri Jan 5 081432 UTC 2024 My preferred color is blue Fri Jan 5
081443 UTC 2024 My preferred color is blue Fri Jan 5 081500 UTC 2024 My
preferred color is blue ```

# # Update configuration via a ConfigMap in a Pod possessing a sidecar
container #rollout-configmap-sidecar

The above scenario can be replicated by using a [Sidecar
Container](docsconceptsworkloadspodssidecar-containers) as a helper
container to write the HTML file. As a Sidecar Container is conceptually
an Init Container, it is guaranteed to start before the main web server
container. This ensures that the HTML file is always available when the
web server is ready to serve it.

If you are continuing from the previous scenario, you can reuse the
ConfigMap named `color` for this scenario. If you are executing this
scenario independently, use the `kubectl create configmap` command to
create a ConfigMap from [literal
values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell kubectl create configmap color --from-literalcolorblue
```

Below is an example manifest for a Deployment that manages a set of
Pods, each with a main container and a sidecar container. The two
containers share an `emptyDir` volume that they use to communicate.
The main container runs a web server (NGINX). The mount path for the
shared volume in the web server container is `usrsharenginxhtml`. The
second container is a Sidecar Container based on Alpine Linux which acts
as a helper container. For this container the `emptyDir` volume is
mounted at `pod-data`. The Sidecar Container writes a file in HTML
that has its content based on a ConfigMap. The web server container
serves the HTML via HTTP.

code_sample
filedeploymentsdeployment-with-configmap-and-sidecar-container.yaml

Create the Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-and-sidecar-container.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell kubectl get pods
--selectorapp.kubernetes.ionameconfigmap-sidecar-container ```

You should see an output similar to

``` NAME READY STATUS RESTARTS AGE
configmap-sidecar-container-5fb59f558b-87rp7 22 Running 0 94s
configmap-sidecar-container-5fb59f558b-ccs7s 22 Running 0 94s
configmap-sidecar-container-5fb59f558b-wnmgk 22 Running 0 94s ```

Expose the Deployment (the `kubectl` tool creates a for you)

```shell kubectl expose deployment configmap-sidecar-container
--nameconfigmap-sidecar-service --port8081 --target-port80 ```

Use `kubectl` to forward the port

```shell # this stays running in the background kubectl port-forward
serviceconfigmap-sidecar-service 80818081 ```

Access the service.

```shell curl httplocalhost8081 ```

You should see an output similar to

``` Sat Feb 17 130905 UTC 2024 My preferred color is blue ```

Edit the ConfigMap

```shell kubectl edit configmap color ```

In the editor that appears, change the value of key `color` from
`blue` to `green`. Save your changes. The kubectl tool updates the
ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml apiVersion v1 data color green kind ConfigMap # You can
leave the existing metadata as they are. # The values youll see wont
exactly match these. metadata creationTimestamp 2024-02-17T122030Z name
color namespace default resourceVersion 1054 uid
e40bb34c-58df-4280-8bea-6ed16edccfaa ```

Loop over the service URL for few seconds.

```shell # Cancel this when youre happy with it (Ctrl-C) while true
do curl --connect-timeout 7.5 httplocalhost8081 sleep 10 done ```

You should see the output change as follows

``` Sat Feb 17 131235 UTC 2024 My preferred color is blue Sat Feb 17
131245 UTC 2024 My preferred color is blue Sat Feb 17 131255 UTC 2024 My
preferred color is blue Sat Feb 17 131305 UTC 2024 My preferred color is
blue Sat Feb 17 131315 UTC 2024 My preferred color is green Sat Feb 17
131325 UTC 2024 My preferred color is green Sat Feb 17 131335 UTC 2024
My preferred color is green ```

# # Update configuration via an immutable ConfigMap that is mounted as
a volume #rollout-configmap-immutable-volume

Immutable ConfigMaps are especially used for configuration that is
constant and is **not** expected to change over time. Marking a
ConfigMap as immutable allows a performance improvement where the
kubelet does not watch for changes.

If you do need to make a change, you should plan to either

- change the name of the ConfigMap, and switch to running Pods that
reference the new name - replace all the nodes in your cluster that have
previously run a Pod that used the old value - restart the kubelet on
any node where the kubelet previously loaded the old ConfigMap

An example manifest for an [Immutable
ConfigMap](docsconceptsconfigurationconfigmap#configmap-immutable) is
shown below. code_sample fileconfigmapimmutable-configmap.yaml

Create the Immutable ConfigMap

```shell kubectl apply -f
httpsk8s.ioexamplesconfigmapimmutable-configmap.yaml ```

Below is an example of a Deployment manifest with the Immutable
ConfigMap `company-name-20150801` mounted as a into the Pods only
container.

code_sample
filedeploymentsdeployment-with-immutable-configmap-as-volume.yaml

Create the Deployment

```shell kubectl apply -f
httpsk8s.ioexamplesdeploymentsdeployment-with-immutable-configmap-as-volume.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell kubectl get pods
--selectorapp.kubernetes.ionameimmutable-configmap-volume ```

You should see an output similar to

``` NAME READY STATUS RESTARTS AGE
immutable-configmap-volume-78b6fbff95-5gsfh 11 Running 0 62s
immutable-configmap-volume-78b6fbff95-7vcj4 11 Running 0 62s
immutable-configmap-volume-78b6fbff95-vdslm 11 Running 0 62s ```

The Pods container refers to the data defined in the ConfigMap and uses
it to print a report to stdout. You can check this report by viewing the
logs for one of the Pods in that Deployment

```shell # Pick one Pod that belongs to the Deployment, and view its
logs kubectl logs deploymentsimmutable-configmap-volume ```

You should see an output similar to

``` Found 3 pods, using
podimmutable-configmap-volume-78b6fbff95-5gsfh Wed Mar 20 035234 UTC
2024 The name of the company is ACME, Inc. Wed Mar 20 035244 UTC 2024
The name of the company is ACME, Inc. Wed Mar 20 035254 UTC 2024 The
name of the company is ACME, Inc. ```

Once a ConfigMap is marked as immutable, it is not possible to revert
this change nor to mutate the contents of the data or the binaryData
field. In order to modify the behavior of the Pods that use this
configuration, you will create a new immutable ConfigMap and edit the
Deployment to define a slightly different pod template, referencing the
new ConfigMap.

Create a new immutable ConfigMap by using the manifest shown below

code_sample fileconfigmapnew-immutable-configmap.yaml

```shell kubectl apply -f
httpsk8s.ioexamplesconfigmapnew-immutable-configmap.yaml ```

You should see an output similar to

``` configmapcompany-name-20240312 created ```

Check the newly created ConfigMap

```shell kubectl get configmap ```

You should see an output displaying both the old and new ConfigMaps

``` NAME DATA AGE company-name-20150801 1 22m company-name-20240312 1
24s ```

Modify the Deployment to reference the new ConfigMap.

Edit the Deployment

```shell kubectl edit deployment immutable-configmap-volume ```

In the editor that appears, update the existing volume definition to use
the new ConfigMap.

```yaml volumes - configMap defaultMode 420 name
company-name-20240312 # Update this field name config-volume ```

You should see the following output

``` deployment.appsimmutable-configmap-volume edited ```

This will trigger a rollout. Wait for all the previous Pods to terminate
and the new Pods to be in a ready state.

Monitor the status of the Pods

```shell kubectl get pods
--selectorapp.kubernetes.ionameimmutable-configmap-volume ```

``` NAME READY STATUS RESTARTS AGE
immutable-configmap-volume-5fdb88fcc8-29v8n 11 Running 0 13s
immutable-configmap-volume-5fdb88fcc8-52ddd 11 Running 0 14s
immutable-configmap-volume-5fdb88fcc8-n5jx4 11 Running 0 15s
immutable-configmap-volume-78b6fbff95-5gsfh 11 Terminating 0 32m
immutable-configmap-volume-78b6fbff95-7vcj4 11 Terminating 0 32m
immutable-configmap-volume-78b6fbff95-vdslm 11 Terminating 0 32m ```

You should eventually see an output similar to

``` NAME READY STATUS RESTARTS AGE
immutable-configmap-volume-5fdb88fcc8-29v8n 11 Running 0 43s
immutable-configmap-volume-5fdb88fcc8-52ddd 11 Running 0 44s
immutable-configmap-volume-5fdb88fcc8-n5jx4 11 Running 0 45s ```

View the logs for a Pod in this Deployment

```shell # Pick one Pod that belongs to the Deployment, and view its
logs kubectl logs deploymentimmutable-configmap-volume ```

You should see an output similar to the below

``` Found 3 pods, using
podimmutable-configmap-volume-5fdb88fcc8-n5jx4 Wed Mar 20 042417 UTC
2024 The name of the company is Fiktivesunternehmen GmbH Wed Mar 20
042427 UTC 2024 The name of the company is Fiktivesunternehmen GmbH Wed
Mar 20 042437 UTC 2024 The name of the company is Fiktivesunternehmen
GmbH ```

Once all the deployments have migrated to use the new immutable
ConfigMap, it is advised to delete the old one.

```shell kubectl delete configmap company-name-20150801 ```

# # Summary

Changes to a ConfigMap mounted as a Volume on a Pod are available
seamlessly after the subsequent kubelet sync.

Changes to a ConfigMap that configures environment variables for a Pod
are available after the subsequent rollout for the Pod.

Once a ConfigMap is marked as immutable, it is not possible to revert
this change (you cannot make an immutable ConfigMap mutable), and you
also cannot make any change to the contents of the `data` or the
`binaryData` field. You can delete and recreate the ConfigMap, or you
can make a new different ConfigMap. When you delete a ConfigMap, running
containers and their Pods maintain a mount point to any volume that
referenced that existing ConfigMap.

# # heading cleanup

Terminate the `kubectl port-forward` commands in case they are
running.

Delete the resources created during the tutorial

```shell kubectl delete deployment configmap-volume configmap-env-var
configmap-two-containers configmap-sidecar-container
immutable-configmap-volume kubectl delete service configmap-service
configmap-sidecar-service kubectl delete configmap sport fruits color
company-name-20240312

kubectl delete configmap company-name-20150801 # In case it was not
handled during the task execution ```

 FILE datakubernetes
website main content-en_docstutorialskubernetes-basics_index.md
 --- title Learn
Kubernetes Basics main_menu true no_list true weight 20 content_type
concept card name tutorials weight 20 title Walkthrough the basics ---

# # heading objectives

This tutorial provides a walkthrough of the basics of the Kubernetes
cluster orchestration system. Each module contains some background
information on major Kubernetes features and concepts, and a tutorial
for you to follow along.

Using the tutorials, you can learn to

* Deploy a containerized application on a cluster. * Scale the
deployment. * Update the containerized application with a new software
version. * Debug the containerized application.

# # What can Kubernetes do for you

With modern web services, users expect applications to be available 247,
and developers expect to deploy new versions of those applications
several times a day. Containerization helps package software to serve
these goals, enabling applications to be released and updated without
downtime. Kubernetes helps you make sure those containerized
applications run where and when you want, and helps them find the
resources and tools they need to work. Kubernetes is a production-ready,
open source platform designed with Googles accumulated experience in
container orchestration, combined with best-of-breed ideas from the
community.

# # Kubernetes Basics Modules

 1. Create a Kubernetes cluster

2. Deploy an app

3. Explore your app

4. Expose your app publicly

5. Scale up your app

6. Update your app

# # heading whatsnext

* Tutorial [Using Minikube to Create a
Cluster](docstutorialskubernetes-basicscreate-cluster)

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicscreate-cluster_index.md
 --- title Create a
Cluster weight 10 ---

Learn about Kubernetes and create a simple cluster using Minikube.

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicscreate-clustercluster-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Creating a
Cluster # before Katacoda shut down. # # There is no need to
localize this Not Found page the website automatically # serves a 404
response when a page is missing. If you have an existing localized #
version of this page, it is OK to remove that localized version. weight
20 headless true toc_hide true _build list never publishResources
false ---

  Content unavailable  
The interactive tutorial for creating a cluster is not available. For
more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicscreate-clustercluster-intro.md
 --- title Using
Minikube to Create a Cluster weight 10 ---

# # heading objectives

* Learn what a Kubernetes cluster is. * Learn what Minikube is. *
Start a Kubernetes cluster on your computer.

# # Kubernetes Clusters

alert _Kubernetes is a production-grade, open-source platform that
orchestrates the placement (scheduling) and execution of application
containers within and across computer clusters._ alert

**Kubernetes coordinates a highly available cluster of computers that
are connected to work as a single unit.** The abstractions in
Kubernetes allow you to deploy containerized applications to a cluster
without tying them specifically to individual machines. To make use of
this new model of deployment, applications need to be packaged in a way
that decouples them from individual hosts they need to be containerized.
Containerized applications are more flexible and available than in past
deployment models, where applications were installed directly onto
specific machines as packages deeply integrated into the host.
**Kubernetes automates the distribution and scheduling of application
containers across a cluster in a more efficient way.** Kubernetes is
an open-source platform and is production-ready.

A Kubernetes cluster consists of two types of resources

* The **Control Plane** coordinates the cluster * **Nodes**
are the workers that run applications

# # # Cluster Diagram

**The Control Plane is responsible for managing the cluster.** The
Control Plane coordinates all activities in your cluster, such as
scheduling applications, maintaining applications desired state, scaling
applications, and rolling out new updates.

alert _Control Planes manage the cluster and the nodes that are used to
host the running applications._ alert

**A node is a VM or a physical computer that serves as a worker
machine in a Kubernetes cluster.** Each node has a Kubelet, which is
an agent for managing the node and communicating with the Kubernetes
control plane. The node should also have tools for handling container
operations, such as or . A Kubernetes cluster that handles production
traffic should have a minimum of three nodes because if one node goes
down, both an [etcd](docsconceptsarchitecture#etcd) member and a
control plane instance are lost, and redundancy is compromised. You can
mitigate this risk by adding more control plane nodes.

When you deploy applications on Kubernetes, you tell the control plane
to start the application containers. The control plane schedules the
containers to run on the clusters nodes. **Node-level components, such
as the kubelet, communicate with the control plane using the
[Kubernetes API](docsconceptsoverviewkubernetes-api)**, which the
control plane exposes. End users can also use the Kubernetes API
directly to interact with the cluster.

A Kubernetes cluster can be deployed on either physical or virtual
machines. To get started with Kubernetes development, you can use
Minikube. Minikube is a lightweight Kubernetes implementation that
creates a VM on your local machine and deploys a simple cluster
containing only one node. Minikube is available for Linux, macOS, and
Windows systems. The Minikube CLI provides basic bootstrapping
operations for working with your cluster, including start, stop, status,
and delete.

# # heading whatsnext

* Tutorial [Hello Minikube](docstutorialshello-minikube). * Learn
more about [Cluster Architecture](docsconceptsarchitecture).

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsdeploy-app_index.md
 --- title Deploy an
App weight 20 ---

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsdeploy-appdeploy-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Deploying an
App # before Katacoda shut down. # # There is no need to localize
this Not Found page the website automatically # serves a 404 response
when a page is missing. If you have an existing localized # version of
this page, it is OK to remove that localized version. weight 20
headless true toc_hide true _build list never publishResources
false --- 

  Content unavailable  
The interactive tutorial for deploying an application to your cluster is
not available. For more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsdeploy-appdeploy-intro.md
 --- title Using
kubectl to Create a Deployment weight 10 ---

# # heading objectives

* Learn about application Deployments. * Deploy your first app on
Kubernetes with kubectl.

# # Kubernetes Deployments

alert _A Deployment is responsible for creating and updating instances
of your application._ alert

This tutorial uses a container that requires the AMD64 architecture. If
you are using minikube on a computer with a different CPU architecture,
you could try using minikube with a driver that can emulate AMD64. For
example, the Docker Desktop driver can do this.

Once you have a [running Kubernetes
cluster](docstutorialskubernetes-basicscreate-clustercluster-intro),
you can deploy your containerized applications on top of it. To do so,
you create a Kubernetes **Deployment**. The Deployment instructs
Kubernetes how to create and update instances of your application. Once
youve created a Deployment, the Kubernetes control plane schedules the
application instances included in that Deployment to run on individual
Nodes in the cluster.

Once the application instances are created, a Kubernetes Deployment
controller continuously monitors those instances. If the Node hosting an
instance goes down or is deleted, the Deployment controller replaces the
instance with an instance on another Node in the cluster. **This
provides a self-healing mechanism to address machine failure or
maintenance.**

In a pre-orchestration world, installation scripts would often be used
to start applications, but they did not allow recovery from machine
failure. By both creating your application instances and keeping them
running across Nodes, Kubernetes Deployments provide a fundamentally
different approach to application management.

# # Deploying your first app on Kubernetes

alert _Applications need to be packaged into one of the supported
container formats in order to be deployed on Kubernetes._ alert

You can create and manage a Deployment by using the Kubernetes command
line interface, [kubectl](docsreferencekubectl). `kubectl` uses the
Kubernetes API to interact with the cluster. In this module, youll learn
the most common `kubectl` commands needed to create Deployments that
run your applications on a Kubernetes cluster.

When you create a Deployment, youll need to specify the container image
for your application and the number of replicas that you want to run.
You can change that information later by updating your Deployment
[Module 5](docstutorialskubernetes-basicsscalescale-intro) and
[Module 6](docstutorialskubernetes-basicsupdateupdate-intro) of the
bootcamp discuss how you can scale and update your Deployments.

For your first Deployment, youll use a hello-node application packaged
in a Docker container that uses NGINX to echo back all the requests. (If
you didnt already try creating a hello-node application and deploying it
using a container, you can do that first by following the instructions
from the [Hello Minikube tutorial](docstutorialshello-minikube).

You will need to have installed kubectl as well. If you need to install
it, visit [install tools](docstaskstools#kubectl).

Now that you know what Deployments are, lets deploy our first app!

# # # kubectl basics

The common format of a kubectl command is `kubectl action resource`.

This performs the specified _action_ (like `create`, `describe` or
`delete`) on the specified _resource_ (like `node` or
`deployment`. You can use `--help` after the subcommand to get
additional info about possible parameters (for example `kubectl get
nodes --help`).

Check that kubectl is configured to talk to your cluster, by running the
`kubectl version` command.

Check that kubectl is installed and that you can see both the client and
the server versions.

To view the nodes in the cluster, run the `kubectl get nodes` command.

You see the available nodes. Later, Kubernetes will choose where to
deploy our application based on Node available resources.

# # # Deploy an app

Lets deploy our first app on Kubernetes with the `kubectl create
deployment` command. We need to provide the deployment name and app
image location (include the full repository url for images hosted
outside Docker Hub).

```shell kubectl create deployment kubernetes-bootcamp
--imagegcr.iogoogle-sampleskubernetes-bootcampv1 ```

Great! You just deployed your first application by creating a
deployment. This performed a few things for you

* searched for a suitable node where an instance of the application
could be run (we have only 1 available node) * scheduled the
application to run on that Node * configured the cluster to reschedule
the instance on a new Node when needed

To list your deployments use the `kubectl get deployments` command

```shell kubectl get deployments ```

We see that there is 1 deployment running a single instance of your app.
The instance is running inside a container on your node.

# # # View the app

[Pods](docsconceptsworkloadspods) that are running inside Kubernetes
are running on a private, isolated network. By default they are visible
from other pods and services within the same Kubernetes cluster, but not
outside that network. When we use `kubectl`, were interacting through
an API endpoint to communicate with our application.

We will cover other options on how to expose your application outside
the Kubernetes cluster later, in [Module
4](docstutorialskubernetes-basicsexpose). Also as a basic tutorial,
were not explaining what `Pods` are in any detail here, it will be
covered in later topics.

The `kubectl proxy` command can create a proxy that will forward
communications into the cluster-wide, private network. The proxy can be
terminated by pressing control-C and wont show any output while its
running.

**You need to open a second terminal window to run the proxy.**

```shell kubectl proxy ``` We now have a connection between our
host (the terminal) and the Kubernetes cluster. The proxy enables direct
access to the API from these terminals.

You can see all those APIs hosted through the proxy endpoint. For
example, we can query the version directly through the API using the
`curl` command

```shell curl httplocalhost8001version ```

If port 8001 is not accessible, ensure that the `kubectl proxy` that
you started above is running in the second terminal.

The API server will automatically create an endpoint for each pod, based
on the pod name, that is also accessible through the proxy.

First we need to get the Pod name, and well store it in the environment
variable `POD_NAME`.

```shell export POD_NAME(kubectl get pods -o go-template --template
range .items.metadata.namenend) echo Name of the Pod POD_NAME ```

You can access the Pod through the proxied API, by running

```shell curl
httplocalhost8001apiv1namespacesdefaultpodsPOD_NAME8080proxy ```

In order for the new Deployment to be accessible without using the
proxy, a Service is required which will be explained in [Module
4](docstutorialskubernetes-basicsexpose).

# # heading whatsnext

* Tutorial [Viewing Pods and
Nodes](docstutorialskubernetes-basicsexploreexplore-intro). * Learn
more about [Deployments](docsconceptsworkloadscontrollersdeployment).

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexplore_index.md
 --- title Explore
Your App weight 30 ---

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexploreexplore-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Exploring
Your App # before Katacoda shut down. # # There is no need to
localize this Not Found page the website automatically # serves a 404
response when a page is missing. If you have an existing localized #
version of this page, it is OK to remove that localized version. weight
20 headless true toc_hide true _build list never publishResources
false --- 

  Content unavailable  
The interactive tutorial for exploring your app is not available. For
more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexploreexplore-intro.md
 --- title Viewing
Pods and Nodes weight 10 ---

# # heading objectives

* Learn about Kubernetes Pods. * Learn about Kubernetes Nodes. *
Troubleshoot deployed applications.

# # Kubernetes Pods

alert _A Pod is a group of one or more application containers (such as
Docker) and includes shared storage (volumes), IP address and
information about how to run them._ alert

When you created a Deployment in [Module
2](docstutorialskubernetes-basicsdeploy-appdeploy-intro), Kubernetes
created a **Pod** to host your application instance. A Pod is a
Kubernetes abstraction that represents a group of one or more
application containers (such as Docker), and some shared resources for
those containers. Those resources include

* Shared storage, as Volumes * Networking, as a unique cluster IP
address * Information about how to run each container, such as the
container image version or specific ports to use

A Pod models an application-specific logical host and can contain
different application containers which are relatively tightly coupled.
For example, a Pod might include both the container with your Node.js
app as well as a different container that feeds the data to be published
by the Node.js webserver. The containers in a Pod share an IP Address
and port space, are always co-located and co-scheduled, and run in a
shared context on the same Node.

Pods are the atomic unit on the Kubernetes platform. When we create a
Deployment on Kubernetes, that Deployment creates Pods with containers
inside them (as opposed to creating containers directly). Each Pod is
tied to the Node where it is scheduled, and remains there until
termination (according to restart policy) or deletion. In case of a Node
failure, identical Pods are scheduled on other available Nodes in the
cluster.

# # # Pods overview

alert _Containers should only be scheduled together in a single Pod if
they are tightly coupled and need to share resources such as disk._
alert

# # Nodes

A Pod always runs on a **Node**. A Node is a worker machine in
Kubernetes and may be either a virtual or a physical machine, depending
on the cluster. Each Node is managed by the control plane. A Node can
have multiple pods, and the Kubernetes control plane automatically
handles scheduling the pods across the Nodes in the cluster. The control
planes automatic scheduling takes into account the available resources
on each Node.

Every Kubernetes Node runs at least

* Kubelet, a process responsible for communication between the
Kubernetes control plane and the Node it manages the Pods and the
containers running on a machine.

* A container runtime (like Docker) responsible for pulling the
container image from a registry, unpacking the container, and running
the application.

# # # Nodes overview

# # Troubleshooting with kubectl

In [Module 2](docstutorialskubernetes-basicsdeploy-appdeploy-intro),
you used the kubectl command-line interface. Youll continue to use it in
Module 3 to get information about deployed applications and their
environments. The most common operations can be done with the following
kubectl subcommands

* `kubectl get` - list resources * `kubectl describe` - show
detailed information about a resource * `kubectl logs` - print the
logs from a container in a pod * `kubectl exec` - execute a command
on a container in a pod

You can use these commands to see when applications were deployed, what
their current statuses are, where they are running and what their
configurations are.

Now that we know more about our cluster components and the command line,
lets explore our application.

# # # Check application configuration

Lets verify that the application we deployed in the previous scenario is
running. Well use the `kubectl get` command and look for existing Pods

```shell kubectl get pods ```

If no pods are running, please wait a couple of seconds and list the
Pods again. You can continue once you see one Pod running.

Next, to view what containers are inside that Pod and what images are
used to build those containers we run the `kubectl describe pods`
command

```shell kubectl describe pods ```

We see here details about the Pods container IP address, the ports used
and a list of events related to the lifecycle of the Pod.

The output of the `describe` subcommand is extensive and covers some
concepts that we didnt explain yet, but dont worry, they will become
familiar by the end of this tutorial.

The `describe` subcommand can be used to get detailed information
about most of the Kubernetes primitives, including Nodes, Pods, and
Deployments. The describe output is designed to be human readable, not
to be scripted against.

# # # Show the app in the terminal

Recall that Pods are running in an isolated, private network - so we
need to proxy access to them so we can debug and interact with them. To
do this, well use the `kubectl proxy` command to run a proxy in a
**second terminal**. Open a new terminal window, and in that new
terminal, run

```shell kubectl proxy ```

Now again, well get the Pod name and query that pod directly through the
proxy. To get the Pod name and store it in the `POD_NAME` environment
variable

```shell export POD_NAME(kubectl get pods -o go-template --template
range .items.metadata.namenend) echo Name of the Pod POD_NAME ```

To see the output of our application, run a `curl` request

```shell curl
httplocalhost8001apiv1namespacesdefaultpodsPOD_NAME8080proxy ```

The URL is the route to the API of the Pod.

We dont need to specify the container name, because we only have one
container inside the pod.

# # # Executing commands on the container

We can execute commands directly on the container once the Pod is up and
running. For this, we use the `exec` subcommand and use the name of
the Pod as a parameter. Lets list the environment variables

```shell kubectl exec POD_NAME -- env ```

Again, its worth mentioning that the name of the container itself can be
omitted since we only have a single container in the Pod.

Next lets start a bash session in the Pods container

```shell kubectl exec -ti POD_NAME -- bash ```

We have now an open console on the container where we run our NodeJS
application. The source code of the app is in the `server.js` file

```shell cat server.js ```

You can check that the application is up by running a curl command

```shell curl httplocalhost8080 ```

Here we used `localhost` because we executed the command inside the
NodeJS Pod. If you cannot connect to `localhost8080`, check to make
sure you have run the `kubectl exec` command and are launching the
command from within the Pod.

To close your container connection, type `exit`.

# # heading whatsnext

* Tutorial [Using A Service To Expose Your
App](docstutorialskubernetes-basicsexposeexpose-intro). * Learn more
about [Pods](docsconceptsworkloadspods). * Learn more about
[Nodes](docsconceptsarchitecturenodes).

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexpose_index.md
 --- title Expose Your
App Publicly weight 40 ---

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexposeexpose-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Exposing
Your App # before Katacoda shut down. # # There is no need to
localize this Not Found page the website automatically # serves a 404
response when a page is missing. If you have an existing localized #
version of this page, it is OK to remove that localized version. weight
20 headless true toc_hide true _build list never publishResources
false --- 

  Content unavailable  
The interactive tutorial for exposing your app is not available. For
more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsexposeexpose-intro.md
 --- title Using a
Service to Expose Your App weight 10 ---

# # heading objectives

* Learn about a Service in Kubernetes. * Understand how labels and
selectors relate to a Service. * Expose an application outside a
Kubernetes cluster.

# # Overview of Kubernetes Services

Kubernetes [Pods](docsconceptsworkloadspods) are mortal. Pods have a
[lifecycle](docsconceptsworkloadspodspod-lifecycle). When a worker
node dies, the Pods running on the Node are also lost. A
[Replicaset](docsconceptsworkloadscontrollersreplicaset) might then
dynamically drive the cluster back to the desired state via the creation
of new Pods to keep your application running. As another example,
consider an image-processing backend with 3 replicas. Those replicas are
exchangeable the front-end system should not care about backend replicas
or even if a Pod is lost and recreated. That said, each Pod in a
Kubernetes cluster has a unique IP address, even Pods on the same Node,
so there needs to be a way of automatically reconciling changes among
Pods so that your applications continue to function.

alert _A Kubernetes Service is an abstraction layer which defines a
logical set of Pods and enables external traffic exposure, load
balancing and service discovery for those Pods._ alert

A [Service](docsconceptsservices-networkingservice) in Kubernetes is
an abstraction which defines a logical set of Pods and a policy by which
to access them. Services enable a loose coupling between dependent Pods.
A Service is defined using YAML or JSON, like all Kubernetes object
manifests. The set of Pods targeted by a Service is usually determined
by a _label selector_ (see below for why you might want a Service
without including a `selector` in the spec).

Although each Pod has a unique IP address, those IPs are not exposed
outside the cluster without a Service. Services allow your applications
to receive traffic. Services can be exposed in different ways by
specifying a `type` in the `spec` of the Service

* _ClusterIP_ (default) - Exposes the Service on an internal IP in
the cluster. This type makes the Service only reachable from within the
cluster.

* _NodePort_ - Exposes the Service on the same port of each selected
Node in the cluster using NAT. Makes a Service accessible from outside
the cluster using `NodeIPNodePort`. Superset of ClusterIP.

* _LoadBalancer_ - Creates an external load balancer in the current
cloud (if supported) and assigns a fixed, external IP to the Service.
Superset of NodePort.

* _ExternalName_ - Maps the Service to the contents of the
`externalName` field (e.g. `foo.bar.example.com`), by returning a
`CNAME` record with its value. No proxying of any kind is set up. This
type requires v1.7 or higher of `kube-dns`, or CoreDNS version 0.0.8
or higher.

More information about the different types of Services can be found in
the [Using Source IP](docstutorialsservicessource-ip) tutorial. Also
see [Connecting Applications with
Services](docstutorialsservicesconnect-applications-service).

Additionally, note that there are some use cases with Services that
involve not defining a `selector` in the spec. A Service created
without `selector` will also not create the corresponding Endpoints
object. This allows users to manually map a Service to specific
endpoints. Another possibility why there may be no selector is you are
strictly using `type ExternalName`.

# # Services and Labels

A Service routes traffic across a set of Pods. Services are the
abstraction that allows pods to die and replicate in Kubernetes without
impacting your application. Discovery and routing among dependent Pods
(such as the frontend and backend components in an application) are
handled by Kubernetes Services.

Services match a set of Pods using [labels and
selectors](docsconceptsoverviewworking-with-objectslabels), a grouping
primitive that allows logical operation on objects in Kubernetes. Labels
are keyvalue pairs attached to objects and can be used in any number of
ways

* Designate objects for development, test, and production * Embed
version tags * Classify an object using tags

Labels can be attached to objects at creation time or later on. They can
be modified at any time. Lets expose our application now using a Service
and apply some labels.

# # # Step 1 Creating a new Service

Lets verify that our application is running. Well use the `kubectl
get` command and look for existing Pods

```shell kubectl get pods ```

If no Pods are running then it means the objects from the previous
tutorials were cleaned up. In this case, go back and recreate the
deployment from the [Using kubectl to create a
Deployment](docstutorialskubernetes-basicsdeploy-appdeploy-intro#deploy-an-app)
tutorial. Please wait a couple of seconds and list the Pods again. You
can continue once you see the one Pod running.

Next, lets list the current Services from our cluster

```shell kubectl get services ```

To expose the deployment to external traffic, well use the kubectl
expose command with the --typeNodePort option

```shell kubectl expose deploymentkubernetes-bootcamp --typeNodePort
--port 8080 ```

We have now a running Service called kubernetes-bootcamp. Here we see
that the Service received a unique cluster-IP, an internal port and an
external-IP (the IP of the Node).

To find out what port was opened externally (for the `type NodePort`
Service) well run the `describe service` subcommand

```shell kubectl describe serviceskubernetes-bootcamp ```

Create an environment variable called `NODE_PORT` that has the value
of the Node port assigned

```shell export NODE_PORT(kubectl get serviceskubernetes-bootcamp -o
go-template(index .spec.ports 0).nodePort) echo NODE_PORTNODE_PORT
```

Now we can test that the app is exposed outside of the cluster using
`curl`, the IP address of the Node and the externally exposed port

```shell curl http(minikube ip)NODE_PORT ```

If youre running minikube with Docker Desktop as the container driver, a
minikube tunnel is needed. This is because containers inside Docker
Desktop are isolated from your host computer.

In a separate terminal window, execute

```shell minikube service kubernetes-bootcamp --url ```

The output looks like this

``` http127.0.0.151082 ! Because you are using a Docker driver on
darwin, the terminal needs to be open to run it. ```

Then use the given URL to access the app

```shell curl 127.0.0.151082 ```

And we get a response from the server. The Service is exposed.

# # # Step 2 Using labels

The Deployment created automatically a label for our Pod. With the
`describe deployment` subcommand you can see the name (the _key_) of
that label

```shell kubectl describe deployment ```

Lets use this label to query our list of Pods. Well use the `kubectl
get pods` command with `-l` as a parameter, followed by the label
values

```shell kubectl get pods -l appkubernetes-bootcamp ``` You can do
the same to list the existing Services

```shell kubectl get services -l appkubernetes-bootcamp ```

Get the name of the Pod and store it in the POD_NAME environment
variable

```shell export POD_NAME(kubectl get pods -o go-template --template
range .items.metadata.namenend) echo Name of the Pod POD_NAME ```

To apply a new label we use the label subcommand followed by the object
type, object name and the new label

```shell kubectl label pods POD_NAME versionv1 ```

This will apply a new label to our Pod (we pinned the application
version to the Pod), and we can check it with the `describe pod`
command

```shell kubectl describe pods POD_NAME ```

We see here that the label is attached now to our Pod. And we can query
now the list of pods using the new label

```shell kubectl get pods -l versionv1 ``` And we see the Pod.

# # # Step 3 Deleting a service

To delete Services you can use the `delete service` subcommand. Labels
can be used also here

```shell kubectl delete service -l appkubernetes-bootcamp ```

Confirm that the Service is gone

```shell kubectl get services ```

This confirms that our Service was removed. To confirm that route is not
exposed anymore you can `curl` the previously exposed IP and port

```shell curl http(minikube ip)NODE_PORT ```

This proves that the application is not reachable anymore from outside
of the cluster. You can confirm that the app is still running with a
`curl` from inside the pod

```shell kubectl exec -ti POD_NAME -- curl httplocalhost8080 ```

We see here that the application is up. This is because the Deployment
is managing the application. To shut down the application, you would
need to delete the Deployment as well.

# # heading whatsnext

* Tutorial [Running Multiple Instances of Your
App](docstutorialskubernetes-basicsscalescale-intro). * Learn more
about [Service](docsconceptsservices-networkingservice).

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsscale_index.md
 --- title Scale Your
App weight 50 ---

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsscalescale-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Scaling Your
App # before Katacoda shut down. # # There is no need to localize
this Not Found page the website automatically # serves a 404 response
when a page is missing. If you have an existing localized # version of
this page, it is OK to remove that localized version. weight 20
headless true toc_hide true _build list never publishResources
false --- 

  Content unavailable  
The interactive tutorial for scaling an application thats running in
your cluster is not available. For more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsscalescale-intro.md
 --- title Running
Multiple Instances of Your App weight 10 ---

# # heading objectives

* Scale an existing app manually using kubectl.

# # Scaling an application

alert _You can create from the start a Deployment with multiple
instances using the --replicas parameter for the kubectl create
deployment command._ alert

Previously we created a
[Deployment](docsconceptsworkloadscontrollersdeployment), and then
exposed it publicly via a
[Service](docsconceptsservices-networkingservice). The Deployment
created only one Pod for running our application. When traffic
increases, we will need to scale the application to keep up with user
demand.

If you havent worked through the earlier sections, start from [Using
minikube to create a
cluster](docstutorialskubernetes-basicscreate-clustercluster-intro).

_Scaling_ is accomplished by changing the number of replicas in a
Deployment.

If you are trying this after the [previous
section](docstutorialskubernetes-basicsexposeexpose-intro), then you
may have deleted the service you created, or have created a Service of
`type NodePort`. In this section, it is assumed that a service with
`type LoadBalancer` is created for the kubernetes-bootcamp Deployment.

If you have _not_ deleted the Service created in [the previous
section](docstutorialskubernetes-basicsexposeexpose-intro), first
delete that Service and then run the following command to create a new
Service with its `type` set to `LoadBalancer`

```shell kubectl expose deploymentkubernetes-bootcamp
--typeLoadBalancer --port 8080 ```

# # Scaling overview

alert _Scaling is accomplished by changing the number of replicas in a
Deployment._ alert

Scaling out a Deployment will ensure new Pods are created and scheduled
to Nodes with available resources. Scaling will increase the number of
Pods to the new desired state. Kubernetes also supports
[autoscaling](docstasksrun-applicationhorizontal-pod-autoscale) of
Pods, but it is outside of the scope of this tutorial. Scaling to zero
is also possible, and it will terminate all Pods of the specified
Deployment.

Running multiple instances of an application will require a way to
distribute the traffic to all of them. Services have an integrated
load-balancer that will distribute network traffic to all Pods of an
exposed Deployment. Services will monitor continuously the running Pods
using endpoints, to ensure the traffic is sent only to available Pods.

Once you have multiple instances of an application running, you would be
able to do Rolling updates without downtime. Well cover that in the next
section of the tutorial. Now, lets go to the terminal and scale our
application.

# # # Scaling a Deployment

To list your Deployments, use the `get deployments` subcommand

```shell kubectl get deployments ```

The output should be similar to

``` NAME READY UP-TO-DATE AVAILABLE AGE kubernetes-bootcamp 11 1 1
11m ```

We should have 1 Pod. If not, run the command again. This shows

* _NAME_ lists the names of the Deployments in the cluster. *
_READY_ shows the ratio of CURRENTDESIRED replicas * _UP-TO-DATE_
displays the number of replicas that have been updated to achieve the
desired state. * _AVAILABLE_ displays how many replicas of the
application are available to your users. * _AGE_ displays the amount
of time that the application has been running.

To see the ReplicaSet created by the Deployment, run

```shell kubectl get rs ```

Notice that the name of the ReplicaSet is always formatted as
[DEPLOYMENT-NAME]-[RANDOM-STRING]. The random string is randomly
generated and uses the pod-template-hash as a seed.

Two important columns of this output are

* _DESIRED_ displays the desired number of replicas of the
application, which you define when you create the Deployment. This is
the desired state. * _CURRENT_ displays how many replicas are
currently running. Next, lets scale the Deployment to 4 replicas. Well
use the `kubectl scale` command, followed by the Deployment type, name
and desired number of instances

```shell kubectl scale deploymentskubernetes-bootcamp --replicas4
```

To list your Deployments once again, use `get deployments`

```shell kubectl get deployments ```

The change was applied, and we have 4 instances of the application
available. Next, lets check if the number of Pods changed

```shell kubectl get pods -o wide ```

There are 4 Pods now, with different IP addresses. The change was
registered in the Deployment events log. To check that, use the
`describe` subcommand

```shell kubectl describe deploymentskubernetes-bootcamp ```

You can also view in the output of this command that there are 4
replicas now.

# # # Load Balancing

Lets check that the Service is load-balancing the traffic. To find out
the exposed IP and Port we can use `describe service` as we learned in
the previous part of the tutorial

```shell kubectl describe serviceskubernetes-bootcamp ```

Create an environment variable called NODE_PORT that has a value as the
Node port

```shell export NODE_PORT(kubectl get serviceskubernetes-bootcamp -o
go-template(index .spec.ports 0).nodePort) echo NODE_PORTNODE_PORT
```

Next, well do a `curl` to the exposed IP address and port. Execute the
command multiple times

```shell curl http(minikube ip)NODE_PORT ```

We hit a different Pod with every request. This demonstrates that the
load-balancing is working.

The output should be similar to

``` Hello Kubernetes bootcamp! Running on
kubernetes-bootcamp-644c5687f4-wp67j v1 Hello Kubernetes bootcamp!
Running on kubernetes-bootcamp-644c5687f4-hs9dj v1 Hello Kubernetes
bootcamp! Running on kubernetes-bootcamp-644c5687f4-4hjvf v1 Hello
Kubernetes bootcamp! Running on kubernetes-bootcamp-644c5687f4-wp67j v1
Hello Kubernetes bootcamp! Running on
kubernetes-bootcamp-644c5687f4-4hjvf v1 ```

If youre running minikube with Docker Desktop as the container driver, a
minikube tunnel is needed. This is because containers inside Docker
Desktop are isolated from your host computer.

In a separate terminal window, execute

```shell minikube service kubernetes-bootcamp --url ```

The output looks like this

``` http127.0.0.151082 ! Because you are using a Docker driver on
darwin, the terminal needs to be open to run it. ```

Then use the given URL to access the app

```shell curl 127.0.0.151082 ```

# # # Scale Down

To scale down the Deployment to 2 replicas, run again the `scale`
subcommand

```shell kubectl scale deploymentskubernetes-bootcamp --replicas2
```

List the Deployments to check if the change was applied with the `get
deployments` subcommand

```shell kubectl get deployments ```

The number of replicas decreased to 2. List the number of Pods, with
`get pods`

```shell kubectl get pods -o wide ```

This confirms that 2 Pods were terminated.

# # heading whatsnext

* Tutorial [Performing a Rolling
Update](docstutorialskubernetes-basicsupdateupdate-intro). * Learn
more about [ReplicaSet](docsconceptsworkloadscontrollersreplicaset).
* Learn more about [Autoscaling](docsconceptsworkloadsautoscaling).

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsupdate_index.md
 --- title Update Your
App weight 60 ---

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsupdateupdate-interactive-gone.html
 --- title Not found
# This page was previously titled Interactive Tutorial - Updating
Your App # before Katacoda shut down. # # There is no need to
localize this Not Found page the website automatically # serves a 404
response when a page is missing. If you have an existing localized #
version of this page, it is OK to remove that localized version. weight
20 headless true toc_hide true _build list never publishResources
false --- 

  Content unavailable  
The interactive tutorial for updating an application in your cluster is
not available. For more information, see the 
shutdown announcement.  

 FILE datakubernetes
website main
content-en_docstutorialskubernetes-basicsupdateupdate-intro.md
 --- title Performing
a Rolling Update weight 10 ---

# # heading objectives

Perform a rolling update using kubectl.

# # Updating an application

alert _Rolling updates allow Deployments update to take place with zero
downtime by incrementally updating Pods instances with new ones._ alert

Users expect applications to be available all the time, and developers
are expected to deploy new versions of them several times a day. In
Kubernetes this is done with rolling updates. A **rolling update**
allows a Deployment update to take place with zero downtime. It does
this by incrementally replacing the current Pods with new ones. The new
Pods are scheduled on Nodes with available resources, and Kubernetes
waits for those new Pods to start before removing the old Pods.

In the previous module we scaled our application to run multiple
instances. This is a requirement for performing updates without
affecting application availability. By default, the maximum number of
Pods that can be unavailable during the update and the maximum number of
new Pods that can be created, is one. Both options can be configured to
either numbers or percentages (of Pods). In Kubernetes, updates are
versioned and any Deployment update can be reverted to a previous
(stable) version.

# # Rolling updates overview

alert _If a Deployment is exposed publicly, the Service will
load-balance the traffic only to available Pods during the update._
alert

Similar to application Scaling, if a Deployment is exposed publicly, the
Service will load-balance the traffic only to available Pods during the
update. An available Pod is an instance that is available to the users
of the application.

Rolling updates allow the following actions

* Promote an application from one environment to another (via container
image updates) * Rollback to previous versions * Continuous
Integration and Continuous Delivery of applications with zero downtime

In the following interactive tutorial, well update our application to a
new version, and also perform a rollback.

# # # Update the version of the app

To list your Deployments, run the `get deployments` subcommand

```shell kubectl get deployments ```

To list the running Pods, run the `get pods` subcommand

```shell kubectl get pods ```

To view the current image version of the app, run the `describe pods`
subcommand and look for the `Image` field

```shell kubectl describe pods ```

To update the image of the application to version 2, use the `set
image` subcommand, followed by the deployment name and the new image
version

```shell kubectl set image deploymentskubernetes-bootcamp
kubernetes-bootcampdocker.iojocatalinkubernetes-bootcampv2 ```

The command notified the Deployment to use a different image for your
app and initiated a rolling update. Check the status of the new Pods,
and view the old one terminating with the `get pods` subcommand

```shell kubectl get pods ```

# # # Verify an update

First, check that the service is running, as you might have deleted it
in previous tutorial step, run `describe serviceskubernetes-bootcamp`.
If its missing, you can create it again with

```shell kubectl expose deploymentkubernetes-bootcamp --typeNodePort
--port 8080 ```

Create an environment variable called `NODE_PORT` that has the value
of the Node port assigned

```shell export NODE_PORT(kubectl get serviceskubernetes-bootcamp -o
go-template(index .spec.ports 0).nodePort) echo NODE_PORTNODE_PORT
```

Next, do a `curl` to the exposed IP and port

```shell curl http(minikube ip)NODE_PORT ```

Every time you run the `curl` command, you will hit a different Pod.
Notice that all Pods are now running the latest version (`v2`).

You can also confirm the update by running the `rollout status`
subcommand

```shell kubectl rollout status deploymentskubernetes-bootcamp ```

To view the current image version of the app, run the describe pods
subcommand

```shell kubectl describe pods ```

In the `Image` field of the output, verify that you are running the
latest image version (`v2`).

# # # Roll back an update

Lets perform another update, and try to deploy an image tagged with
`v10`

```shell kubectl set image deploymentskubernetes-bootcamp
kubernetes-bootcampgcr.iogoogle-sampleskubernetes-bootcampv10 ```

Use `get deployments` to see the status of the deployment

```shell kubectl get deployments ```

Notice that the output doesnt list the desired number of available Pods.
Run the `get pods` subcommand to list all Pods

```shell kubectl get pods ```

Notice that some of the Pods have a status of `ImagePullBackOff`.

To get more insight into the problem, run the `describe pods`
subcommand

```shell kubectl describe pods ```

In the `Events` section of the output for the affected Pods, notice
that the `v10` image version did not exist in the repository.

To roll back the deployment to your last working version, use the
`rollout undo` subcommand

```shell kubectl rollout undo deploymentskubernetes-bootcamp ```

The `rollout undo` command reverts the deployment to the previous
known state (`v2` of the image). Updates are versioned and you can
revert to any previously known state of a Deployment.

Use the `get pods` subcommand to list the Pods again

```shell kubectl get pods ```

To check the image deployed on the running Pods, use the `describe
pods` subcommand

```shell kubectl describe pods ```

The Deployment is once again using a stable version of the app (`v2`).
The rollback was successful.

Remember to clean up your local cluster.

```shell kubectl delete deploymentskubernetes-bootcamp
serviceskubernetes-bootcamp ```

# # heading whatsnext

* Learn more about
[Deployments](docsconceptsworkloadscontrollersdeployment).

 FILE datakubernetes
website main content-en_docstutorialssecurity_index.md
 --- title Security
weight 40 ---

Security is an important concern for most organizations and people who
run Kubernetes clusters. You can find a basic [security
checklist](docsconceptssecuritysecurity-checklist) elsewhere in the
Kubernetes documentation.

To learn how to deploy and manage security aspects of Kubernetes, you
can follow the tutorials in this section.

 FILE datakubernetes
website main content-en_docstutorialssecurityapparmor.md
 --- reviewers -
stclair title Restrict a Containers Access to Resources with AppArmor
content_type tutorial weight 30 ---

This page shows you how to load AppArmor profiles on your nodes and
enforce those profiles in Pods. To learn more about how Kubernetes can
confine Pods using AppArmor, see [Linux kernel security constraints for
Pods and
containers](docsconceptssecuritylinux-kernel-security-constraints#apparmor).

# # heading objectives

* See an example of how to load a profile on a Node * Learn how to
enforce the profile on a Pod * Learn how to check that the profile is
loaded * See what happens when a profile is violated * See what
happens when a profile cannot be loaded

# # heading prerequisites

AppArmor is an optional kernel module and Kubernetes feature, so verify
it is supported on your Nodes before proceeding

1. AppArmor kernel module is enabled -- For the Linux kernel to
enforce an AppArmor profile, the AppArmor kernel module must be
installed and enabled. Several distributions enable the module by
default, such as Ubuntu and SUSE, and many others provide optional
support. To check whether the module is enabled, check the
`sysmoduleapparmorparametersenabled` file

```shell cat sysmoduleapparmorparametersenabled Y ```

The kubelet verifies that AppArmor is enabled on the host before
admitting a pod with AppArmor explicitly configured.

1. Container runtime supports AppArmor -- All common
Kubernetes-supported container runtimes should support AppArmor,
including and . Please refer to the corresponding runtime documentation
and verify that the cluster fulfills the requirements to use AppArmor.

1. Profile is loaded -- AppArmor is applied to a Pod by specifying an
AppArmor profile that each container should be run with. If any of the
specified profiles are not loaded in the kernel, the kubelet will reject
the Pod. You can view which profiles are loaded on a node by checking
the `syskernelsecurityapparmorprofiles` file. For example

```shell ssh gke-test-default-pool-239f5d02-gyn2 sudo cat
syskernelsecurityapparmorprofiles sort ``` ```
apparmor-test-deny-write (enforce) apparmor-test-audit-write (enforce)
docker-default (enforce) k8s-nginx (enforce) ```

For more details on loading profiles on nodes, see [Setting up nodes
with profiles](#setting-up-nodes-with-profiles).

# # Securing a Pod

Prior to Kubernetes v1.30, AppArmor was specified through annotations.
Use the documentation version selector to view the documentation with
this deprecated API.

AppArmor profiles can be specified at the pod level or container level.
The container AppArmor profile takes precedence over the pod profile.

```yaml securityContext appArmorProfile type ```

Where `` is one of

* `RuntimeDefault` to use the runtimes default profile *
`Localhost` to use a profile loaded on the host (see below) *
`Unconfined` to run without AppArmor

See [Specifying AppArmor
Confinement](#specifying-apparmor-confinement) for full details on the
AppArmor profile API.

To verify that the profile was applied, you can check that the
containers root process is running with the correct profile by examining
its proc attr

```shell kubectl exec -- cat proc1attrcurrent ```

The output should look something like this

``` cri-containerd.apparmor.d (enforce) ```

# # Example

*This example assumes you have already set up a cluster with AppArmor
support.*

First, load the profile you want to use onto your Nodes. This profile
blocks all file write operations

``` # include

profile k8s-apparmor-example-deny-write flags(attach_disconnected)
# include

file,

# Deny all file writes. deny ** w,

```

The profile needs to be loaded onto all nodes, since you dont know where
the pod will be scheduled. For this example you can use SSH to install
the profiles, but other approaches are discussed in [Setting up nodes
with profiles](#setting-up-nodes-with-profiles).

```shell # This example assumes that node names match host names,
and are reachable via SSH. NODES(( kubectl get node -o
jsonpath.items[*].status.addresses[(.type Hostname)].address ))

for NODE in NODES[*] do ssh NODE sudo apparmor_parser -q

profile k8s-apparmor-example-deny-write flags(attach_disconnected)
# include

file,

# Deny all file writes. deny ** w,

EOF done ```

Next, run a simple Hello AppArmor Pod with the deny-write profile

code_sample filepodssecurityhello-apparmor.yaml

```shell kubectl create -f hello-apparmor.yaml ```

You can verify that the container is actually running with that profile
by checking `proc1attrcurrent`

```shell kubectl exec hello-apparmor -- cat proc1attrcurrent ```

The output should be ``` k8s-apparmor-example-deny-write (enforce)
```

Finally, you can see what happens if you violate the profile by writing
to a file

```shell kubectl exec hello-apparmor -- touch tmptest ``` ```
touch tmptest Permission denied error error executing remote command
command terminated with non-zero exit code Error executing in Docker
Container 1 ```

To wrap up, see what happens if you try to specify a profile that hasnt
been loaded

```shell kubectl create -f devstdin Annotations
container.apparmor.security.beta.kubernetes.iohellolocalhostk8s-apparmor-example-allow-write
Status Pending ... Events Type Reason Age From Message ----
------ ---- ---- ------- Normal Scheduled 10s
default-scheduler Successfully assigned defaulthello-apparmor to
gke-test-default-pool-239f5d02-x1kf Normal Pulled 8s kubelet
Successfully pulled image busybox1.28 in 370.157088ms (370.172701ms
including waiting) Normal Pulling 7s (x2 over 9s) kubelet Pulling image
busybox1.28 Warning Failed 7s (x2 over 8s) kubelet Error failed to get
container spec opts failed to generate apparmor spec opts apparmor
profile not found k8s-apparmor-example-allow-write Normal Pulled 7s
kubelet Successfully pulled image busybox1.28 in 90.980331ms
(91.005869ms including waiting) ```

An Event provides the error message with the reason, the specific
wording is runtime-dependent ``` Warning Failed 7s (x2 over 8s)
kubelet Error failed to get container spec opts failed to generate
apparmor spec opts apparmor profile not found ```

# # Administration

# # # Setting up Nodes with profiles

Kubernetes does not provide any built-in mechanisms for loading AppArmor
profiles onto Nodes. Profiles can be loaded through custom
infrastructure or tools like the [Kubernetes Security Profiles
Operator](httpsgithub.comkubernetes-sigssecurity-profiles-operator).

The scheduler is not aware of which profiles are loaded onto which Node,
so the full set of profiles must be loaded onto every Node. An
alternative approach is to add a Node label for each profile (or class
of profiles) on the Node, and use a [node
selector](docsconceptsscheduling-evictionassign-pod-node) to ensure the
Pod is run on a Node with the required profile.

# # Authoring Profiles

Getting AppArmor profiles specified correctly can be a tricky business.
Fortunately there are some tools to help with that

* `aa-genprof` and `aa-logprof` generate profile rules by
monitoring an applications activity and logs, and admitting the actions
it takes. Further instructions are provided by the [AppArmor
documentation](httpsgitlab.comapparmorapparmorwikisProfiling_with_tools).
* [bane](httpsgithub.comjfrazellebane) is an AppArmor profile
generator for Docker that uses a simplified profile language.

To debug problems with AppArmor, you can check the system logs to see
what, specifically, was denied. AppArmor logs verbose messages to
`dmesg`, and errors can usually be found in the system logs or through
`journalctl`. More information is provided in [AppArmor
failures](httpsgitlab.comapparmorapparmorwikisAppArmor_Failures).

# # Specifying AppArmor confinement

Prior to Kubernetes v1.30, AppArmor was specified through annotations.
Use the documentation version selector to view the documentation with
this deprecated API.

# # # AppArmor profile within security context #appArmorProfile

You can specify the `appArmorProfile` on either a containers
`securityContext` or on a Pods `securityContext`. If the profile is
set at the pod level, it will be used as the default profile for all
containers in the pod (including init, sidecar, and ephemeral
containers). If both a pod container AppArmor profile are set, the
containers profile will be used.

An AppArmor profile has 2 fields

`type` _(required)_ - indicates which kind of AppArmor profile will
be applied. Valid options are

`Localhost` a profile pre-loaded on the node (specified by
`localhostProfile`).

`RuntimeDefault` the container runtimes default profile.

`Unconfined` no AppArmor enforcement.

`localhostProfile` - The name of a profile loaded on the node that
should be used. The profile must be preconfigured on the node to work.
This option must be provided if and only if the `type` is
`Localhost`.

# # heading whatsnext

Additional resources

* [Quick guide to the AppArmor profile
language](httpsgitlab.comapparmorapparmorwikisQuickProfileLanguage) *
[AppArmor core policy
reference](httpsgitlab.comapparmorapparmorwikisPolicy_Layout)

 FILE datakubernetes
website main content-en_docstutorialssecuritycluster-level-pss.md
 --- title Apply Pod
Security Standards at the Cluster Level content_type tutorial weight 10
---

alert titleNote This tutorial applies only for new clusters. alert

Pod Security is an admission controller that carries out checks against
the Kubernetes [Pod Security
Standards](docsconceptssecuritypod-security-standards) when new pods
are created. It is a feature GAed in v1.25. This tutorial shows you how
to enforce the `baseline` Pod Security Standard at the cluster level
which applies a standard configuration to all namespaces in a cluster.

To apply Pod Security Standards to specific namespaces, refer to [Apply
Pod Security Standards at the namespace
level](docstutorialssecurityns-level-pss).

If you are running a version of Kubernetes other than v, check the
documentation for that version.

# # heading prerequisites

Install the following on your workstation

- [kind](httpskind.sigs.k8s.iodocsuserquick-start#installation) -
[kubectl](docstaskstools)

This tutorial demonstrates what you can configure for a Kubernetes
cluster that you fully control. If you are learning how to configure Pod
Security Admission for a managed cluster where you are not able to
configure the control plane, read [Apply Pod Security Standards at the
namespace level](docstutorialssecurityns-level-pss).

# # Choose the right Pod Security Standard to apply

[Pod Security Admission](docsconceptssecuritypod-security-admission)
lets you apply built-in [Pod Security
Standards](docsconceptssecuritypod-security-standards) with the
following modes `enforce`, `audit`, and `warn`.

To gather information that helps you to choose the Pod Security
Standards that are most appropriate for your configuration, do the
following

1. Create a cluster with no Pod Security Standards applied

```shell kind create cluster --name psa-wo-cluster-pss ``` The
output is similar to ``` Creating cluster psa-wo-cluster-pss ...
Ensuring node image (kindestnodev) Preparing nodes Writing configuration
Starting control-plane Installing CNI Installing StorageClass Set
kubectl context to kind-psa-wo-cluster-pss You can now use your cluster
with

kubectl cluster-info --context kind-psa-wo-cluster-pss

Thanks for using kind! ```

1. Set the kubectl context to the new cluster

```shell kubectl cluster-info --context kind-psa-wo-cluster-pss
``` The output is similar to this

``` Kubernetes control plane is running at https127.0.0.161350

CoreDNS is running at
https127.0.0.161350apiv1namespaceskube-systemserviceskube-dnsdnsproxy

To further debug and diagnose cluster problems, use kubectl cluster-info
dump. ```

1. Get a list of namespaces in the cluster

```shell kubectl get ns ``` The output is similar to this ```
NAME STATUS AGE default Active 9m30s kube-node-lease Active 9m32s
kube-public Active 9m32s kube-system Active 9m32s local-path-storage
Active 9m26s ```

1. Use `--dry-runserver` to understand what happens when different
Pod Security Standards are applied

 1. Privileged ```shell kubectl label --dry-runserver --overwrite
ns --all pod-security.kubernetes.ioenforceprivileged ```

The output is similar to ``` namespacedefault labeled
namespacekube-node-lease labeled namespacekube-public labeled
namespacekube-system labeled namespacelocal-path-storage labeled ```
2. Baseline ```shell kubectl label --dry-runserver --overwrite ns
--all pod-security.kubernetes.ioenforcebaseline ```

The output is similar to ``` namespacedefault labeled
namespacekube-node-lease labeled namespacekube-public labeled Warning
existing pods in namespace kube-system violate the new PodSecurity
enforce level baselinelatest Warning
etcd-psa-wo-cluster-pss-control-plane (and 3 other pods) host
namespaces, hostPath volumes Warning kindnet-vzj42 non-default
capabilities, host namespaces, hostPath volumes Warning kube-proxy-m6hwf
host namespaces, hostPath volumes, privileged namespacekube-system
labeled namespacelocal-path-storage labeled ```

3. Restricted ```shell kubectl label --dry-runserver --overwrite ns
--all pod-security.kubernetes.ioenforcerestricted ```

The output is similar to ``` namespacedefault labeled
namespacekube-node-lease labeled namespacekube-public labeled Warning
existing pods in namespace kube-system violate the new PodSecurity
enforce level restrictedlatest Warning coredns-7bb9c7b568-hsptc (and 1
other pod) unrestricted capabilities, runAsNonRoot ! true,
seccompProfile Warning etcd-psa-wo-cluster-pss-control-plane (and 3
other pods) host namespaces, hostPath volumes, allowPrivilegeEscalation
! false, unrestricted capabilities, restricted volume types,
runAsNonRoot ! true Warning kindnet-vzj42 non-default capabilities, host
namespaces, hostPath volumes, allowPrivilegeEscalation ! false,
unrestricted capabilities, restricted volume types, runAsNonRoot ! true,
seccompProfile Warning kube-proxy-m6hwf host namespaces, hostPath
volumes, privileged, allowPrivilegeEscalation ! false, unrestricted
capabilities, restricted volume types, runAsNonRoot ! true,
seccompProfile namespacekube-system labeled Warning existing pods in
namespace local-path-storage violate the new PodSecurity enforce level
restrictedlatest Warning local-path-provisioner-d6d9f7ffc-lw9lh
allowPrivilegeEscalation ! false, unrestricted capabilities,
runAsNonRoot ! true, seccompProfile namespacelocal-path-storage labeled
```

From the previous output, youll notice that applying the `privileged`
Pod Security Standard shows no warnings for any namespaces. However,
`baseline` and `restricted` standards both have warnings,
specifically in the `kube-system` namespace.

# # Set modes, versions and standards

In this section, you apply the following Pod Security Standards to the
`latest` version

* `baseline` standard in `enforce` mode. * `restricted` standard
in `warn` and `audit` mode.

The `baseline` Pod Security Standard provides a convenient middle
ground that allows keeping the exemption list short and prevents known
privilege escalations.

Additionally, to prevent pods from failing in `kube-system`, youll
exempt the namespace from having Pod Security Standards applied.

When you implement Pod Security Admission in your own environment,
consider the following

1. Based on the risk posture applied to a cluster, a stricter Pod
Security Standard like `restricted` might be a better choice. 1.
Exempting the `kube-system` namespace allows pods to run as
`privileged` in this namespace. For real world use, the Kubernetes
project strongly recommends that you apply strict RBAC policies that
limit access to `kube-system`, following the principle of least
privilege. To implement the preceding standards, do the following 1.
Create a configuration file that can be consumed by the Pod Security
Admission Controller to implement these Pod Security Standards

``` mkdir -p tmppss cat tmppsscluster-level-pss.yaml apiVersion
apiserver.config.k8s.iov1 kind AdmissionConfiguration plugins  - name
PodSecurity configuration apiVersion
pod-security.admission.config.k8s.iov1 kind PodSecurityConfiguration
defaults enforce baseline enforce-version latest audit restricted
audit-version latest warn restricted warn-version latest exemptions
usernames [] runtimeClasses [] namespaces [kube-system] EOF ```

`pod-security.admission.config.k8s.iov1` configuration requires v1.25.
For v1.23 and v1.24, use
[v1beta1](httpsv1-24.docs.kubernetes.iodocstasksconfigure-pod-containerenforce-standards-admission-controller).
For v1.22, use
[v1alpha1](httpsv1-22.docs.kubernetes.iodocstasksconfigure-pod-containerenforce-standards-admission-controller).

1. Configure the API server to consume this file during cluster
creation

``` cat tmppsscluster-config.yaml kind Cluster apiVersion
kind.x-k8s.iov1alpha4 nodes  - role control-plane kubeadmConfigPatches
 - kind ClusterConfiguration apiServer extraArgs
admission-control-config-file etcconfigcluster-level-pss.yaml
extraVolumes  - name accf hostPath etcconfig mountPath etcconfig
readOnly false pathType DirectoryOrCreate extraMounts  - hostPath tmppss
containerPath etcconfig # optional if set, the mount is read-only. #
default false readOnly false # optional if set, the mount needs SELinux
relabeling. # default false selinuxRelabel false # optional set
propagation mode (None, HostToContainer or Bidirectional) # see
httpskubernetes.iodocsconceptsstoragevolumes#mount-propagation #
default None propagation None EOF ```

If you use Docker Desktop with *kind* on macOS, you can add `tmp` as
a Shared Directory under the menu item **Preferences Resources File
Sharing**.

1. Create a cluster that uses Pod Security Admission to apply these Pod
Security Standards

```shell kind create cluster --name psa-with-cluster-pss --config
tmppsscluster-config.yaml ``` The output is similar to this ```
Creating cluster psa-with-cluster-pss ... Ensuring node image
(kindestnodev) Preparing nodes Writing configuration Starting
control-plane Installing CNI Installing StorageClass Set kubectl context
to kind-psa-with-cluster-pss You can now use your cluster with

kubectl cluster-info --context kind-psa-with-cluster-pss

Have a question, bug, or feature request Let us know!
httpskind.sigs.k8s.io#community ```

1. Point kubectl to the cluster ```shell kubectl cluster-info
--context kind-psa-with-cluster-pss ``` The output is similar to
this ``` Kubernetes control plane is running at https127.0.0.163855
CoreDNS is running at
https127.0.0.163855apiv1namespaceskube-systemserviceskube-dnsdnsproxy

To further debug and diagnose cluster problems, use kubectl cluster-info
dump. ```

1. Create a Pod in the default namespace

code_sample filesecurityexample-baseline-pod.yaml

```shell kubectl apply -f
httpsk8s.ioexamplessecurityexample-baseline-pod.yaml ```

The pod is started normally, but the output includes a warning ```
Warning would violate PodSecurity restrictedlatest
allowPrivilegeEscalation ! false (container nginx must set
securityContext.allowPrivilegeEscalationfalse), unrestricted
capabilities (container nginx must set
securityContext.capabilities.drop[ALL]), runAsNonRoot ! true (pod or
container nginx must set securityContext.runAsNonRoottrue),
seccompProfile (pod or container nginx must set
securityContext.seccompProfile.type to RuntimeDefault or Localhost)
podnginx created ```

# # Clean up

Now delete the clusters which you created above by running the following
command

```shell kind delete cluster --name psa-with-cluster-pss ```
```shell kind delete cluster --name psa-wo-cluster-pss ```

# # heading whatsnext

- Run a [shell
script](examplessecuritykind-with-cluster-level-baseline-pod-security.sh)
to perform all the preceding steps at once  1. Create a Pod Security
Standards based cluster level Configuration 2. Create a file to let API
server consume this configuration 3. Create a cluster that creates an
API server with this configuration 4. Set kubectl context to this new
cluster 5. Create a minimal pod yaml file 6. Apply this file to create a
Pod in the new cluster - [Pod Security
Admission](docsconceptssecuritypod-security-admission) - [Pod Security
Standards](docsconceptssecuritypod-security-standards) - [Apply Pod
Security Standards at the namespace
level](docstutorialssecurityns-level-pss)

 FILE datakubernetes
website main content-en_docstutorialssecurityns-level-pss.md
 --- title Apply Pod
Security Standards at the Namespace Level content_type tutorial weight
20 ---

alert titleNote This tutorial applies only for new clusters. alert

Pod Security Admission is an admission controller that applies [Pod
Security Standards](docsconceptssecuritypod-security-standards) when
pods are created. It is a feature GAed in v1.25. In this tutorial, you
will enforce the `baseline` Pod Security Standard, one namespace at a
time.

You can also apply Pod Security Standards to multiple namespaces at once
at the cluster level. For instructions, refer to [Apply Pod Security
Standards at the cluster
level](docstutorialssecuritycluster-level-pss).

# # heading prerequisites

Install the following on your workstation

- [kind](httpskind.sigs.k8s.iodocsuserquick-start#installation) -
[kubectl](docstaskstools)

# # Create cluster

1. Create a `kind` cluster as follows

```shell kind create cluster --name psa-ns-level ```

The output is similar to this

``` Creating cluster psa-ns-level ... Ensuring node image
(kindestnodev) Preparing nodes Writing configuration Starting
control-plane Installing CNI Installing StorageClass Set kubectl context
to kind-psa-ns-level You can now use your cluster with

kubectl cluster-info --context kind-psa-ns-level

Not sure what to do next Check out
httpskind.sigs.k8s.iodocsuserquick-start ```

1. Set the kubectl context to the new cluster

```shell kubectl cluster-info --context kind-psa-ns-level ``` The
output is similar to this

``` Kubernetes control plane is running at https127.0.0.150996
CoreDNS is running at
https127.0.0.150996apiv1namespaceskube-systemserviceskube-dnsdnsproxy

To further debug and diagnose cluster problems, use kubectl cluster-info
dump. ```

# # Create a namespace

Create a new namespace called `example`

```shell kubectl create ns example ```

The output is similar to this

``` namespaceexample created ```

# # Enable Pod Security Standards checking for that namespace

1. Enable Pod Security Standards on this namespace using labels
supported by built-in Pod Security Admission. In this step you will
configure a check to warn on Pods that dont meet the latest version of
the _baseline_ pod security standard.

```shell kubectl label --overwrite ns example
pod-security.kubernetes.iowarnbaseline
pod-security.kubernetes.iowarn-versionlatest ```

2. You can configure multiple pod security standard checks on any
namespace, using labels. The following command will `enforce` the
`baseline` Pod Security Standard, but `warn` and `audit` for
`restricted` Pod Security Standards as per the latest version (default
value)

```shell kubectl label --overwrite ns example
pod-security.kubernetes.ioenforcebaseline
pod-security.kubernetes.ioenforce-versionlatest
pod-security.kubernetes.iowarnrestricted
pod-security.kubernetes.iowarn-versionlatest
pod-security.kubernetes.ioauditrestricted
pod-security.kubernetes.ioaudit-versionlatest ```

# # Verify the Pod Security Standard enforcement

1. Create a baseline Pod in the `example` namespace

```shell kubectl apply -n example -f
httpsk8s.ioexamplessecurityexample-baseline-pod.yaml ``` The Pod does
start OK the output includes a warning. For example

``` Warning would violate PodSecurity restrictedlatest
allowPrivilegeEscalation ! false (container nginx must set
securityContext.allowPrivilegeEscalationfalse), unrestricted
capabilities (container nginx must set
securityContext.capabilities.drop[ALL]), runAsNonRoot ! true (pod or
container nginx must set securityContext.runAsNonRoottrue),
seccompProfile (pod or container nginx must set
securityContext.seccompProfile.type to RuntimeDefault or Localhost)
podnginx created ```

1. Create a baseline Pod in the `default` namespace

```shell kubectl apply -n default -f
httpsk8s.ioexamplessecurityexample-baseline-pod.yaml ``` Output is
similar to this

``` podnginx created ```

The Pod Security Standards enforcement and warning settings were applied
only to the `example` namespace. You could create the same Pod in the
`default` namespace with no warnings.

# # Clean up

Now delete the cluster which you created above by running the following
command

```shell kind delete cluster --name psa-ns-level ```

# # heading whatsnext

- Run a [shell
script](examplessecuritykind-with-namespace-level-baseline-pod-security.sh)
to perform all the preceding steps all at once.

 1. Create kind cluster 2. Create new namespace 3. Apply `baseline`
Pod Security Standard in `enforce` mode while applying `restricted`
Pod Security Standard also in `warn` and `audit` mode. 4. Create a
new pod with the following pod security standards applied

- [Pod Security
Admission](docsconceptssecuritypod-security-admission) - [Pod Security
Standards](docsconceptssecuritypod-security-standards) - [Apply Pod
Security Standards at the cluster
level](docstutorialssecuritycluster-level-pss)

 FILE datakubernetes
website main content-en_docstutorialssecurityseccomp.md
 --- reviewers -
hasheddan - pjbgf - saschagrunert title Restrict a Containers Syscalls
with seccomp content_type tutorial weight 40
min-kubernetes-server-version v1.22 ---

Seccomp stands for secure computing mode and has been a feature of the
Linux kernel since version 2.6.12. It can be used to sandbox the
privileges of a process, restricting the calls it is able to make from
userspace into the kernel. Kubernetes lets you automatically apply
seccomp profiles loaded onto a to your Pods and containers.

Identifying the privileges required for your workloads can be difficult.
In this tutorial, you will go through how to load seccomp profiles into
a local Kubernetes cluster, how to apply them to a Pod, and how you can
begin to craft profiles that give only the necessary privileges to your
container processes.

# # heading objectives

* Learn how to load seccomp profiles on a node * Learn how to apply a
seccomp profile to a container * Observe auditing of syscalls made by a
container process * Observe behavior when a missing profile is
specified * Observe a violation of a seccomp profile * Learn how to
create fine-grained seccomp profiles * Learn how to apply a container
runtime default seccomp profile

# # heading prerequisites

In order to complete all steps in this tutorial, you must install
[kind](docstaskstools#kind) and [kubectl](docstaskstools#kubectl).

The commands used in the tutorial assume that you are using
[Docker](httpswww.docker.com) as your container runtime. (The cluster
that `kind` creates may use a different container runtime internally).
You could also use [Podman](httpspodman.io) but in that case, you
would have to follow specific
[instructions](httpskind.sigs.k8s.iodocsuserrootless) in order to
complete the tasks successfully.

This tutorial shows some examples that are still beta (since v1.25) and
others that use only generally available seccomp functionality. You
should make sure that your cluster is [configured
correctly](httpskind.sigs.k8s.iodocsuserquick-start#setting-kubernetes-version)
for the version you are using.

The tutorial also uses the `curl` tool for downloading examples to
your computer. You can adapt the steps to use a different tool if you
prefer.

It is not possible to apply a seccomp profile to a container running
with `privileged true` set in the containers `securityContext`.
Privileged containers always run as `Unconfined`.

# # Download example seccomp profiles #download-profiles

The contents of these profiles will be explored later on, but for now go
ahead and download them into a directory named `profiles` so that they
can be loaded into the cluster.

code_sample filepodssecurityseccompprofilesaudit.json

code_sample filepodssecurityseccompprofilesviolation.json

code_sample filepodssecurityseccompprofilesfine-grained.json

Run these commands

```shell mkdir .profiles curl -L -o profilesaudit.json
httpsk8s.ioexamplespodssecurityseccompprofilesaudit.json curl -L -o
profilesviolation.json
httpsk8s.ioexamplespodssecurityseccompprofilesviolation.json curl -L -o
profilesfine-grained.json
httpsk8s.ioexamplespodssecurityseccompprofilesfine-grained.json ls
profiles ```

You should see three profiles listed at the end of the final step ```
audit.json fine-grained.json violation.json ```

# # Create a local Kubernetes cluster with kind

For simplicity, [kind](httpskind.sigs.k8s.io) can be used to create a
single node cluster with the seccomp profiles loaded. Kind runs
Kubernetes in Docker, so each node of the cluster is a container. This
allows for files to be mounted in the filesystem of each container
similar to loading files onto a node.

code_sample filepodssecurityseccompkind.yaml

Download that example kind configuration, and save it to a file named
`kind.yaml` ```shell curl -L -O
httpsk8s.ioexamplespodssecurityseccompkind.yaml ```

You can set a specific Kubernetes version by setting the nodes container
image. See [Nodes](httpskind.sigs.k8s.iodocsuserconfiguration#nodes)
within the kind documentation about configuration for more details on
this. This tutorial assumes you are using Kubernetes .

As a beta feature, you can configure Kubernetes to use the profile that
the

prefers by default, rather than falling back to `Unconfined`. If you
want to try that, see [enable the use of `RuntimeDefault` as the
default seccomp profile for all
workloads](#enable-the-use-of-runtimedefault-as-the-default-seccomp-profile-for-all-workloads)
before you continue.

Once you have a kind configuration in place, create the kind cluster
with that configuration

```shell kind create cluster --configkind.yaml ```

After the new Kubernetes cluster is ready, identify the Docker container
running as the single node cluster

```shell docker ps ```

You should see output indicating that a container is running with name
`kind-control-plane`. The output is similar to

``` CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
6a96207fed4b kindestnodev1.18.2 usrlocalbinentr 27 seconds ago Up 24
seconds 127.0.0.142223-6443tcp kind-control-plane ```

If observing the filesystem of that container, you should see that the
`profiles` directory has been successfully loaded into the default
seccomp path of the kubelet. Use `docker exec` to run a command in the
Pod

```shell # Change 6a96207fed4b to the container ID you saw from
docker ps docker exec -it 6a96207fed4b ls varlibkubeletseccompprofiles
```

``` audit.json fine-grained.json violation.json ```

You have verified that these seccomp profiles are available to the
kubelet running within kind.

# # Create a Pod that uses the container runtime default seccomp
profile

Most container runtimes provide a sane set of default syscalls that are
allowed or not. You can adopt these defaults for your workload by
setting the seccomp type in the security context of a pod or container
to `RuntimeDefault`.

If you have the `seccompDefault`
[configuration](docsreferenceconfig-apikubelet-config.v1beta1)
enabled, then Pods use the `RuntimeDefault` seccomp profile whenever
no other seccomp profile is specified. Otherwise, the default is
`Unconfined`.

Heres a manifest for a Pod that requests the `RuntimeDefault` seccomp
profile for all its containers

code_sample filepodssecurityseccompgadefault-pod.yaml

Create that Pod ```shell kubectl apply -f
httpsk8s.ioexamplespodssecurityseccompgadefault-pod.yaml ```

```shell kubectl get pod default-pod ```

The Pod should be showing as having started successfully ``` NAME
READY STATUS RESTARTS AGE default-pod 11 Running 0 20s ```

Delete the Pod before moving to the next section

```shell kubectl delete pod default-pod --wait --now ```

# # Create a Pod with a seccomp profile for syscall auditing

To start off, apply the `audit.json` profile, which will log all
syscalls of the process, to a new Pod.

Heres a manifest for that Pod

code_sample filepodssecurityseccompgaaudit-pod.yaml

Older versions of Kubernetes allowed you to configure seccomp behavior
using . Kubernetes only supports using fields within
`.spec.securityContext` to configure seccomp, and this tutorial
explains that approach.

Create the Pod in the cluster

```shell kubectl apply -f
httpsk8s.ioexamplespodssecurityseccompgaaudit-pod.yaml ```

This profile does not restrict any syscalls, so the Pod should start
successfully.

```shell kubectl get pod audit-pod ```

``` NAME READY STATUS RESTARTS AGE audit-pod 11 Running 0 30s ```

In order to be able to interact with this endpoint exposed by this
container, create a NodePort that allows access to the endpoint from
inside the kind control plane container.

```shell kubectl expose pod audit-pod --type NodePort --port 5678
```

Check what port the Service has been assigned on the node.

```shell kubectl get service audit-pod ```

The output is similar to ``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S)
AGE audit-pod NodePort 10.111.36.142 567832373TCP 72s ```

Now you can use `curl` to access that endpoint from inside the kind
control plane container, at the port exposed by this Service. Use
`docker exec` to run the `curl` command within the container
belonging to that control plane container

```shell # Change 6a96207fed4b to the control plane container ID and
32373 to the port number you saw from docker ps docker exec -it
6a96207fed4b curl localhost32373 ```

``` just made some syscalls! ```

You can see that the process is running, but what syscalls did it
actually make Because this Pod is running in a local cluster, you should
be able to see those in `varlogsyslog` on your local system. Open up a
new terminal window and `tail` the output for calls from `http-echo`

```shell # The log path on your computer might be different from
varlogsyslog tail -f varlogsyslog grep http-echo ```

You should already see some logs of syscalls made by `http-echo`, and
if you run `curl` again inside the control plane container you will
see more output written to the log.

For example ``` Jul 6 153740 my-machine kernel [369128.669452]
audit type1326 audit(1594067860.48414536) auid4294967295 uid0 gid0
ses4294967295 pid29064 commhttp-echo exehttp-echo sig0 archc000003e
syscall51 compat0 ip0x46fe1f code0x7ffc0000 Jul 6 153740 my-machine
kernel [369128.669453] audit type1326 audit(1594067860.48414537)
auid4294967295 uid0 gid0 ses4294967295 pid29064 commhttp-echo
exehttp-echo sig0 archc000003e syscall54 compat0 ip0x46fdba
code0x7ffc0000 Jul 6 153740 my-machine kernel [369128.669455] audit
type1326 audit(1594067860.48414538) auid4294967295 uid0 gid0
ses4294967295 pid29064 commhttp-echo exehttp-echo sig0 archc000003e
syscall202 compat0 ip0x455e53 code0x7ffc0000 Jul 6 153740 my-machine
kernel [369128.669456] audit type1326 audit(1594067860.48414539)
auid4294967295 uid0 gid0 ses4294967295 pid29064 commhttp-echo
exehttp-echo sig0 archc000003e syscall288 compat0 ip0x46fdba
code0x7ffc0000 Jul 6 153740 my-machine kernel [369128.669517] audit
type1326 audit(1594067860.48414540) auid4294967295 uid0 gid0
ses4294967295 pid29064 commhttp-echo exehttp-echo sig0 archc000003e
syscall0 compat0 ip0x46fd44 code0x7ffc0000 Jul 6 153740 my-machine
kernel [369128.669519] audit type1326 audit(1594067860.48414541)
auid4294967295 uid0 gid0 ses4294967295 pid29064 commhttp-echo
exehttp-echo sig0 archc000003e syscall270 compat0 ip0x4559b1
code0x7ffc0000 Jul 6 153840 my-machine kernel [369188.671648] audit
type1326 audit(1594067920.48814559) auid4294967295 uid0 gid0
ses4294967295 pid29064 commhttp-echo exehttp-echo sig0 archc000003e
syscall270 compat0 ip0x4559b1 code0x7ffc0000 Jul 6 153840 my-machine
kernel [369188.671726] audit type1326 audit(1594067920.48814560)
auid4294967295 uid0 gid0 ses4294967295 pid29064 commhttp-echo
exehttp-echo sig0 archc000003e syscall202 compat0 ip0x455e53
code0x7ffc0000 ```

You can begin to understand the syscalls required by the `http-echo`
process by looking at the `syscall` entry on each line. While these
are unlikely to encompass all syscalls it uses, it can serve as a basis
for a seccomp profile for this container.

Delete the Service and the Pod before moving to the next section

```shell kubectl delete service audit-pod --wait kubectl delete pod
audit-pod --wait --now ```

# # Create a Pod with a seccomp profile that causes violation

For demonstration, apply a profile to the Pod that does not allow for
any syscalls.

The manifest for this demonstration is

code_sample filepodssecurityseccompgaviolation-pod.yaml

Attempt to create the Pod in the cluster

```shell kubectl apply -f
httpsk8s.ioexamplespodssecurityseccompgaviolation-pod.yaml ```

The Pod creates, but there is an issue. If you check the status of the
Pod, you should see that it failed to start.

```shell kubectl get pod violation-pod ```

``` NAME READY STATUS RESTARTS AGE violation-pod 01 CrashLoopBackOff
1 6s ```

As seen in the previous example, the `http-echo` process requires
quite a few syscalls. Here seccomp has been instructed to error on any
syscall by setting `defaultAction SCMP_ACT_ERRNO`. This is extremely
secure, but removes the ability to do anything meaningful. What you
really want is to give workloads only the privileges they need.

Delete the Pod before moving to the next section

```shell kubectl delete pod violation-pod --wait --now ```

# # Create a Pod with a seccomp profile that only allows necessary
syscalls

If you take a look at the `fine-grained.json` profile, you will notice
some of the syscalls seen in syslog of the first example where the
profile set `defaultAction SCMP_ACT_LOG`. Now the profile is setting
`defaultAction SCMP_ACT_ERRNO`, but explicitly allowing a set of
syscalls in the `action SCMP_ACT_ALLOW` block. Ideally, the container
will run successfully and you will see no messages sent to `syslog`.

The manifest for this example is

code_sample filepodssecurityseccompgafine-pod.yaml

Create the Pod in your cluster

```shell kubectl apply -f
httpsk8s.ioexamplespodssecurityseccompgafine-pod.yaml ```

```shell kubectl get pod fine-pod ```

The Pod should be showing as having started successfully ``` NAME
READY STATUS RESTARTS AGE fine-pod 11 Running 0 30s ```

Open up a new terminal window and use `tail` to monitor for log
entries that mention calls from `http-echo`

```shell # The log path on your computer might be different from
varlogsyslog tail -f varlogsyslog grep http-echo ```

Next, expose the Pod with a NodePort Service

```shell kubectl expose pod fine-pod --type NodePort --port 5678
```

Check what port the Service has been assigned on the node

```shell kubectl get service fine-pod ```

The output is similar to ``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S)
AGE fine-pod NodePort 10.111.36.142 567832373TCP 72s ```

Use `curl` to access that endpoint from inside the kind control plane
container

```shell # Change 6a96207fed4b to the control plane container ID and
32373 to the port number you saw from docker ps docker exec -it
6a96207fed4b curl localhost32373 ```

``` just made some syscalls! ```

You should see no output in the `syslog`. This is because the profile
allowed all necessary syscalls and specified that an error should occur
if one outside of the list is invoked. This is an ideal situation from a
security perspective, but required some effort in analyzing the program.
It would be nice if there was a simple way to get closer to this
security without requiring as much effort.

Delete the Service and the Pod before moving to the next section

```shell kubectl delete service fine-pod --wait kubectl delete pod
fine-pod --wait --now ```

# # Enable the use of `RuntimeDefault` as the default seccomp
profile for all workloads

To use seccomp profile defaulting, you must run the kubelet with the
`--seccomp-default` [command line
flag](docsreferencecommand-line-tools-referencekubelet) enabled for
each node where you want to use it.

If enabled, the kubelet will use the `RuntimeDefault` seccomp profile
by default, which is defined by the container runtime, instead of using
the `Unconfined` (seccomp disabled) mode. The default profiles aim to
provide a strong set of security defaults while preserving the
functionality of the workload. It is possible that the default profiles
differ between container runtimes and their release versions, for
example when comparing those from CRI-O and containerd.

Enabling the feature will neither change the Kubernetes
`securityContext.seccompProfile` API field nor add the deprecated
annotations of the workload. This provides users the possibility to
rollback anytime without actually changing the workload configuration.
Tools like [`crictl
inspect`](httpsgithub.comkubernetes-sigscri-tools) can be used to
verify which seccomp profile is being used by a container.

Some workloads may require a lower amount of syscall restrictions than
others. This means that they can fail during runtime even with the
`RuntimeDefault` profile. To mitigate such a failure, you can

- Run the workload explicitly as `Unconfined`. - Disable the
`SeccompDefault` feature for the nodes. Also making sure that
workloads get scheduled on nodes where the feature is disabled. - Create
a custom seccomp profile for the workload.

If you were introducing this feature into production-like cluster, the
Kubernetes project recommends that you enable this feature gate on a
subset of your nodes and then test workload execution before rolling the
change out cluster-wide.

You can find more detailed information about a possible upgrade and
downgrade strategy in the related Kubernetes Enhancement Proposal (KEP)
[Enable seccomp by
default](httpsgithub.comkubernetesenhancementstree9a124fd29d1f9ddf2ff455c49a630e3181992c25kepssig-node2413-seccomp-by-default#upgrade--downgrade-strategy).

Kubernetes lets you configure the seccomp profile that applies when the
spec for a Pod doesnt define a specific seccomp profile. However, you
still need to enable this defaulting for each node where you would like
to use it.

If you are running a Kubernetes cluster and want to enable the feature,
either run the kubelet with the `--seccomp-default` command line
flag, or enable it through the [kubelet configuration
file](docstasksadminister-clusterkubelet-config-file). To enable the
feature gate in [kind](httpskind.sigs.k8s.io), ensure that `kind`
provides the minimum required Kubernetes version and enables the
`SeccompDefault` feature [in the kind
configuration](httpskind.sigs.k8s.iodocsuserquick-start#enable-feature-gates-in-your-cluster)

```yaml kind Cluster apiVersion kind.x-k8s.iov1alpha4 nodes  - role
control-plane image
kindestnodev1.28.0sha2569f3ff58f19dcf1a0611d11e8ac989fdb30a28f40f236f59f0bea31fb956ccf5c
kubeadmConfigPatches  - kind JoinConfiguration nodeRegistration
kubeletExtraArgs seccomp-default true  - role worker image
kindestnodev1.28.0sha2569f3ff58f19dcf1a0611d11e8ac989fdb30a28f40f236f59f0bea31fb956ccf5c
kubeadmConfigPatches  - kind JoinConfiguration nodeRegistration
kubeletExtraArgs seccomp-default true ```

If the cluster is ready, then running a pod

```shell kubectl run --rm -it --restartNever --imagealpine alpine
-- sh ```

Should now have the default seccomp profile attached. This can be
verified by using `docker exec` to run `crictl inspect` for the
container on the kind worker

```shell docker exec -it kind-worker bash -c crictl inspect (crictl
ps --namealpine -q) jq .info.runtimeSpec.linux.seccomp ```

```json

defaultAction SCMP_ACT_ERRNO, architectures [SCMP_ARCH_X86_64,
SCMP_ARCH_X86, SCMP_ARCH_X32], syscalls [

names [...]

]

```

# # heading whatsnext

You can learn more about Linux seccomp

* [A seccomp Overview](httpslwn.netArticles656307) * [Seccomp
Security Profiles for
Docker](httpsdocs.docker.comenginesecurityseccomp)

 FILE datakubernetes
website main content-en_docstutorialsservices_index.md
 --- title Services
weight 70 ---

 FILE datakubernetes
website main
content-en_docstutorialsservicesconnect-applications-service.md
 --- reviewers -
caesarxuchao - lavalamp - thockin title Connecting Applications with
Services content_type tutorial weight 20 ---

# # The Kubernetes model for connecting containers

Now that you have a continuously running, replicated application you can
expose it on a network.

Kubernetes assumes that pods can communicate with other pods, regardless
of which host they land on. Kubernetes gives every pod its own
cluster-private IP address, so you do not need to explicitly create
links between pods or map container ports to host ports. This means that
containers within a Pod can all reach each others ports on localhost,
and all pods in a cluster can see each other without NAT. The rest of
this document elaborates on how you can run reliable services on such a
networking model.

This tutorial uses a simple nginx web server to demonstrate the concept.

# # Exposing pods to the cluster

We did this in a previous example, but lets do it once again and focus
on the networking perspective. Create an nginx Pod, and note that it has
a container port specification

code_sample fileservicenetworkingrun-my-nginx.yaml

This makes it accessible from any node in your cluster. Check the nodes
the Pod is running on

```shell kubectl apply -f .run-my-nginx.yaml kubectl get pods -l
runmy-nginx -o wide ``` ``` NAME READY STATUS RESTARTS AGE IP NODE
my-nginx-3800858182-jr4a2 11 Running 0 13s 10.244.3.4
kubernetes-minion-905m my-nginx-3800858182-kna2y 11 Running 0 13s
10.244.2.5 kubernetes-minion-ljyd ```

Check your pods IPs

```shell kubectl get pods -l runmy-nginx -o
custom-columnsPOD_IP.status.podIPs POD_IP [map[ip10.244.3.4]]
[map[ip10.244.2.5]] ```

You should be able to ssh into any node in your cluster and use a tool
such as `curl` to make queries against both IPs. Note that the
containers are *not* using port 80 on the node, nor are there any
special NAT rules to route traffic to the pod. This means you can run
multiple nginx pods on the same node all using the same
`containerPort`, and access them from any other pod or node in your
cluster using the assigned IP address for the pod. If you want to
arrange for a specific port on the host Node to be forwarded to backing
Pods, you can - but the networking model should mean that you do not
need to do so.

You can read more about the [Kubernetes Networking
Model](docsconceptscluster-administrationnetworking#the-kubernetes-network-model)
if youre curious.

# # Creating a Service

So we have pods running nginx in a flat, cluster wide, address space. In
theory, you could talk to these pods directly, but what happens when a
node dies The pods die with it, and the ReplicaSet inside the Deployment
will create new ones, with different IPs. This is the problem a Service
solves.

A Kubernetes Service is an abstraction which defines a logical set of
Pods running somewhere in your cluster, that all provide the same
functionality. When created, each Service is assigned a unique IP
address (also called clusterIP). This address is tied to the lifespan of
the Service, and will not change while the Service is alive. Pods can be
configured to talk to the Service, and know that communication to the
Service will be automatically load-balanced out to some pod that is a
member of the Service.

You can create a Service for your 2 nginx replicas with `kubectl
expose`

```shell kubectl expose deploymentmy-nginx ``` ```
servicemy-nginx exposed ```

This is equivalent to `kubectl apply -f` in the following yaml

code_sample fileservicenetworkingnginx-svc.yaml

This specification will create a Service which targets TCP port 80 on
any Pod with the `run my-nginx` label, and expose it on an abstracted
Service port (`targetPort` is the port the container accepts traffic
on, `port` is the abstracted Service port, which can be any port other
pods use to access the Service). View
[Service](docsreferencegeneratedkubernetes-api#service-v1-core) API
object to see the list of supported fields in service definition. Check
your Service

```shell kubectl get svc my-nginx ``` ``` NAME TYPE CLUSTER-IP
EXTERNAL-IP PORT(S) AGE my-nginx ClusterIP 10.0.162.149 80TCP 21s ```

As mentioned previously, a Service is backed by a group of Pods. These
Pods are exposed through . The Services selector will be evaluated
continuously and the results will be POSTed to an EndpointSlice that is
connected to the Service using . When a Pod dies, it is automatically
removed from the EndpointSlices that contain it as an endpoint. New Pods
that match the Services selector will automatically get added to an
EndpointSlice for that Service. Check the endpoints, and note that the
IPs are the same as the Pods created in the first step

```shell kubectl describe svc my-nginx ``` ``` Name my-nginx
Namespace default Labels runmy-nginx Annotations Selector runmy-nginx
Type ClusterIP IP Family Policy SingleStack IP Families IPv4 IP
10.0.162.149 IPs 10.0.162.149 Port 80TCP TargetPort 80TCP Endpoints
10.244.2.580,10.244.3.480 Session Affinity None Events ```
```shell kubectl get endpointslices -l
kubernetes.ioservice-namemy-nginx ``` ``` NAME ADDRESSTYPE PORTS
ENDPOINTS AGE my-nginx-7vzhx IPv4 80 10.244.2.5,10.244.3.4 21s ```

You should now be able to curl the nginx Service on `` from any node
in your cluster. Note that the Service IP is completely virtual, it
never hits the wire. If youre curious about how this works you can read
more about the [service proxy](docsreferencenetworkingvirtual-ips).

# # Accessing the Service

Kubernetes supports 2 primary modes of finding a Service - environment
variables and DNS. The former works out of the box while the latter
requires the [CoreDNS cluster
addon](httpsreleases.k8s.iovclusteraddonsdnscoredns).

If the service environment variables are not desired (because possible
clashing with expected program ones, too many variables to process, only
using DNS, etc) you can disable this mode by setting the
`enableServiceLinks` flag to `false` on the [pod
spec](docsreferencegeneratedkubernetes-apiv#pod-v1-core).

# # # Environment Variables

When a Pod runs on a Node, the kubelet adds a set of environment
variables for each active Service. This introduces an ordering problem.
To see why, inspect the environment of your running nginx Pods (your Pod
name will be different)

```shell kubectl exec my-nginx-3800858182-jr4a2 -- printenv grep
SERVICE ``` ``` KUBERNETES_SERVICE_HOST10.0.0.1
KUBERNETES_SERVICE_PORT443 KUBERNETES_SERVICE_PORT_HTTPS443 ```

Note theres no mention of your Service. This is because you created the
replicas before the Service. Another disadvantage of doing this is that
the scheduler might put both Pods on the same machine, which will take
your entire Service down if it dies. We can do this the right way by
killing the 2 Pods and waiting for the Deployment to recreate them. This
time the Service exists *before* the replicas. This will give you
scheduler-level Service spreading of your Pods (provided all your nodes
have equal capacity), as well as the right environment variables

```shell kubectl scale deployment my-nginx --replicas0 kubectl scale
deployment my-nginx --replicas2

kubectl get pods -l runmy-nginx -o wide ``` ``` NAME READY STATUS
RESTARTS AGE IP NODE my-nginx-3800858182-e9ihh 11 Running 0 5s
10.244.2.7 kubernetes-minion-ljyd my-nginx-3800858182-j4rm4 11 Running 0
5s 10.244.3.8 kubernetes-minion-905m ```

You may notice that the pods have different names, since they are killed
and recreated.

```shell kubectl exec my-nginx-3800858182-e9ihh -- printenv grep
SERVICE ``` ``` KUBERNETES_SERVICE_PORT443
MY_NGINX_SERVICE_HOST10.0.162.149 KUBERNETES_SERVICE_HOST10.0.0.1
MY_NGINX_SERVICE_PORT80 KUBERNETES_SERVICE_PORT_HTTPS443 ```

# # # DNS

Kubernetes offers a DNS cluster addon Service that automatically assigns
dns names to other Services. You can check if its running on your
cluster

```shell kubectl get services kube-dns --namespacekube-system ```
``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE kube-dns ClusterIP
10.0.0.10 53UDP,53TCP 8m ```

The rest of this section will assume you have a Service with a long
lived IP (my-nginx), and a DNS server that has assigned a name to that
IP. Here we use the CoreDNS cluster addon (application name
`kube-dns`), so you can talk to the Service from any pod in your
cluster using standard methods (e.g. `gethostbyname()`). If CoreDNS
isnt running, you can enable it referring to the [CoreDNS
README](httpsgithub.comcorednsdeploymenttreemasterkubernetes) or
[Installing
CoreDNS](docstasksadminister-clustercoredns#installing-coredns). Lets
run another curl application to test this

```shell kubectl run curl --imageradialbusyboxpluscurl -i --tty
--rm ``` ``` Waiting for pod defaultcurl-131556218-9fnch to be
running, status is Pending, pod ready false Hit enter for command prompt
```

Then, hit enter and run `nslookup my-nginx`

```shell [ rootcurl-131556218-9fnch ] nslookup my-nginx Server
10.0.0.10 Address 1 10.0.0.10

Name my-nginx Address 1 10.0.162.149 ```

# # Securing the Service

Till now we have only accessed the nginx server from within the cluster.
Before exposing the Service to the internet, you want to make sure the
communication channel is secure. For this, you will need

* Self signed certificates for https (unless you already have an
identity certificate) * An nginx server configured to use the
certificates * A [secret](docsconceptsconfigurationsecret) that makes
the certificates accessible to pods

You can acquire all these from the [nginx https
example](httpsgithub.comkubernetesexamplestreemasterstaginghttps-nginx).
This requires having go and make tools installed. If you dont want to
install those, then follow the manual steps later. In short

```shell make keys KEYtmpnginx.key CERTtmpnginx.crt kubectl create
secret tls nginxsecret --key tmpnginx.key --cert tmpnginx.crt ```
``` secretnginxsecret created ``` ```shell kubectl get secrets
``` ``` NAME TYPE DATA AGE nginxsecret kubernetes.iotls 2 1m
``` And also the configmap ```shell kubectl create configmap
nginxconfigmap --from-filedefault.conf ```

You can find an example for `default.conf` in [the Kubernetes
examples project
repo](httpsgithub.comkubernetesexamplestreebc9ca4ca32bb28762ef216386934bef20f1f9930staginghttps-nginx).

``` configmapnginxconfigmap created ``` ```shell kubectl get
configmaps ``` ``` NAME DATA AGE nginxconfigmap 1 114s ```

You can view the details of the `nginxconfigmap` ConfigMap using the
following command

```shell kubectl describe configmap nginxconfigmap ```

The output is similar to

```console Name nginxconfigmap Namespace default Labels Annotations

Data

default.conf ---- server listen 80 default_server listen []80
default_server ipv6onlyon

listen 443 ssl

root usrsharenginxhtml index index.html

server_name localhost ssl_certificate etcnginxssltls.crt
ssl_certificate_key etcnginxssltls.key

location try_files uri uri 404

BinaryData

Events ```

Following are the manual steps to follow in case you run into problems
running make (on windows for example)

```shell # Create a public private key pair openssl req -x509 -nodes
-days 365 -newkey rsa2048 -keyout dtmpnginx.key -out dtmpnginx.crt -subj
CNmy-nginxOmy-nginx # Convert the keys to base64 encoding cat
dtmpnginx.crt base64 cat dtmpnginx.key base64 ```

Use the output from the previous commands to create a yaml file as
follows. The base64 encoded value should all be on a single line.

```yaml apiVersion v1 kind Secret metadata name nginxsecret namespace
default type kubernetes.iotls data tls.crt
LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURIekNDQWdlZ0F3SUJBZ0lKQUp5M3lQK0pzMlpJTUEwR0NTcUdTSWIzRFFFQkJRVUFNQ1l4RVRBUEJnTlYKQkFNVENHNW5hVzU0YzNaak1SRXdEd1lEVlFRS0V3aHVaMmx1ZUhOMll6QWVGdzB4TnpFd01qWXdOekEzTVRKYQpGdzB4T0RFd01qWXdOekEzTVRKYU1DWXhFVEFQQmdOVkJBTVRDRzVuYVc1NGMzWmpNUkV3RHdZRFZRUUtFd2h1CloybHVlSE4yWXpDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBSjFxSU1SOVdWM0IKMlZIQlRMRmtobDRONXljMEJxYUhIQktMSnJMcy8vdzZhU3hRS29GbHlJSU94NGUrMlN5ajBFcndCLzlYTnBwbQppeW1CL3JkRldkOXg5UWhBQUxCZkVaTmNiV3NsTVFVcnhBZW50VWt1dk1vLzgvMHRpbGhjc3paenJEYVJ4NEo5Ci82UVRtVVI3a0ZTWUpOWTVQZkR3cGc3dlVvaDZmZ1Voam92VG42eHNVR0M2QURVODBpNXFlZWhNeVI1N2lmU2YKNHZpaXdIY3hnL3lZR1JBRS9mRTRqakxCdmdONjc2SU90S01rZXV3R0ljNDFhd05tNnNTSzRqYUNGeGpYSnZaZQp2by9kTlEybHhHWCtKT2l3SEhXbXNhdGp4WTRaNVk3R1ZoK0QrWnYvcW1mMFgvbVY0Rmo1NzV3ajFMWVBocWtsCmdhSXZYRyt4U1FVQ0F3RUFBYU5RTUU0d0hRWURWUjBPQkJZRUZPNG9OWkI3YXc1OUlsYkROMzhIYkduYnhFVjcKTUI4R0ExVWRJd1FZTUJhQUZPNG9OWkI3YXc1OUlsYkROMzhIYkduYnhFVjdNQXdHQTFVZEV3UUZNQU1CQWY4dwpEUVlKS29aSWh2Y05BUUVGQlFBRGdnRUJBRVhTMW9FU0lFaXdyMDhWcVA0K2NwTHI3TW5FMTducDBvMm14alFvCjRGb0RvRjdRZnZqeE04Tzd2TjB0clcxb2pGSW0vWDE4ZnZaL3k4ZzVaWG40Vm8zc3hKVmRBcStNZC9jTStzUGEKNmJjTkNUekZqeFpUV0UrKzE5NS9zb2dmOUZ3VDVDK3U2Q3B5N0M3MTZvUXRUakViV05VdEt4cXI0Nk1OZWNCMApwRFhWZmdWQTRadkR4NFo3S2RiZDY5eXM3OVFHYmg5ZW1PZ05NZFlsSUswSGt0ejF5WU4vbVpmK3FqTkJqbWZjCkNnMnlwbGQ0Wi8rUUNQZjl3SkoybFIrY2FnT0R4elBWcGxNSEcybzgvTHFDdnh6elZPUDUxeXdLZEtxaUMwSVEKQ0I5T2wwWW5scE9UNEh1b2hSUzBPOStlMm9KdFZsNUIyczRpbDlhZ3RTVXFxUlU9Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
tls.key
LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUV2UUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktjd2dnU2pBZ0VBQW9JQkFRQ2RhaURFZlZsZHdkbFIKd1V5eFpJWmVEZWNuTkFhbWh4d1NpeWF5N1AvOE9ta3NVQ3FCWmNpQ0RzZUh2dGtzbzlCSzhBZi9WemFhWm9zcApnZjYzUlZuZmNmVUlRQUN3WHhHVFhHMXJKVEVGSzhRSHA3VkpMcnpLUC9QOUxZcFlYTE0yYzZ3MmtjZUNmZitrCkU1bEVlNUJVbUNUV09UM3c4S1lPNzFLSWVuNEZJWTZMMDUrc2JGQmd1Z0ExUE5JdWFubm9UTWtlZTRuMG4rTDQKb3NCM01ZUDhtQmtRQlAzeE9JNHl3YjREZXUraURyU2pKSHJzQmlIT05Xc0RadXJFaXVJMmdoY1kxeWIyWHI2UAozVFVOcGNSbC9pVG9zQngxcHJHclk4V09HZVdPeGxZZmcvbWIvNnBuOUYvNWxlQlkrZStjSTlTMkQ0YXBKWUdpCkwxeHZzVWtGQWdNQkFBRUNnZ0VBZFhCK0xkbk8ySElOTGo5bWRsb25IUGlHWWVzZ294RGQwci9hQ1Zkank4dlEKTjIwL3FQWkUxek1yall6Ry9kVGhTMmMwc0QxaTBXSjdwR1lGb0xtdXlWTjltY0FXUTM5SjM0VHZaU2FFSWZWNgo5TE1jUHhNTmFsNjRLMFRVbUFQZytGam9QSFlhUUxLOERLOUtnNXNrSE5pOWNzMlY5ckd6VWlVZWtBL0RBUlBTClI3L2ZjUFBacDRuRWVBZmI3WTk1R1llb1p5V21SU3VKdlNyblBESGtUdW1vVlVWdkxMRHRzaG9reUxiTWVtN3oKMmJzVmpwSW1GTHJqbGtmQXlpNHg0WjJrV3YyMFRrdWtsZU1jaVlMbjk4QWxiRi9DSmRLM3QraTRoMTVlR2ZQegpoTnh3bk9QdlVTaDR2Q0o3c2Q5TmtEUGJvS2JneVVHOXBYamZhRGR2UVFLQmdRRFFLM01nUkhkQ1pKNVFqZWFKClFGdXF4cHdnNzhZTjQyL1NwenlUYmtGcVFoQWtyczJxWGx1MDZBRzhrZzIzQkswaHkzaE9zSGgxcXRVK3NHZVAKOWRERHBsUWV0ODZsY2FlR3hoc0V0L1R6cEdtNGFKSm5oNzVVaTVGZk9QTDhPTm1FZ3MxMVRhUldhNzZxelRyMgphRlpjQ2pWV1g0YnRSTHVwSkgrMjZnY0FhUUtCZ1FEQmxVSUUzTnNVOFBBZEYvL25sQVB5VWs1T3lDdWc3dmVyClUycXlrdXFzYnBkSi9hODViT1JhM05IVmpVM25uRGpHVHBWaE9JeXg5TEFrc2RwZEFjVmxvcG9HODhXYk9lMTAKMUdqbnkySmdDK3JVWUZiRGtpUGx1K09IYnRnOXFYcGJMSHBzUVpsMGhucDBYSFNYVm9CMUliQndnMGEyOFVadApCbFBtWmc2d1BRS0JnRHVIUVV2SDZHYTNDVUsxNFdmOFhIcFFnMU16M2VvWTBPQm5iSDRvZUZKZmcraEppSXlnCm9RN3hqWldVR3BIc3AyblRtcHErQWlSNzdyRVhsdlhtOElVU2FsbkNiRGlKY01Pc29RdFBZNS9NczJMRm5LQTQKaENmL0pWb2FtZm1nZEN0ZGtFMXNINE9MR2lJVHdEbTRpb0dWZGIwMllnbzFyb2htNUpLMUI3MkpBb0dBUW01UQpHNDhXOTVhL0w1eSt5dCsyZ3YvUHM2VnBvMjZlTzRNQ3lJazJVem9ZWE9IYnNkODJkaC8xT2sybGdHZlI2K3VuCnc1YytZUXRSTHlhQmd3MUtpbGhFZDBKTWU3cGpUSVpnQWJ0LzVPbnlDak9OVXN2aDJjS2lrQ1Z2dTZsZlBjNkQKckliT2ZIaHhxV0RZK2Q1TGN1YSt2NzJ0RkxhenJsSlBsRzlOZHhrQ2dZRUF5elIzT3UyMDNRVVV6bUlCRkwzZAp4Wm5XZ0JLSEo3TnNxcGFWb2RjL0d5aGVycjFDZzE2MmJaSjJDV2RsZkI0VEdtUjZZdmxTZEFOOFRwUWhFbUtKCnFBLzVzdHdxNWd0WGVLOVJmMWxXK29xNThRNTBxMmk1NVdUTThoSDZhTjlaMTltZ0FGdE5VdGNqQUx2dFYxdEYKWSs4WFJkSHJaRnBIWll2NWkwVW1VbGc9Ci0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0K
``` Now create the secrets using the file

```shell kubectl apply -f nginxsecrets.yaml kubectl get secrets
``` ``` NAME TYPE DATA AGE nginxsecret kubernetes.iotls 2 1m
```

Now modify your nginx replicas to start an https server using the
certificate in the secret, and the Service, to expose both ports (80 and
443)

code_sample fileservicenetworkingnginx-secure-app.yaml

Noteworthy points about the nginx-secure-app manifest

- It contains both Deployment and Service specification in the same
file. - The [nginx
server](httpsgithub.comkubernetesexamplestreemasterstaginghttps-nginxdefault.conf)
serves HTTP traffic on port 80 and HTTPS traffic on 443, and nginx
Service exposes both ports. - Each container has access to the keys
through a volume mounted at `etcnginxssl`. This is set up *before*
the nginx server is started.

```shell kubectl delete deployments,svc my-nginx kubectl create -f
.nginx-secure-app.yaml ```

At this point you can reach the nginx server from any node.

```shell kubectl get pods -l runmy-nginx -o
custom-columnsPOD_IP.status.podIPs POD_IP [map[ip10.244.3.5]] ```

```shell node curl -k https10.244.3.5 ... Welcome to nginx! ```

Note how we supplied the `-k` parameter to curl in the last step, this
is because we dont know anything about the pods running nginx at
certificate generation time, so we have to tell curl to ignore the CName
mismatch. By creating a Service we linked the CName used in the
certificate with the actual DNS name used by pods during Service lookup.
Lets test this from a pod (the same secret is being reused for
simplicity, the pod only needs nginx.crt to access the Service)

code_sample fileservicenetworkingcurlpod.yaml

```shell kubectl apply -f .curlpod.yaml kubectl get pods -l
appcurlpod ``` ``` NAME READY STATUS RESTARTS AGE
curl-deployment-1515033274-1410r 11 Running 0 1m ``` ```shell
kubectl exec curl-deployment-1515033274-1410r -- curl httpsmy-nginx
--cacert etcnginxssltls.crt ... Welcome to nginx! ... ```

# # Exposing the Service

For some parts of your applications you may want to expose a Service
onto an external IP address. Kubernetes supports two ways of doing this
NodePorts and LoadBalancers. The Service created in the last section
already used `NodePort`, so your nginx HTTPS replica is ready to serve
traffic on the internet if your node has a public IP.

```shell kubectl get svc my-nginx -o yaml grep nodePort -C 5 uid
07191fb3-f61a-11e5-8ae5-42010af00002 spec clusterIP 10.0.162.149 ports
 - name http nodePort 31704 port 8080 protocol TCP targetPort 80  - name
https nodePort 32453 port 443 protocol TCP targetPort 443 selector run
my-nginx ``` ```shell kubectl get nodes -o yaml grep ExternalIP -C
1  - address 104.197.41.11 type ExternalIP allocatable --  - address
23.251.152.56 type ExternalIP allocatable ...

curl https -k ... Welcome to nginx! ```

Lets now recreate the Service to use a cloud load balancer. Change the
`Type` of `my-nginx` Service from `NodePort` to `LoadBalancer`

```shell kubectl edit svc my-nginx kubectl get svc my-nginx ```
``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE my-nginx
LoadBalancer 10.0.162.149 xx.xxx.xxx.xxx 808030163TCP 21s ``` ```
curl https -k ... Welcome to nginx! ```

The IP address in the `EXTERNAL-IP` column is the one that is
available on the public internet. The `CLUSTER-IP` is only available
inside your clusterprivate cloud network.

Note that on AWS, type `LoadBalancer` creates an ELB, which uses a
(long) hostname, not an IP. Its too long to fit in the standard
`kubectl get svc` output, in fact, so youll need to do `kubectl
describe service my-nginx` to see it. Youll see something like this

```shell kubectl describe service my-nginx ... LoadBalancer Ingress
a320587ffd19711e5a37606cf4a74574-1142138393.us-east-1.elb.amazonaws.com
... ```

# # heading whatsnext

* Learn more about [Using a Service to Access an Application in a
Cluster](docstasksaccess-application-clusterservice-access-application-cluster)
* Learn more about [Connecting a Front End to a Back End Using a
Service](docstasksaccess-application-clusterconnecting-frontend-backend)
* Learn more about [Creating an External Load
Balancer](docstasksaccess-application-clustercreate-external-load-balancer)

 FILE datakubernetes
website main
content-en_docstutorialsservicespods-and-endpoint-termination-flow.md
 --- title Explore
Termination Behavior for Pods And Their Endpoints content_type tutorial
weight 60 ---

Once you connected your Application with Service following steps like
those outlined in [Connecting Applications with
Services](docstutorialsservicesconnect-applications-service), you have
a continuously running, replicated application, that is exposed on a
network. This tutorial helps you look at the termination flow for Pods
and to explore ways to implement graceful connection draining.

# # Termination process for Pods and their endpoints

There are often cases when you need to terminate a Pod - be it to
upgrade or scale down. In order to improve application availability, it
may be important to implement a proper active connections draining.

This tutorial explains the flow of Pod termination in connection with
the corresponding endpoint state and removal by using a simple nginx web
server to demonstrate the concept.

# # Example flow with endpoint termination

The following is the example flow described in the [Termination of
Pods](docsconceptsworkloadspodspod-lifecycle#pod-termination) document.

Lets say you have a Deployment containing a single `nginx` replica
(say just for the sake of demonstration purposes) and a Service

code_sample fileservicepod-with-graceful-termination.yaml

code_sample fileserviceexplore-graceful-termination-nginx.yaml

Now create the Deployment Pod and Service using the above files

```shell kubectl apply -f pod-with-graceful-termination.yaml kubectl
apply -f explore-graceful-termination-nginx.yaml ```

Once the Pod and Service are running, you can get the name of any
associated EndpointSlices

```shell kubectl get endpointslice ```

The output is similar to this

```none NAME ADDRESSTYPE PORTS ENDPOINTS AGE nginx-service-6tjbr IPv4
80 10.12.1.199,10.12.1.201 22m ```

You can see its status, and validate that there is one endpoint
registered

```shell kubectl get endpointslices -o json -l
kubernetes.ioservice-namenginx-service ```

The output is similar to this

```none

addressType IPv4, apiVersion discovery.k8s.iov1, endpoints [

addresses [ 10.12.1.201 ], conditions ready true, serving true,
terminating false ```

Now lets terminate the Pod and validate that the Pod is being terminated
respecting the graceful termination period configuration

```shell kubectl delete pod nginx-deployment-7768647bf9-b4b9s ```

All pods

```shell kubectl get pods ```

The output is similar to this

```none NAME READY STATUS RESTARTS AGE
nginx-deployment-7768647bf9-b4b9s 11 Terminating 0 4m1s
nginx-deployment-7768647bf9-rkxlw 11 Running 0 8s ```

You can see that the new pod got scheduled.

While the new endpoint is being created for the new Pod, the old
endpoint is still around in the terminating state

```shell kubectl get endpointslice -o json nginx-service-6tjbr ```

The output is similar to this

```none

addressType IPv4, apiVersion discovery.k8s.iov1, endpoints [

addresses [ 10.12.1.201 ], conditions ready false, serving true,
terminating true , nodeName gke-main-default-pool-dca1511c-d17b,
targetRef kind Pod, name nginx-deployment-7768647bf9-b4b9s, namespace
default, uid 66fa831c-7eb2-407f-bd2c-f96dfe841478 , zone us-central1-c ,

addresses [ 10.12.1.202 ], conditions ready true, serving true,
terminating false , nodeName gke-main-default-pool-dca1511c-d17b,
targetRef kind Pod, name nginx-deployment-7768647bf9-rkxlw, namespace
default, uid 722b1cbe-dcd7-4ed4-8928-4a4d0e2bbe35 , zone us-central1-c
```

This allows applications to communicate their state during termination
and clients (such as load balancers) to implement connection draining
functionality. These clients may detect terminating endpoints and
implement a special logic for them.

In Kubernetes, endpoints that are terminating always have their
`ready` status set as `false`. This needs to happen for backward
compatibility, so existing load balancers will not use it for regular
traffic. If traffic draining on terminating pod is needed, the actual
readiness can be checked as a condition `serving`.

When Pod is deleted, the old endpoint will also be deleted.

# # heading whatsnext

* Learn how to [Connect Applications with
Services](docstutorialsservicesconnect-applications-service) * Learn
more about [Using a Service to Access an Application in a
Cluster](docstasksaccess-application-clusterservice-access-application-cluster)
* Learn more about [Connecting a Front End to a Back End Using a
Service](docstasksaccess-application-clusterconnecting-frontend-backend)
* Learn more about [Creating an External Load
Balancer](docstasksaccess-application-clustercreate-external-load-balancer)

 FILE datakubernetes
website main content-en_docstutorialsservicessource-ip.md
 --- title Using
Source IP content_type tutorial min-kubernetes-server-version v1.5
weight 40 ---

Applications running in a Kubernetes cluster find and communicate with
each other, and the outside world, through the Service abstraction. This
document explains what happens to the source IP of packets sent to
different types of Services, and how you can toggle this behavior
according to your needs.

# # heading prerequisites

# # # Terminology

This document makes use of the following terms

If localizing this section, link to the equivalent Wikipedia pages for
the target localization.

[NAT](httpsen.wikipedia.orgwikiNetwork_address_translation) Network
address translation

[Source
NAT](httpsen.wikipedia.orgwikiNetwork_address_translation#SNAT)
Replacing the source IP on a packet in this page, that usually means
replacing with the IP address of a node.

[Destination
NAT](httpsen.wikipedia.orgwikiNetwork_address_translation#DNAT)
Replacing the destination IP on a packet in this page, that usually
means replacing with the IP address of a

[VIP](docsconceptsservices-networkingservice#virtual-ips-and-service-proxies)
A virtual IP address, such as the one assigned to every in Kubernetes

[kube-proxy](docsconceptsservices-networkingservice#virtual-ips-and-service-proxies)
A network daemon that orchestrates Service VIP management on every node

# # # Prerequisites

The examples use a small nginx webserver that echoes back the source IP
of requests it receives through an HTTP header. You can create it as
follows

The image in the following command only runs on AMD64 architectures.

```shell kubectl create deployment source-ip-app
--imageregistry.k8s.ioechoserver1.10 ``` The output is ```
deployment.appssource-ip-app created ```

# # heading objectives

* Expose a simple application through various types of Services *
Understand how each Service type handles source IP NAT * Understand the
tradeoffs involved in preserving source IP

# # Source IP for Services with `TypeClusterIP`

Packets sent to ClusterIP from within the cluster are never source NATd
if youre running kube-proxy in [iptables
mode](docsreferencenetworkingvirtual-ips#proxy-mode-iptables), (the
default). You can query the kube-proxy mode by fetching
`httplocalhost10249proxyMode` on the node where kube-proxy is running.

```console kubectl get nodes ``` The output is similar to this
``` NAME STATUS ROLES AGE VERSION kubernetes-node-6jst Ready 2h
v1.13.0 kubernetes-node-cx31 Ready 2h v1.13.0 kubernetes-node-jj1t Ready
2h v1.13.0 ```

Get the proxy mode on one of the nodes (kube-proxy listens on port
10249) ```shell # Run this in a shell on the node you want to query.
curl httplocalhost10249proxyMode ``` The output is ``` iptables
```

You can test source IP preservation by creating a Service over the
source IP app

```shell kubectl expose deployment source-ip-app --nameclusterip
--port80 --target-port8080 ``` The output is ```
serviceclusterip exposed ``` ```shell kubectl get svc clusterip
``` The output is similar to ``` NAME TYPE CLUSTER-IP EXTERNAL-IP
PORT(S) AGE clusterip ClusterIP 10.0.170.92 80TCP 51s ```

And hitting the `ClusterIP` from a pod in the same cluster

```shell kubectl run busybox -it --imagebusybox1.28 --restartNever
--rm ``` The output is similar to this ``` Waiting for pod
defaultbusybox to be running, status is Pending, pod ready false If you
dont see a command prompt, try pressing enter.

``` You can then run a command inside that Pod

```shell # Run this inside the terminal from kubectl run ip addr
``` ``` 1 lo mtu 65536 qdisc noqueue linkloopback 000000000000 brd
000000000000 inet 127.0.0.18 scope host lo valid_lft forever
preferred_lft forever inet6 1128 scope host valid_lft forever
preferred_lft forever 3 eth0 mtu 1460 qdisc noqueue linkether
0a580af40308 brd ffffffffffff inet 10.244.3.824 scope global eth0
valid_lft forever preferred_lft forever inet6 fe80188a84fffeb026a564
scope link valid_lft forever preferred_lft forever ```

then use `wget` to query the local webserver ```shell # Replace
10.0.170.92 with the IPv4 address of the Service named clusterip wget
-qO - 10.0.170.92 ``` ``` CLIENT VALUES client_address10.244.3.8
commandGET ... ``` The `client_address` is always the client pods
IP address, whether the client pod and server pod are in the same node
or in different nodes.

# # Source IP for Services with `TypeNodePort`

Packets sent to Services with
[`TypeNodePort`](docsconceptsservices-networkingservice#type-nodeport)
are source NATd by default. You can test this by creating a `NodePort`
Service

```shell kubectl expose deployment source-ip-app --namenodeport
--port80 --target-port8080 --typeNodePort ``` The output is ```
servicenodeport exposed ```

```shell NODEPORT(kubectl get -o jsonpath.spec.ports[0].nodePort
services nodeport) NODES(kubectl get nodes -o jsonpath
.items[*].status.addresses[(.typeInternalIP)].address ) ```

If youre running on a cloud provider, you may need to open up a
firewall-rule for the `nodesnodeport` reported above. Now you can try
reaching the Service from outside the cluster through the node port
allocated above.

```shell for node in NODES do curl -s nodeNODEPORT grep -i
client_address done ``` The output is similar to ```
client_address10.180.1.1 client_address10.240.0.5
client_address10.240.0.3 ```

Note that these are not the correct client IPs, theyre cluster internal
IPs. This is what happens

* Client sends packet to `node2nodePort` * `node2` replaces the
source IP address (SNAT) in the packet with its own IP address *
`node2` replaces the destination IP on the packet with the pod IP *
packet is routed to node 1, and then to the endpoint * the pods reply
is routed back to node2 * the pods reply is sent back to the client

Visually

To avoid this, Kubernetes has a feature to [preserve the client source
IP](docstasksaccess-application-clustercreate-external-load-balancer#preserving-the-client-source-ip).
If you set `service.spec.externalTrafficPolicy` to the value
`Local`, kube-proxy only proxies proxy requests to local endpoints,
and does not forward traffic to other nodes. This approach preserves the
original source IP address. If there are no local endpoints, packets
sent to the node are dropped, so you can rely on the correct source-ip
in any packet processing rules you might apply a packet that make it
through to the endpoint.

Set the `service.spec.externalTrafficPolicy` field as follows

```shell kubectl patch svc nodeport -p specexternalTrafficPolicyLocal
``` The output is ``` servicenodeport patched ```

Now, re-run the test

```shell for node in NODES do curl --connect-timeout 1 -s
nodeNODEPORT grep -i client_address done ``` The output is similar to
``` client_address198.51.100.79 ```

Note that you only got one reply, with the *right* client IP, from the
one node on which the endpoint pod is running.

This is what happens

* client sends packet to `node2nodePort`, which doesnt have any
endpoints * packet is dropped * client sends packet to
`node1nodePort`, which *does* have endpoints * node1 routes packet
to endpoint with the correct source IP

Visually

# # Source IP for Services with `TypeLoadBalancer`

Packets sent to Services with
[`TypeLoadBalancer`](docsconceptsservices-networkingservice#loadbalancer)
are source NATd by default, because all schedulable Kubernetes nodes in
the `Ready` state are eligible for load-balanced traffic. So if
packets arrive at a node without an endpoint, the system proxies it to a
node *with* an endpoint, replacing the source IP on the packet with
the IP of the node (as described in the previous section).

You can test this by exposing the source-ip-app through a load balancer

```shell kubectl expose deployment source-ip-app --nameloadbalancer
--port80 --target-port8080 --typeLoadBalancer ``` The output is
``` serviceloadbalancer exposed ```

Print out the IP addresses of the Service ```console kubectl get svc
loadbalancer ``` The output is similar to this ``` NAME TYPE
CLUSTER-IP EXTERNAL-IP PORT(S) AGE loadbalancer LoadBalancer 10.0.65.118
203.0.113.140 80TCP 5m ```

Next, send a request to this Services external-ip

```shell curl 203.0.113.140 ``` The output is similar to this
``` CLIENT VALUES client_address10.240.0.5 ... ```

However, if youre running on Google Kubernetes EngineGCE, setting the
same `service.spec.externalTrafficPolicy` field to `Local` forces
nodes *without* Service endpoints to remove themselves from the list
of nodes eligible for loadbalanced traffic by deliberately failing
health checks.

Visually

![Source IP with
externalTrafficPolicy](imagesdocssourceip-externaltrafficpolicy.svg)

You can test this by setting the annotation

```shell kubectl patch svc loadbalancer -p
specexternalTrafficPolicyLocal ```

You should immediately see the `service.spec.healthCheckNodePort`
field allocated by Kubernetes

```shell kubectl get svc loadbalancer -o yaml grep -i
healthCheckNodePort ``` The output is similar to this ```yaml
healthCheckNodePort 32122 ```

The `service.spec.healthCheckNodePort` field points to a port on every
node serving the health check at `healthz`. You can test this

```shell kubectl get pod -o wide -l appsource-ip-app ``` The
output is similar to this ``` NAME READY STATUS RESTARTS AGE IP NODE
source-ip-app-826191075-qehz4 11 Running 0 20h 10.180.1.136
kubernetes-node-6jst ```

Use `curl` to fetch the `healthz` endpoint on various nodes
```shell # Run this locally on a node you choose curl
localhost32122healthz ``` ``` 1 Service Endpoints found ```

On a different node you might get a different result ```shell # Run
this locally on a node you choose curl localhost32122healthz ```
``` No Service Endpoints Found ```

A controller running on the is responsible for allocating the cloud load
balancer. The same controller also allocates HTTP health checks pointing
to this portpath on each node. Wait about 10 seconds for the 2 nodes
without endpoints to fail health checks, then use `curl` to query the
IPv4 address of the load balancer

```shell curl 203.0.113.140 ``` The output is similar to this
``` CLIENT VALUES client_address198.51.100.79 ... ```

# # Cross-platform support

Only some cloud providers offer support for source IP preservation
through Services with `TypeLoadBalancer`. The cloud provider youre
running on might fulfill the request for a loadbalancer in a few
different ways

1. With a proxy that terminates the client connection and opens a new
connection to your nodesendpoints. In such cases the source IP will
always be that of the cloud LB, not that of the client.

2. With a packet forwarder, such that requests from the client sent to
the loadbalancer VIP end up at the node with the source IP of the
client, not an intermediate proxy.

Load balancers in the first category must use an agreed upon protocol
between the loadbalancer and backend to communicate the true client IP
such as the HTTP
[Forwarded](httpstools.ietf.orghtmlrfc7239#section-5.2) or
[X-FORWARDED-FOR](httpsen.wikipedia.orgwikiX-Forwarded-For) headers,
or the [proxy
protocol](httpswww.haproxy.orgdownload1.8docproxy-protocol.txt). Load
balancers in the second category can leverage the feature described
above by creating an HTTP health check pointing at the port stored in
the `service.spec.healthCheckNodePort` field on the Service.

# # heading cleanup

Delete the Services

```shell kubectl delete svc -l appsource-ip-app ```

Delete the Deployment, ReplicaSet and Pod

```shell kubectl delete deployment source-ip-app ```

# # heading whatsnext

* Learn more about [connecting applications via
services](docstutorialsservicesconnect-applications-service) * Read
how to [Create an External Load
Balancer](docstasksaccess-application-clustercreate-external-load-balancer)

 FILE datakubernetes
website main content-en_docstutorialsstateful-application_index.md
 --- title Stateful
Applications weight 50 ---

 FILE datakubernetes
website main
content-en_docstutorialsstateful-applicationbasic-stateful-set.md
 --- reviewers -
enisoc - erictune - foxish - janetkuo - kow3ns - smarterclayton title
StatefulSet Basics content_type tutorial weight 10 ---

This tutorial provides an introduction to managing applications with .
It demonstrates how to create, delete, scale, and update the Pods of
StatefulSets.

# # heading prerequisites

Before you begin this tutorial, you should familiarize yourself with the
following Kubernetes concepts

* [Pods](docsconceptsworkloadspods) * [Cluster
DNS](docsconceptsservices-networkingdns-pod-service) * [Headless
Services](docsconceptsservices-networkingservice#headless-services) *
[PersistentVolumes](docsconceptsstoragepersistent-volumes) *
[PersistentVolume
Provisioning](httpsgithub.comkubernetesexamplestreemasterstagingpersistent-volume-provisioning)
* The [kubectl](docsreferencekubectlkubectl) command line tool

include task-tutorial-prereqs.md You should configure `kubectl` to use
a context that uses the `default` namespace. If you are using an
existing cluster, make sure that its OK to use that clusters default
namespace to practice. Ideally, practice in a cluster that doesnt run
any real workloads.

Its also useful to read the concept page about
[StatefulSets](docsconceptsworkloadscontrollersstatefulset).

This tutorial assumes that your cluster is configured to dynamically
provision PersistentVolumes. Youll also need to have a [default
StorageClass](docsconceptsstoragestorage-classes#default-storageclass).
If your cluster is not configured to provision storage dynamically, you
will have to manually provision two 1 GiB volumes prior to starting this
tutorial and set up your cluster so that those PersistentVolumes map to
the PersistentVolumeClaim templates that the StatefulSet defines.

# # heading objectives

StatefulSets are intended to be used with stateful applications and
distributed systems. However, the administration of stateful
applications and distributed systems on Kubernetes is a broad, complex
topic. In order to demonstrate the basic features of a StatefulSet, and
not to conflate the former topic with the latter, you will deploy a
simple web application using a StatefulSet.

After this tutorial, you will be familiar with the following.

* How to create a StatefulSet * How a StatefulSet manages its Pods *
How to delete a StatefulSet * How to scale a StatefulSet * How to
update a StatefulSets Pods

# # Creating a StatefulSet

Begin by creating a StatefulSet (and the Service that it relies upon)
using the example below. It is similar to the example presented in the
[StatefulSets](docsconceptsworkloadscontrollersstatefulset) concept.
It creates a [headless
Service](docsconceptsservices-networkingservice#headless-services),
`nginx`, to publish the IP addresses of Pods in the StatefulSet,
`web`.

code_sample fileapplicationwebweb.yaml

You will need to use at least two terminal windows. In the first
terminal, use [`kubectl
get`](docsreferencegeneratedkubectlkubectl-commands#get) to the
creation of the StatefulSets Pods.

```shell # use this terminal to run commands that specify --watch
# end this watch when you are asked to start a new watch kubectl get
pods --watch -l appnginx ```

In the second terminal, use [`kubectl
apply`](docsreferencegeneratedkubectlkubectl-commands#apply) to create
the headless Service and StatefulSet

```shell kubectl apply -f httpsk8s.ioexamplesapplicationwebweb.yaml
``` ``` servicenginx created statefulset.appsweb created ```

The command above creates two Pods, each running an
[NGINX](httpswww.nginx.com) webserver. Get the `nginx` Service...
```shell kubectl get service nginx ``` ``` NAME TYPE CLUSTER-IP
EXTERNAL-IP PORT(S) AGE nginx ClusterIP None 80TCP 12s ``` ...then
get the `web` StatefulSet, to verify that both were created
successfully ```shell kubectl get statefulset web ``` ``` NAME
READY AGE web 22 37s ```

# # # Ordered Pod creation

A StatefulSet defaults to creating its Pods in a strict order.

For a StatefulSet with _n_ replicas, when Pods are being deployed,
they are created sequentially, ordered from _0..n-1_. Examine the
output of the `kubectl get` command in the first terminal. Eventually,
the output will look like the example below.

```shell # Do not start a new watch # this should already be
running kubectl get pods --watch -l appnginx ``` ``` NAME READY
STATUS RESTARTS AGE web-0 01 Pending 0 0s web-0 01 Pending 0 0s web-0 01
ContainerCreating 0 0s web-0 11 Running 0 19s web-1 01 Pending 0 0s
web-1 01 Pending 0 0s web-1 01 ContainerCreating 0 0s web-1 11 Running 0
18s ```

Notice that the `web-1` Pod is not launched until the `web-0` Pod is
_Running_ (see [Pod
Phase](docsconceptsworkloadspodspod-lifecycle#pod-phase)) and _Ready_
(see `type` in [Pod
Conditions](docsconceptsworkloadspodspod-lifecycle#pod-conditions)).

Later in this tutorial you will practice [parallel
startup](#parallel-pod-management).

To configure the integer ordinal assigned to each Pod in a StatefulSet,
see [Start
ordinal](docsconceptsworkloadscontrollersstatefulset#start-ordinal).

# # Pods in a StatefulSet

Pods in a StatefulSet have a unique ordinal index and a stable network
identity.

# # # Examining the Pods ordinal index

Get the StatefulSets Pods

```shell kubectl get pods -l appnginx ``` ``` NAME READY STATUS
RESTARTS AGE web-0 11 Running 0 1m web-1 11 Running 0 1m ```

As mentioned in the
[StatefulSets](docsconceptsworkloadscontrollersstatefulset) concept,
the Pods in a StatefulSet have a sticky, unique identity. This identity
is based on a unique ordinal index that is assigned to each Pod by the
StatefulSet . The Pods names take the form `-`. Since the `web`
StatefulSet has two replicas, it creates two Pods, `web-0` and
`web-1`.

# # # Using stable network identities

Each Pod has a stable hostname based on its ordinal index. Use
[`kubectl exec`](docsreferencegeneratedkubectlkubectl-commands#exec)
to execute the `hostname` command in each Pod

```shell for i in 0 1 do kubectl exec web-i -- sh -c hostname done
``` ``` web-0 web-1 ```

Use [`kubectl
run`](docsreferencegeneratedkubectlkubectl-commands#run) to execute a
container that provides the `nslookup` command from the `dnsutils`
package. Using `nslookup` on the Pods hostnames, you can examine their
in-cluster DNS addresses

```shell kubectl run -i --tty --image busybox1.28 dns-test
--restartNever --rm ``` which starts a new shell. In that new
shell, run ```shell # Run this in the dns-test container shell
nslookup web-0.nginx ``` The output is similar to ``` Server
10.0.0.10 Address 1 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name web-0.nginx Address 1 10.244.1.6

nslookup web-1.nginx Server 10.0.0.10 Address 1 10.0.0.10
kube-dns.kube-system.svc.cluster.local

Name web-1.nginx Address 1 10.244.2.6 ```

(and now exit the container shell `exit`)

The CNAME of the headless service points to SRV records (one for each
Pod that is Running and Ready). The SRV records point to A record
entries that contain the Pods IP addresses.

In one terminal, watch the StatefulSets Pods

```shell # Start a new watch # End this watch when youve seen that
the delete is finished kubectl get pod --watch -l appnginx ``` In a
second terminal, use [`kubectl
delete`](docsreferencegeneratedkubectlkubectl-commands#delete) to
delete all the Pods in the StatefulSet

```shell kubectl delete pod -l appnginx ``` ``` pod web-0
deleted pod web-1 deleted ```

Wait for the StatefulSet to restart them, and for both Pods to
transition to Running and Ready

```shell # This should already be running kubectl get pod --watch
-l appnginx ``` ``` NAME READY STATUS RESTARTS AGE web-0 01
ContainerCreating 0 0s NAME READY STATUS RESTARTS AGE web-0 11 Running 0
2s web-1 01 Pending 0 0s web-1 01 Pending 0 0s web-1 01
ContainerCreating 0 0s web-1 11 Running 0 34s ```

Use `kubectl exec` and `kubectl run` to view the Pods hostnames and
in-cluster DNS entries. First, view the Pods hostnames

```shell for i in 0 1 do kubectl exec web-i -- sh -c hostname done
``` ``` web-0 web-1 ``` then, run ```shell kubectl run -i
--tty --image busybox1.28 dns-test --restartNever --rm ``` which
starts a new shell. In that new shell, run ```shell # Run this in
the dns-test container shell nslookup web-0.nginx ``` The output is
similar to ``` Server 10.0.0.10 Address 1 10.0.0.10
kube-dns.kube-system.svc.cluster.local

Name web-0.nginx Address 1 10.244.1.7

nslookup web-1.nginx Server 10.0.0.10 Address 1 10.0.0.10
kube-dns.kube-system.svc.cluster.local

Name web-1.nginx Address 1 10.244.2.8 ```

(and now exit the container shell `exit`)

The Pods ordinals, hostnames, SRV records, and A record names have not
changed, but the IP addresses associated with the Pods may have changed.
In the cluster used for this tutorial, they have. This is why it is
important not to configure other applications to connect to Pods in a
StatefulSet by the IP address of a particular Pod (it is OK to connect
to Pods by resolving their hostname).

# # # # Discovery for specific Pods in a StatefulSet

If you need to find and connect to the active members of a StatefulSet,
you should query the CNAME of the headless Service
(`nginx.default.svc.cluster.local`). The SRV records associated with
the CNAME will contain only the Pods in the StatefulSet that are Running
and Ready.

If your application already implements connection logic that tests for
liveness and readiness, you can use the SRV records of the Pods (
`web-0.nginx.default.svc.cluster.local`,
`web-1.nginx.default.svc.cluster.local`), as they are stable, and your
application will be able to discover the Pods addresses when they
transition to Running and Ready.

If your application wants to find any healthy Pod in a StatefulSet, and
therefore does not need to track each specific Pod, you could also
connect to the IP address of a `type ClusterIP` Service, backed by the
Pods in that StatefulSet. You can use the same Service that tracks the
StatefulSet (specified in the `serviceName` of the StatefulSet) or a
separate Service that selects the right set of Pods.

# # # Writing to stable storage

Get the PersistentVolumeClaims for `web-0` and `web-1`

```shell kubectl get pvc -l appnginx ``` The output is similar to
``` NAME STATUS VOLUME CAPACITY ACCESSMODES AGE www-web-0 Bound
pvc-15c268c7-b507-11e6-932f-42010a800002 1Gi RWO 48s www-web-1 Bound
pvc-15c79307-b507-11e6-932f-42010a800002 1Gi RWO 48s ```

The StatefulSet controller created two

that are bound to two .

As the cluster used in this tutorial is configured to dynamically
provision PersistentVolumes, the PersistentVolumes were created and
bound automatically.

The NGINX webserver, by default, serves an index file from
`usrsharenginxhtmlindex.html`. The `volumeMounts` field in the
StatefulSets `spec` ensures that the `usrsharenginxhtml` directory
is backed by a PersistentVolume.

Write the Pods hostnames to their `index.html` files and verify that
the NGINX webservers serve the hostnames

```shell for i in 0 1 do kubectl exec web-i -- sh -c echo (hostname)
usrsharenginxhtmlindex.html done

for i in 0 1 do kubectl exec -i -t web-i -- curl httplocalhost done
``` ``` web-0 web-1 ```

If you instead see **403 Forbidden** responses for the above curl
command, you will need to fix the permissions of the directory mounted
by the `volumeMounts` (due to a [bug when using hostPath
volumes](httpsgithub.comkuberneteskubernetesissues2630)), by running

`for i in 0 1 do kubectl exec web-i -- chmod 755 usrsharenginxhtml
done`

before retrying the `curl` command above.

In one terminal, watch the StatefulSets Pods

```shell # End this watch when youve reached the end of the section.
# At the start of Scaling a StatefulSet youll start a new watch.
kubectl get pod --watch -l appnginx ```

In a second terminal, delete all of the StatefulSets Pods

```shell kubectl delete pod -l appnginx ``` ``` pod web-0
deleted pod web-1 deleted ``` Examine the output of the `kubectl
get` command in the first terminal, and wait for all of the Pods to
transition to Running and Ready.

```shell # This should already be running kubectl get pod --watch
-l appnginx ``` ``` NAME READY STATUS RESTARTS AGE web-0 01
ContainerCreating 0 0s NAME READY STATUS RESTARTS AGE web-0 11 Running 0
2s web-1 01 Pending 0 0s web-1 01 Pending 0 0s web-1 01
ContainerCreating 0 0s web-1 11 Running 0 34s ```

Verify the web servers continue to serve their hostnames

``` for i in 0 1 do kubectl exec -i -t web-i -- curl httplocalhost
done ``` ``` web-0 web-1 ```

Even though `web-0` and `web-1` were rescheduled, they continue to
serve their hostnames because the PersistentVolumes associated with
their PersistentVolumeClaims are remounted to their `volumeMounts`. No
matter what node `web-0`and `web-1` are scheduled on, their
PersistentVolumes will be mounted to the appropriate mount points.

# # Scaling a StatefulSet

Scaling a StatefulSet refers to increasing or decreasing the number of
replicas (horizontal scaling). This is accomplished by updating the
`replicas` field. You can use either [`kubectl
scale`](docsreferencegeneratedkubectlkubectl-commands#scale) or
[`kubectl
patch`](docsreferencegeneratedkubectlkubectl-commands#patch) to scale
a StatefulSet.

# # # Scaling up

Scaling up means adding more replicas. Provided that your app is able to
distribute work across the StatefulSet, the new larger set of Pods can
perform more of that work.

In one terminal window, watch the Pods in the StatefulSet

```shell # If you already have a watch running, you can continue
using that. # Otherwise, start one. # End this watch when there are 5
healthy Pods for the StatefulSet kubectl get pods --watch -l appnginx
```

In another terminal window, use `kubectl scale` to scale the number of
replicas to 5

```shell kubectl scale sts web --replicas5 ``` ```
statefulset.appsweb scaled ```

Examine the output of the `kubectl get` command in the first terminal,
and wait for the three additional Pods to transition to Running and
Ready.

```shell # This should already be running kubectl get pod --watch
-l appnginx ``` ``` NAME READY STATUS RESTARTS AGE web-0 11
Running 0 2h web-1 11 Running 0 2h NAME READY STATUS RESTARTS AGE web-2
01 Pending 0 0s web-2 01 Pending 0 0s web-2 01 ContainerCreating 0 0s
web-2 11 Running 0 19s web-3 01 Pending 0 0s web-3 01 Pending 0 0s web-3
01 ContainerCreating 0 0s web-3 11 Running 0 18s web-4 01 Pending 0 0s
web-4 01 Pending 0 0s web-4 01 ContainerCreating 0 0s web-4 11 Running 0
19s ```

The StatefulSet controller scaled the number of replicas. As with
[StatefulSet creation](#ordered-pod-creation), the StatefulSet
controller created each Pod sequentially with respect to its ordinal
index, and it waited for each Pods predecessor to be Running and Ready
before launching the subsequent Pod.

# # # Scaling down

Scaling down means reducing the number of replicas. For example, you
might do this because the level of traffic to a service has decreased,
and at the current scale there are idle resources.

In one terminal, watch the StatefulSets Pods

```shell # End this watch when there are only 3 Pods for the
StatefulSet kubectl get pod --watch -l appnginx ```

In another terminal, use `kubectl patch` to scale the StatefulSet back
down to three replicas

```shell kubectl patch sts web -p specreplicas3 ``` ```
statefulset.appsweb patched ```

Wait for `web-4` and `web-3` to transition to Terminating.

```shell # This should already be running kubectl get pods --watch
-l appnginx ``` ``` NAME READY STATUS RESTARTS AGE web-0 11
Running 0 3h web-1 11 Running 0 3h web-2 11 Running 0 55s web-3 11
Running 0 36s web-4 01 ContainerCreating 0 18s NAME READY STATUS
RESTARTS AGE web-4 11 Running 0 19s web-4 11 Terminating 0 24s web-4 11
Terminating 0 24s web-3 11 Terminating 0 42s web-3 11 Terminating 0 42s
```

# # # Ordered Pod termination

The control plane deleted one Pod at a time, in reverse order with
respect to its ordinal index, and it waited for each Pod to be
completely shut down before deleting the next one.

Get the StatefulSets PersistentVolumeClaims

```shell kubectl get pvc -l appnginx ``` ``` NAME STATUS VOLUME
CAPACITY ACCESSMODES AGE www-web-0 Bound
pvc-15c268c7-b507-11e6-932f-42010a800002 1Gi RWO 13h www-web-1 Bound
pvc-15c79307-b507-11e6-932f-42010a800002 1Gi RWO 13h www-web-2 Bound
pvc-e1125b27-b508-11e6-932f-42010a800002 1Gi RWO 13h www-web-3 Bound
pvc-e1176df6-b508-11e6-932f-42010a800002 1Gi RWO 13h www-web-4 Bound
pvc-e11bb5f8-b508-11e6-932f-42010a800002 1Gi RWO 13h

```

There are still five PersistentVolumeClaims and five PersistentVolumes.
When exploring a Pods [stable storage](#writing-to-stable-storage),
you saw that the PersistentVolumes mounted to the Pods of a StatefulSet
are not deleted when the StatefulSets Pods are deleted. This is still
true when Pod deletion is caused by scaling the StatefulSet down.

# # Updating StatefulSets

The StatefulSet controller supports automated updates. The strategy used
is determined by the `spec.updateStrategy` field of the StatefulSet
API object. This feature can be used to upgrade the container images,
resource requests andor limits, labels, and annotations of the Pods in a
StatefulSet.

There are two valid update strategies, `RollingUpdate` (the default)
and `OnDelete`.

# # # RollingUpdate #rolling-update

The `RollingUpdate` update strategy will update all Pods in a
StatefulSet, in reverse ordinal order, while respecting the StatefulSet
guarantees.

You can split updates to a StatefulSet that uses the `RollingUpdate`
strategy into _partitions_, by specifying
`.spec.updateStrategy.rollingUpdate.partition`. Youll practice that
later in this tutorial.

First, try a simple rolling update.

In one terminal window, patch the `web` StatefulSet to change the
container image again

```shell kubectl patch statefulset web --typejson -p[op replace,
path spectemplatespeccontainers0image,
valueregistry.k8s.ionginx-slim0.24] ``` ``` statefulset.appsweb
patched ```

In another terminal, watch the Pods in the StatefulSet

```shell # End this watch when the rollout is complete # # If
youre not sure, leave it running one more minute kubectl get pod -l
appnginx --watch ``` The output is similar to ``` NAME READY
STATUS RESTARTS AGE web-0 11 Running 0 7m web-1 11 Running 0 7m web-2 11
Running 0 8m web-2 11 Terminating 0 8m web-2 11 Terminating 0 8m web-2
01 Terminating 0 8m web-2 01 Terminating 0 8m web-2 01 Terminating 0 8m
web-2 01 Terminating 0 8m web-2 01 Pending 0 0s web-2 01 Pending 0 0s
web-2 01 ContainerCreating 0 0s web-2 11 Running 0 19s web-1 11
Terminating 0 8m web-1 01 Terminating 0 8m web-1 01 Terminating 0 8m
web-1 01 Terminating 0 8m web-1 01 Pending 0 0s web-1 01 Pending 0 0s
web-1 01 ContainerCreating 0 0s web-1 11 Running 0 6s web-0 11
Terminating 0 7m web-0 11 Terminating 0 7m web-0 01 Terminating 0 7m
web-0 01 Terminating 0 7m web-0 01 Terminating 0 7m web-0 01 Terminating
0 7m web-0 01 Pending 0 0s web-0 01 Pending 0 0s web-0 01
ContainerCreating 0 0s web-0 11 Running 0 10s ```

The Pods in the StatefulSet are updated in reverse ordinal order. The
StatefulSet controller terminates each Pod, and waits for it to
transition to Running and Ready prior to updating the next Pod. Note
that, even though the StatefulSet controller will not proceed to update
the next Pod until its ordinal successor is Running and Ready, it will
restore any Pod that fails during the update to that Pods existing
version.

Pods that have already received the update will be restored to the
updated version, and Pods that have not yet received the update will be
restored to the previous version. In this way, the controller attempts
to continue to keep the application healthy and the update consistent in
the presence of intermittent failures.

Get the Pods to view their container images

```shell for p in 0 1 2 do kubectl get pod web-p --template range i,
c .spec.containersc.imageend echo done ``` ```
registry.k8s.ionginx-slim0.24 registry.k8s.ionginx-slim0.24
registry.k8s.ionginx-slim0.24

```

All the Pods in the StatefulSet are now running the previous container
image.

You can also use `kubectl rollout status sts` to view the status of a
rolling update to a StatefulSet

# # # # Staging an update

You can split updates to a StatefulSet that uses the `RollingUpdate`
strategy into _partitions_, by specifying
`.spec.updateStrategy.rollingUpdate.partition`.

For more context, you can read [Partitioned rolling
updates](docsconceptsworkloadscontrollersstatefulset#partitions) in the
StatefulSet concept page.

You can stage an update to a StatefulSet by using the `partition`
field within `.spec.updateStrategy.rollingUpdate`. For this update,
you will keep the existing Pods in the StatefulSet unchanged whilst you
change the pod template for the StatefulSet. Then you - or, outside of a
tutorial, some external automation - can trigger that prepared update.

First, patch the `web` StatefulSet to add a partition to the
`updateStrategy` field

```shell # The value of partition determines which ordinals a change
applies to # Make sure to use a number bigger than the last ordinal for
the # StatefulSet kubectl patch statefulset web -p
specupdateStrategytypeRollingUpdate,rollingUpdatepartition3 ```
``` statefulset.appsweb patched ```

Patch the StatefulSet again to change the container image that this
StatefulSet uses

```shell kubectl patch statefulset web --typejson -p[op replace,
path spectemplatespeccontainers0image,
valueregistry.k8s.ionginx-slim0.21] ``` ``` statefulset.appsweb
patched ```

Delete a Pod in the StatefulSet

```shell kubectl delete pod web-2 ``` ``` pod web-2 deleted
```

Wait for the replacement `web-2` Pod to be Running and Ready

```shell # End the watch when you see that web-2 is healthy kubectl
get pod -l appnginx --watch ``` ``` NAME READY STATUS RESTARTS
AGE web-0 11 Running 0 4m web-1 11 Running 0 4m web-2 01
ContainerCreating 0 11s web-2 11 Running 0 18s ```

Get the Pods container image

```shell kubectl get pod web-2 --template range i, c
.spec.containersc.imageend ``` ``` registry.k8s.ionginx-slim0.24
```

Notice that, even though the update strategy is `RollingUpdate` the
StatefulSet restored the Pod with the original container image. This is
because the ordinal of the Pod is less than the `partition` specified
by the `updateStrategy`.

# # # # Rolling out a canary

Youre now going to try a [canary
rollout](httpsglossary.cncf.iocanary-deployment) of that staged change.

You can roll out a canary (to test the modified template) by
decrementing the `partition` you specified
[above](#staging-an-update).

Patch the StatefulSet to decrement the partition

```shell # The value of partition should match the highest existing
ordinal for # the StatefulSet kubectl patch statefulset web -p
specupdateStrategytypeRollingUpdate,rollingUpdatepartition2 ```
``` statefulset.appsweb patched ```

The control plane triggers replacement for `web-2` (implemented by a
graceful **delete** followed by creating a new Pod once the deletion
is complete). Wait for the new `web-2` Pod to be Running and Ready.

```shell # This should already be running kubectl get pod -l
appnginx --watch ``` ``` NAME READY STATUS RESTARTS AGE web-0 11
Running 0 4m web-1 11 Running 0 4m web-2 01 ContainerCreating 0 11s
web-2 11 Running 0 18s ```

Get the Pods container

```shell kubectl get pod web-2 --template range i, c
.spec.containersc.imageend ``` ``` registry.k8s.ionginx-slim0.21

```

When you changed the `partition`, the StatefulSet controller
automatically updated the `web-2` Pod because the Pods ordinal was
greater than or equal to the `partition`.

Delete the `web-1` Pod

```shell kubectl delete pod web-1 ``` ``` pod web-1 deleted
```

Wait for the `web-1` Pod to be Running and Ready.

```shell # This should already be running kubectl get pod -l
appnginx --watch ``` The output is similar to ``` NAME READY
STATUS RESTARTS AGE web-0 11 Running 0 6m web-1 01 Terminating 0 6m
web-2 11 Running 0 2m web-1 01 Terminating 0 6m web-1 01 Terminating 0
6m web-1 01 Terminating 0 6m web-1 01 Pending 0 0s web-1 01 Pending 0 0s
web-1 01 ContainerCreating 0 0s web-1 11 Running 0 18s ```

Get the `web-1` Pods container image

```shell kubectl get pod web-1 --template range i, c
.spec.containersc.imageend ``` ``` registry.k8s.ionginx-slim0.24
```

`web-1` was restored to its original configuration because the Pods
ordinal was less than the partition. When a partition is specified, all
Pods with an ordinal that is greater than or equal to the partition will
be updated when the StatefulSets `.spec.template` is updated. If a Pod
that has an ordinal less than the partition is deleted or otherwise
terminated, it will be restored to its original configuration.

# # # # Phased roll outs

You can perform a phased roll out (e.g. a linear, geometric, or
exponential roll out) using a partitioned rolling update in a similar
manner to how you rolled out a [canary](#rolling-out-a-canary). To
perform a phased roll out, set the `partition` to the ordinal at which
you want the controller to pause the update.

The partition is currently set to `2`. Set the partition to `0`

```shell kubectl patch statefulset web -p
specupdateStrategytypeRollingUpdate,rollingUpdatepartition0 ```
``` statefulset.appsweb patched ```

Wait for all of the Pods in the StatefulSet to become Running and Ready.

```shell # This should already be running kubectl get pod -l
appnginx --watch ``` The output is similar to ``` NAME READY
STATUS RESTARTS AGE web-0 11 Running 0 3m web-1 01 ContainerCreating 0
11s web-2 11 Running 0 2m web-1 11 Running 0 18s web-0 11 Terminating 0
3m web-0 11 Terminating 0 3m web-0 01 Terminating 0 3m web-0 01
Terminating 0 3m web-0 01 Terminating 0 3m web-0 01 Terminating 0 3m
web-0 01 Pending 0 0s web-0 01 Pending 0 0s web-0 01 ContainerCreating 0
0s web-0 11 Running 0 3s ```

Get the container image details for the Pods in the StatefulSet

```shell for p in 0 1 2 do kubectl get pod web-p --template range i,
c .spec.containersc.imageend echo done ``` ```
registry.k8s.ionginx-slim0.21 registry.k8s.ionginx-slim0.21
registry.k8s.ionginx-slim0.21 ```

By moving the `partition` to `0`, you allowed the StatefulSet to
continue the update process.

# # # OnDelete #on-delete

You select this update strategy for a StatefulSet by setting the
`.spec.template.updateStrategy.type` to `OnDelete`.

Patch the `web` StatefulSet to use the `OnDelete` update strategy

```shell kubectl patch statefulset web -p
specupdateStrategytypeOnDelete, rollingUpdate null ``` ```
statefulset.appsweb patched ```

When you select this update strategy, the StatefulSet controller does
not automatically update Pods when a modification is made to the
StatefulSets `.spec.template` field. You need to manage the rollout
yourself - either manually, or using separate automation.

# # Deleting StatefulSets

StatefulSet supports both _non-cascading_ and _cascading_ deletion.
In a non-cascading **delete**, the StatefulSets Pods are not deleted
when the StatefulSet is deleted. In a cascading **delete**, both the
StatefulSet and its Pods are deleted.

Read [Use Cascading Deletion in a
Cluster](docstasksadminister-clusteruse-cascading-deletion) to learn
about cascading deletion generally.

# # # Non-cascading delete

In one terminal window, watch the Pods in the StatefulSet.

``` # End this watch when there are no Pods for the StatefulSet
kubectl get pods --watch -l appnginx ```

Use [`kubectl
delete`](docsreferencegeneratedkubectlkubectl-commands#delete) to
delete the StatefulSet. Make sure to supply the `--cascadeorphan`
parameter to the command. This parameter tells Kubernetes to only delete
the StatefulSet, and to **not** delete any of its Pods.

```shell kubectl delete statefulset web --cascadeorphan ```
``` statefulset.apps web deleted ```

Get the Pods, to examine their status

```shell kubectl get pods -l appnginx ``` ``` NAME READY STATUS
RESTARTS AGE web-0 11 Running 0 6m web-1 11 Running 0 7m web-2 11
Running 0 5m ```

Even though `web` has been deleted, all of the Pods are still Running
and Ready. Delete `web-0`

```shell kubectl delete pod web-0 ``` ``` pod web-0 deleted
```

Get the StatefulSets Pods

```shell kubectl get pods -l appnginx ``` ``` NAME READY STATUS
RESTARTS AGE web-1 11 Running 0 10m web-2 11 Running 0 7m ```

As the `web` StatefulSet has been deleted, `web-0` has not been
relaunched.

In one terminal, watch the StatefulSets Pods.

```shell # Leave this watch running until the next time you start a
watch kubectl get pods --watch -l appnginx ```

In a second terminal, recreate the StatefulSet. Note that, unless you
deleted the `nginx` Service (which you should not have), you will see
an error indicating that the Service already exists.

```shell kubectl apply -f httpsk8s.ioexamplesapplicationwebweb.yaml
``` ``` statefulset.appsweb created servicenginx unchanged ```

Ignore the error. It only indicates that an attempt was made to create
the _nginx_ headless Service even though that Service already exists.

Examine the output of the `kubectl get` command running in the first
terminal.

```shell # This should already be running kubectl get pods --watch
-l appnginx ``` ``` NAME READY STATUS RESTARTS AGE web-1 11
Running 0 16m web-2 11 Running 0 2m NAME READY STATUS RESTARTS AGE web-0
01 Pending 0 0s web-0 01 Pending 0 0s web-0 01 ContainerCreating 0 0s
web-0 11 Running 0 18s web-2 11 Terminating 0 3m web-2 01 Terminating 0
3m web-2 01 Terminating 0 3m web-2 01 Terminating 0 3m ```

When the `web` StatefulSet was recreated, it first relaunched
`web-0`. Since `web-1` was already Running and Ready, when `web-0`
transitioned to Running and Ready, it adopted this Pod. Since you
recreated the StatefulSet with `replicas` equal to 2, once `web-0`
had been recreated, and once `web-1` had been determined to already be
Running and Ready, `web-2` was terminated.

Now take another look at the contents of the `index.html` file served
by the Pods webservers

```shell for i in 0 1 do kubectl exec -i -t web-i -- curl
httplocalhost done ```

``` web-0 web-1 ```

Even though you deleted both the StatefulSet and the `web-0` Pod, it
still serves the hostname originally entered into its `index.html`
file. This is because the StatefulSet never deletes the
PersistentVolumes associated with a Pod. When you recreated the
StatefulSet and it relaunched `web-0`, its original PersistentVolume
was remounted.

# # # Cascading delete

In one terminal window, watch the Pods in the StatefulSet.

```shell # Leave this running until the next page section kubectl
get pods --watch -l appnginx ```

In another terminal, delete the StatefulSet again. This time, omit the
`--cascadeorphan` parameter.

```shell kubectl delete statefulset web ```

``` statefulset.apps web deleted ```

Examine the output of the `kubectl get` command running in the first
terminal, and wait for all of the Pods to transition to Terminating.

```shell # This should already be running kubectl get pods --watch
-l appnginx ```

``` NAME READY STATUS RESTARTS AGE web-0 11 Running 0 11m web-1 11
Running 0 27m NAME READY STATUS RESTARTS AGE web-0 11 Terminating 0 12m
web-1 11 Terminating 0 29m web-0 01 Terminating 0 12m web-0 01
Terminating 0 12m web-0 01 Terminating 0 12m web-1 01 Terminating 0 29m
web-1 01 Terminating 0 29m web-1 01 Terminating 0 29m

```

As you saw in the [Scaling Down](#scaling-down) section, the Pods are
terminated one at a time, with respect to the reverse order of their
ordinal indices. Before terminating a Pod, the StatefulSet controller
waits for the Pods successor to be completely terminated.

Although a cascading delete removes a StatefulSet together with its
Pods, the cascade does **not** delete the headless Service
associated with the StatefulSet. You must delete the `nginx` Service
manually.

```shell kubectl delete service nginx ```

``` service nginx deleted ```

Recreate the StatefulSet and headless Service one more time

```shell kubectl apply -f httpsk8s.ioexamplesapplicationwebweb.yaml
```

``` servicenginx created statefulset.appsweb created ```

When all of the StatefulSets Pods transition to Running and Ready,
retrieve the contents of their `index.html` files

```shell for i in 0 1 do kubectl exec -i -t web-i -- curl
httplocalhost done ```

``` web-0 web-1 ```

Even though you completely deleted the StatefulSet, and all of its Pods,
the Pods are recreated with their PersistentVolumes mounted, and
`web-0` and `web-1` continue to serve their hostnames.

Finally, delete the `nginx` Service...

```shell kubectl delete service nginx ```

``` service nginx deleted ```

...and the `web` StatefulSet

```shell kubectl delete statefulset web ```

``` statefulset web deleted ```

# # Pod management policy

For some distributed systems, the StatefulSet ordering guarantees are
unnecessary andor undesirable. These systems require only uniqueness and
identity.

You can specify a [Pod management
policy](docsconceptsworkloadscontrollersstatefulset#pod-management-policies)
to avoid this strict ordering either `OrderedReady` (the default), or
`Parallel`.

# # # OrderedReady Pod management

`OrderedReady` pod management is the default for StatefulSets. It
tells the StatefulSet controller to respect the ordering guarantees
demonstrated above.

Use this when your application requires or expects that changes, such as
rolling out a new version of your application, happen in the strict
order of the ordinal (pod number) that the StatefulSet provides. In
other words, if you have Pods `app-0`, `app-1` and `app-2`,
Kubernetes will update `app-0` first and check it. Once the checks are
good, Kubernetes updates `app-1` and finally `app-2`.

If you added two more Pods, Kubernetes would set up `app-3` and wait
for that to become healthy before deploying `app-4`.

Because this is the default setting, youve already practised using it.

# # # Parallel Pod management

The alternative, `Parallel` pod management, tells the StatefulSet
controller to launch or terminate all Pods in parallel, and not to wait
for Pods to become `Running` and `Ready` or completely terminated
prior to launching or terminating another Pod.

The `Parallel` pod management option only affects the behavior for
scaling operations. Updates are not affected Kubernetes still rolls out
changes in order. For this tutorial, the application is very simple a
webserver that tells you its hostname (because this is a StatefulSet,
the hostname for each Pod is different and predictable).

code_sample fileapplicationwebweb-parallel.yaml

This manifest is identical to the one you downloaded above except that
the `.spec.podManagementPolicy` of the `web` StatefulSet is set to
`Parallel`.

In one terminal, watch the Pods in the StatefulSet.

```shell # Leave this watch running until the end of the section
kubectl get pod -l appnginx --watch ```

In another terminal, reconfigure the StatefulSet for `Parallel` Pod
management

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationwebweb-parallel.yaml ``` ```
servicenginx updated statefulset.appsweb updated ```

Keep the terminal open where youre running the watch. In another
terminal window, scale the StatefulSet

```shell kubectl scale statefulsetweb --replicas5 ``` ```
statefulset.appsweb scaled ```

Examine the output of the terminal where the `kubectl get` command is
running. It may look something like

``` web-3 01 Pending 0 0s web-3 01 Pending 0 0s web-3 01 Pending 0 7s
web-3 01 ContainerCreating 0 7s web-2 01 Pending 0 0s web-4 01 Pending 0
0s web-2 11 Running 0 8s web-4 01 ContainerCreating 0 4s web-3 11
Running 0 26s web-4 11 Running 0 2s ```

The StatefulSet launched three new Pods, and it did not wait for the
first to become Running and Ready prior to launching the second and
third Pods.

This approach is useful if your workload has a stateful element, or
needs Pods to be able to identify each other with predictable naming,
and especially if you sometimes need to provide a lot more capacity
quickly. If this simple web service for the tutorial suddenly got an
extra 1,000,000 requests per minute then you would want to run some more
Pods - but you also would not want to wait for each new Pod to launch.
Starting the extra Pods in parallel cuts the time between requesting the
extra capacity and having it available for use.

# # heading cleanup

You should have two terminals open, ready for you to run `kubectl`
commands as part of cleanup.

```shell kubectl delete sts web # sts is an abbreviation for
statefulset ```

You can watch `kubectl get` to see those Pods being deleted.
```shell # end the watch when youve seen what you need to kubectl
get pod -l appnginx --watch ``` ``` web-3 11 Terminating 0 9m
web-2 11 Terminating 0 9m web-3 11 Terminating 0 9m web-2 11 Terminating
0 9m web-1 11 Terminating 0 44m web-0 11 Terminating 0 44m web-0 01
Terminating 0 44m web-3 01 Terminating 0 9m web-2 01 Terminating 0 9m
web-1 01 Terminating 0 44m web-0 01 Terminating 0 44m web-2 01
Terminating 0 9m web-2 01 Terminating 0 9m web-2 01 Terminating 0 9m
web-1 01 Terminating 0 44m web-1 01 Terminating 0 44m web-1 01
Terminating 0 44m web-0 01 Terminating 0 44m web-0 01 Terminating 0 44m
web-0 01 Terminating 0 44m web-3 01 Terminating 0 9m web-3 01
Terminating 0 9m web-3 01 Terminating 0 9m ```

During deletion, a StatefulSet removes all Pods concurrently it does not
wait for a Pods ordinal successor to terminate prior to deleting that
Pod.

Close the terminal where the `kubectl get` command is running and
delete the `nginx` Service

```shell kubectl delete svc nginx ```

Delete the persistent storage media for the PersistentVolumes used in
this tutorial.

```shell kubectl get pvc ``` ``` NAME STATUS VOLUME CAPACITY
ACCESS MODES STORAGECLASS AGE www-web-0 Bound
pvc-2bf00408-d366-4a12-bad0-1869c65d0bee 1Gi RWO standard 25m www-web-1
Bound pvc-ba3bfe9c-413e-4b95-a2c0-3ea8a54dbab4 1Gi RWO standard 24m
www-web-2 Bound pvc-cba6cfa6-3a47-486b-a138-db5930207eaf 1Gi RWO
standard 15m www-web-3 Bound pvc-0c04d7f0-787a-4977-8da3-d9d3a6d8d752
1Gi RWO standard 15m www-web-4 Bound
pvc-b2c73489-e70b-4a4e-9ec1-9eab439aa43e 1Gi RWO standard 14m ```

```shell kubectl get pv ``` ``` NAME CAPACITY ACCESS MODES
RECLAIM POLICY STATUS CLAIM STORAGECLASS REASON AGE
pvc-0c04d7f0-787a-4977-8da3-d9d3a6d8d752 1Gi RWO Delete Bound
defaultwww-web-3 standard 15m pvc-2bf00408-d366-4a12-bad0-1869c65d0bee
1Gi RWO Delete Bound defaultwww-web-0 standard 25m
pvc-b2c73489-e70b-4a4e-9ec1-9eab439aa43e 1Gi RWO Delete Bound
defaultwww-web-4 standard 14m pvc-ba3bfe9c-413e-4b95-a2c0-3ea8a54dbab4
1Gi RWO Delete Bound defaultwww-web-1 standard 24m
pvc-cba6cfa6-3a47-486b-a138-db5930207eaf 1Gi RWO Delete Bound
defaultwww-web-2 standard 15m ```

```shell kubectl delete pvc www-web-0 www-web-1 www-web-2 www-web-3
www-web-4 ```

``` persistentvolumeclaim www-web-0 deleted persistentvolumeclaim
www-web-1 deleted persistentvolumeclaim www-web-2 deleted
persistentvolumeclaim www-web-3 deleted persistentvolumeclaim www-web-4
deleted ```

```shell kubectl get pvc ```

``` No resources found in default namespace. ```

You also need to delete the persistent storage media for the
PersistentVolumes used in this tutorial. Follow the necessary steps,
based on your environment, storage configuration, and provisioning
method, to ensure that all storage is reclaimed.

 FILE datakubernetes
website main content-en_docstutorialsstateful-applicationcassandra.md
 --- title Example
Deploying Cassandra with a StatefulSet reviewers - ahmetb content_type
tutorial weight 30 ---

This tutorial shows you how to run [Apache
Cassandra](httpscassandra.apache.org) on Kubernetes. Cassandra, a
database, needs persistent storage to provide data durability
(application _state_). In this example, a custom Cassandra seed
provider lets the database discover new Cassandra instances as they join
the Cassandra cluster.

*StatefulSets* make it easier to deploy stateful applications into
your Kubernetes cluster. For more information on the features used in
this tutorial, see
[StatefulSet](docsconceptsworkloadscontrollersstatefulset).

Cassandra and Kubernetes both use the term _node_ to mean a member of
a cluster. In this tutorial, the Pods that belong to the StatefulSet are
Cassandra nodes and are members of the Cassandra cluster (called a
_ring_). When those Pods run in your Kubernetes cluster, the
Kubernetes control plane schedules those Pods onto Kubernetes .

When a Cassandra node starts, it uses a _seed list_ to bootstrap
discovery of other nodes in the ring. This tutorial deploys a custom
Cassandra seed provider that lets the database discover new Cassandra
Pods as they appear inside your Kubernetes cluster.

# # heading objectives

* Create and validate a Cassandra headless . * Use a to create a
Cassandra ring. * Validate the StatefulSet. * Modify the StatefulSet.
* Delete the StatefulSet and its .

# # heading prerequisites

To complete this tutorial, you should already have a basic familiarity
with , , and .

# # # Additional Minikube setup instructions

[Minikube](httpsminikube.sigs.k8s.iodocs) defaults to 2048MB of memory
and 2 CPU. Running Minikube with the default resource configuration
results in insufficient resource errors during this tutorial. To avoid
these errors, start Minikube with the following settings

```shell minikube start --memory 5120 --cpus4 ```

# # Creating a headless Service for Cassandra
# creating-a-cassandra-headless-service

In Kubernetes, a describes a set of that perform the same task.

The following Service is used for DNS lookups between Cassandra Pods and
clients within your cluster

code_sample fileapplicationcassandracassandra-service.yaml

Create a Service to track all Cassandra StatefulSet members from the
`cassandra-service.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationcassandracassandra-service.yaml ```

# # # Validating (optional) #validating

Get the Cassandra Service.

```shell kubectl get svc cassandra ```

The response is

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE cassandra ClusterIP
None 9042TCP 45s ```

If you dont see a Service named `cassandra`, that means creation
failed. Read [Debug
Services](docstasksdebugdebug-applicationdebug-service) for help
troubleshooting common issues.

# # Using a StatefulSet to create a Cassandra ring

The StatefulSet manifest, included below, creates a Cassandra ring that
consists of three Pods.

This example uses the default provisioner for Minikube. Please update
the following StatefulSet for the cloud you are working with.

code_sample fileapplicationcassandracassandra-statefulset.yaml

Create the Cassandra StatefulSet from the `cassandra-statefulset.yaml`
file

```shell # Use this if you are able to apply
cassandra-statefulset.yaml unmodified kubectl apply -f
httpsk8s.ioexamplesapplicationcassandracassandra-statefulset.yaml ```

If you need to modify `cassandra-statefulset.yaml` to suit your
cluster, download
httpsk8s.ioexamplesapplicationcassandracassandra-statefulset.yaml and
then apply that manifest, from the folder you saved the modified version
into ```shell # Use this if you needed to modify
cassandra-statefulset.yaml locally kubectl apply -f
cassandra-statefulset.yaml ```

# # Validating the Cassandra StatefulSet

1. Get the Cassandra StatefulSet

```shell kubectl get statefulset cassandra ```

The response should be similar to

``` NAME DESIRED CURRENT AGE cassandra 3 0 13s ```

The `StatefulSet` resource deploys Pods sequentially.

1. Get the Pods to see the ordered creation status

```shell kubectl get pods -lappcassandra ```

The response should be similar to

```shell NAME READY STATUS RESTARTS AGE cassandra-0 11 Running 0 1m
cassandra-1 01 ContainerCreating 0 8s ```

It can take several minutes for all three Pods to deploy. Once they are
deployed, the same command returns output similar to

``` NAME READY STATUS RESTARTS AGE cassandra-0 11 Running 0 10m
cassandra-1 11 Running 0 9m cassandra-2 11 Running 0 8m ```

3. Run the Cassandra
[nodetool](httpscwiki.apache.orgconfluencedisplayCASSANDRA2NodeTool)
inside the first Pod, to display the status of the ring.

```shell kubectl exec -it cassandra-0 -- nodetool status ```

The response should look something like

``` Datacenter DC1-K8Demo

StatusUpDown StateNormalLeavingJoiningMoving -- Address Load Tokens
Owns (effective) Host ID Rack UN 172.17.0.5 83.57 KiB 32 74.0
e2dd09e6-d9d3-477e-96c5-45094c08db0f Rack1-K8Demo UN 172.17.0.4 101.04
KiB 32 58.8 f89d6835-3a42-4419-92b3-0e62cae1479c Rack1-K8Demo UN
172.17.0.6 84.74 KiB 32 67.1 a6a1e8c2-3dc5-4417-b1a0-26507af2aaad
Rack1-K8Demo ```

# # Modifying the Cassandra StatefulSet

Use `kubectl edit` to modify the size of a Cassandra StatefulSet.

1. Run the following command

```shell kubectl edit statefulset cassandra ```

This command opens an editor in your terminal. The line you need to
change is the `replicas` field. The following sample is an excerpt of
the StatefulSet file

```yaml # Please edit the object below. Lines beginning with a #
will be ignored, # and an empty file will abort the edit. If an error
occurs while saving this file will be # reopened with the relevant
failures. # apiVersion appsv1 kind StatefulSet metadata
creationTimestamp 2016-08-13T184058Z generation 1 labels app cassandra
name cassandra namespace default resourceVersion 323 uid
7a219483-6185-11e6-a910-42010a8a0fc0 spec replicas 3 ```

1. Change the number of replicas to 4, and then save the manifest.

The StatefulSet now scales to run with 4 Pods.

1. Get the Cassandra StatefulSet to verify your change

```shell kubectl get statefulset cassandra ```

The response should be similar to

``` NAME DESIRED CURRENT AGE cassandra 4 4 36m ```

# # heading cleanup

Deleting or scaling a StatefulSet down does not delete the volumes
associated with the StatefulSet. This setting is for your safety because
your data is more valuable than automatically purging all related
StatefulSet resources.

Depending on the storage class and reclaim policy, deleting the
*PersistentVolumeClaims* may cause the associated volumes to also be
deleted. Never assume youll be able to access data if its volume claims
are deleted.

1. Run the following commands (chained together into a single command)
to delete everything in the Cassandra StatefulSet

```shell grace(kubectl get pod cassandra-0
-ojsonpath.spec.terminationGracePeriodSeconds) kubectl delete
statefulset -l appcassandra echo Sleeping grace seconds 12 sleep grace
kubectl delete persistentvolumeclaim -l appcassandra ```

1. Run the following command to delete the Service you set up for
Cassandra

```shell kubectl delete service -l appcassandra ```

# # Cassandra container environment variables

The Pods in this tutorial use the
[`gcr.iogoogle-samplescassandrav13`](httpsgithub.comkubernetesexamplesblobmastercassandraimageDockerfile)
image from Googles [container
registry](httpscloud.google.comcontainer-registrydocs). The Docker
image above is based on
[debian-base](httpsgithub.comkubernetesreleasetreemasterimagesbuilddebian-base)
and includes OpenJDK 8.

This image includes a standard Cassandra installation from the Apache
Debian repo. By using environment variables you can change values that
are inserted into `cassandra.yaml`.

Environment variable Default value
------------------------
--------------- `CASSANDRA_CLUSTER_NAME` `Test
Cluster` `CASSANDRA_NUM_TOKENS` `32` `CASSANDRA_RPC_ADDRESS`
`0.0.0.0`

# # heading whatsnext

* Learn how to [Scale a
StatefulSet](docstasksrun-applicationscale-stateful-set). * Learn more
about the
[*KubernetesSeedProvider*](httpsgithub.comkubernetesexamplesblobmastercassandrajavasrcmainjavaiok8scassandraKubernetesSeedProvider.java)
* See more custom [Seed Provider
Configurations](httpsgit.k8s.ioexamplescassandrajavaREADME.md)

 FILE datakubernetes
website main
content-en_docstutorialsstateful-applicationmysql-wordpress-persistent-volume.md
 --- title Example
Deploying WordPress and MySQL with Persistent Volumes reviewers - ahmetb
content_type tutorial weight 20 card name tutorials weight 40 title
Stateful Example Wordpress with Persistent Volumes ---

This tutorial shows you how to deploy a WordPress site and a MySQL
database using Minikube. Both applications use PersistentVolumes and
PersistentVolumeClaims to store data.

A [PersistentVolume](docsconceptsstoragepersistent-volumes) (PV) is a
piece of storage in the cluster that has been manually provisioned by an
administrator, or dynamically provisioned by Kubernetes using a
[StorageClass](docsconceptsstoragestorage-classes). A
[PersistentVolumeClaim](docsconceptsstoragepersistent-volumes#persistentvolumeclaims)
(PVC) is a request for storage by a user that can be fulfilled by a PV.
PersistentVolumes and PersistentVolumeClaims are independent from Pod
lifecycles and preserve data through restarting, rescheduling, and even
deleting Pods.

This deployment is not suitable for production use cases, as it uses
single instance WordPress and MySQL Pods. Consider using [WordPress
Helm Chart](httpsgithub.combitnamichartstreemasterbitnamiwordpress) to
deploy WordPress in production.

The files provided in this tutorial are using GA Deployment APIs and are
specific to kubernetes version 1.9 and later. If you wish to use this
tutorial with an earlier version of Kubernetes, please update the API
version appropriately, or reference earlier versions of this tutorial.

# # heading objectives

* Create PersistentVolumeClaims and PersistentVolumes * Create a
`kustomization.yaml` with * a Secret generator * MySQL resource
configs * WordPress resource configs * Apply the kustomization
directory by `kubectl apply -k .` * Clean up

# # heading prerequisites

The example shown on this page works with `kubectl` 1.27 and above.

Download the following configuration files

1.
[mysql-deployment.yaml](examplesapplicationwordpressmysql-deployment.yaml)

1.
[wordpress-deployment.yaml](examplesapplicationwordpresswordpress-deployment.yaml)

# # Create PersistentVolumeClaims and PersistentVolumes

MySQL and Wordpress each require a PersistentVolume to store data. Their
PersistentVolumeClaims will be created at the deployment step.

Many cluster environments have a default StorageClass installed. When a
StorageClass is not specified in the PersistentVolumeClaim, the clusters
default StorageClass is used instead.

When a PersistentVolumeClaim is created, a PersistentVolume is
dynamically provisioned based on the StorageClass configuration.

In local clusters, the default StorageClass uses the `hostPath`
provisioner. `hostPath` volumes are only suitable for development and
testing. With `hostPath` volumes, your data lives in `tmp` on the
node the Pod is scheduled onto and does not move between nodes. If a Pod
dies and gets scheduled to another node in the cluster, or the node is
rebooted, the data is lost.

If you are bringing up a cluster that needs to use the `hostPath`
provisioner, the `--enable-hostpath-provisioner` flag must be set in
the `controller-manager` component.

If you have a Kubernetes cluster running on Google Kubernetes Engine,
please follow [this
guide](httpscloud.google.comkubernetes-enginedocstutorialspersistent-disk).

# # Create a kustomization.yaml

# # # Add a Secret generator

A [Secret](docsconceptsconfigurationsecret) is an object that stores a
piece of sensitive data like a password or key. Since 1.14, `kubectl`
supports the management of Kubernetes objects using a kustomization
file. You can create a Secret by generators in `kustomization.yaml`.

Add a Secret generator in `kustomization.yaml` from the following
command. You will need to replace `YOUR_PASSWORD` with the password
you want to use.

```shell cat .kustomization.yaml secretGenerator - name mysql-pass
literals  - passwordYOUR_PASSWORD EOF ```

# # Add resource configs for MySQL and WordPress

The following manifest describes a single-instance MySQL Deployment. The
MySQL container mounts the PersistentVolume at varlibmysql. The
`MYSQL_ROOT_PASSWORD` environment variable sets the database password
from the Secret.

code_sample fileapplicationwordpressmysql-deployment.yaml

The following manifest describes a single-instance WordPress Deployment.
The WordPress container mounts the PersistentVolume at `varwwwhtml`
for website data files. The `WORDPRESS_DB_HOST` environment variable
sets the name of the MySQL Service defined above, and WordPress will
access the database by Service. The `WORDPRESS_DB_PASSWORD`
environment variable sets the database password from the Secret
kustomize generated.

code_sample fileapplicationwordpresswordpress-deployment.yaml

1. Download the MySQL deployment configuration file.

```shell curl -LO
httpsk8s.ioexamplesapplicationwordpressmysql-deployment.yaml ```

2. Download the WordPress configuration file.

```shell curl -LO
httpsk8s.ioexamplesapplicationwordpresswordpress-deployment.yaml ```

3. Add them to `kustomization.yaml` file.

```shell cat .kustomization.yaml resources  - mysql-deployment.yaml
 - wordpress-deployment.yaml EOF ```

# # Apply and Verify

The `kustomization.yaml` contains all the resources for deploying a
WordPress site and a MySQL database. You can apply the directory by

```shell kubectl apply -k . ```

Now you can verify that all objects exist.

1. Verify that the Secret exists by running the following command

```shell kubectl get secrets ```

The response should be like this

``` NAME TYPE DATA AGE mysql-pass-c57bb4t7mf Opaque 1 9s ```

2. Verify that a PersistentVolume got dynamically provisioned.

```shell kubectl get pvc ```

It can take up to a few minutes for the PVs to be provisioned and bound.

The response should be like this

``` NAME STATUS VOLUME CAPACITY ACCESS MODES STORAGECLASS AGE
mysql-pv-claim Bound pvc-8cbd7b2e-4044-11e9-b2bb-42010a800002 20Gi RWO
standard 77s wp-pv-claim Bound pvc-8cd0df54-4044-11e9-b2bb-42010a800002
20Gi RWO standard 77s ```

3. Verify that the Pod is running by running the following command

```shell kubectl get pods ```

It can take up to a few minutes for the Pods Status to be `RUNNING`.

The response should be like this

``` NAME READY STATUS RESTARTS AGE wordpress-mysql-1894417608-x5dzt
11 Running 0 40s ```

4. Verify that the Service is running by running the following command

```shell kubectl get services wordpress ```

The response should be like this

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE wordpress
LoadBalancer 10.0.0.89 8032406TCP 4m ```

Minikube can only expose Services through `NodePort`. The EXTERNAL-IP
is always pending.

5. Run the following command to get the IP Address for the WordPress
Service

```shell minikube service wordpress --url ```

The response should be like this

``` http1.2.3.432406 ```

6. Copy the IP address, and load the page in your browser to view your
site.

You should see the WordPress set up page similar to the following
screenshot.

![wordpress-init](httpsraw.githubusercontent.comkubernetesexamplesmastermysql-wordpress-pdWordPress.png)

Do not leave your WordPress installation on this page. If another user
finds it, they can set up a website on your instance and use it to serve
malicious content. Either install WordPress by creating a username and
password or delete your instance.

# # heading cleanup

1. Run the following command to delete your Secret, Deployments,
Services and PersistentVolumeClaims

```shell kubectl delete -k . ```

# # heading whatsnext

* Learn more about [Introspection and
Debugging](docstasksdebugdebug-applicationdebug-running-pod) * Learn
more about [Jobs](docsconceptsworkloadscontrollersjob) * Learn more
about [Port
Forwarding](docstasksaccess-application-clusterport-forward-access-application-cluster)
* Learn how to [Get a Shell to a
Container](docstasksdebugdebug-applicationget-shell-running-container)

 FILE datakubernetes
website main content-en_docstutorialsstateful-applicationzookeeper.md
 --- reviewers -
bprashanth - enisoc - erictune - foxish - janetkuo - kow3ns -
smarterclayton title Running ZooKeeper, A Distributed System Coordinator
content_type tutorial weight 40 ---

This tutorial demonstrates running [Apache
Zookeeper](httpszookeeper.apache.org) on Kubernetes using
[StatefulSets](docsconceptsworkloadscontrollersstatefulset),
[PodDisruptionBudgets](docsconceptsworkloadspodsdisruptions#pod-disruption-budget),
and
[PodAntiAffinity](docsconceptsscheduling-evictionassign-pod-node#affinity-and-anti-affinity).

# # heading prerequisites

Before starting this tutorial, you should be familiar with the following
Kubernetes concepts

- [Pods](docsconceptsworkloadspods) - [Cluster
DNS](docsconceptsservices-networkingdns-pod-service) - [Headless
Services](docsconceptsservices-networkingservice#headless-services) -
[PersistentVolumes](docsconceptsstoragepersistent-volumes) -
[PersistentVolume
Provisioning](httpsgithub.comkubernetesexamplestreemasterstagingpersistent-volume-provisioning) -
[StatefulSets](docsconceptsworkloadscontrollersstatefulset) -
[PodDisruptionBudgets](docsconceptsworkloadspodsdisruptions#pod-disruption-budget) -
[PodAntiAffinity](docsconceptsscheduling-evictionassign-pod-node#affinity-and-anti-affinity) -
[kubectl CLI](docsreferencekubectlkubectl)

You must have a cluster with at least four nodes, and each node requires
at least 2 CPUs and 4 GiB of memory. In this tutorial you will cordon
and drain the clusters nodes. **This means that the cluster will
terminate and evict all Pods on its nodes, and the nodes will
temporarily become unschedulable.** You should use a dedicated cluster
for this tutorial, or you should ensure that the disruption you cause
will not interfere with other tenants.

This tutorial assumes that you have configured your cluster to
dynamically provision PersistentVolumes. If your cluster is not
configured to do so, you will have to manually provision three 20 GiB
volumes before starting this tutorial.

# # heading objectives

After this tutorial, you will know the following.

- How to deploy a ZooKeeper ensemble using StatefulSet. - How to
consistently configure the ensemble. - How to spread the deployment of
ZooKeeper servers in the ensemble. - How to use PodDisruptionBudgets to
ensure service availability during planned maintenance.

# # # ZooKeeper

[Apache ZooKeeper](httpszookeeper.apache.orgdoccurrent) is a
distributed, open-source coordination service for distributed
applications. ZooKeeper allows you to read, write, and observe updates
to data. Data are organized in a file system like hierarchy and
replicated to all ZooKeeper servers in the ensemble (a set of ZooKeeper
servers). All operations on data are atomic and sequentially consistent.
ZooKeeper ensures this by using the
[Zab](httpspdfs.semanticscholar.orgb02c6b00bd5dbdbd951fddb00b906c82fa80f0b3.pdf)
consensus protocol to replicate a state machine across all servers in
the ensemble.

The ensemble uses the Zab protocol to elect a leader, and the ensemble
cannot write data until that election is complete. Once complete, the
ensemble uses Zab to ensure that it replicates all writes to a quorum
before it acknowledges and makes them visible to clients. Without
respect to weighted quorums, a quorum is a majority component of the
ensemble containing the current leader. For instance, if the ensemble
has three servers, a component that contains the leader and one other
server constitutes a quorum. If the ensemble can not achieve a quorum,
the ensemble cannot write data.

ZooKeeper servers keep their entire state machine in memory, and write
every mutation to a durable WAL (Write Ahead Log) on storage media. When
a server crashes, it can recover its previous state by replaying the
WAL. To prevent the WAL from growing without bound, ZooKeeper servers
will periodically snapshot them in memory state to storage media. These
snapshots can be loaded directly into memory, and all WAL entries that
preceded the snapshot may be discarded.

# # Creating a ZooKeeper ensemble

The manifest below contains a [Headless
Service](docsconceptsservices-networkingservice#headless-services), a
[Service](docsconceptsservices-networkingservice), a
[PodDisruptionBudget](docsconceptsworkloadspodsdisruptions#pod-disruption-budgets),
and a [StatefulSet](docsconceptsworkloadscontrollersstatefulset).

code_sample fileapplicationzookeeperzookeeper.yaml

Open a terminal, and use the [`kubectl
apply`](docsreferencegeneratedkubectlkubectl-commands#apply) command
to create the manifest.

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationzookeeperzookeeper.yaml ```

This creates the `zk-hs` Headless Service, the `zk-cs` Service, the
`zk-pdb` PodDisruptionBudget, and the `zk` StatefulSet.

``` servicezk-hs created servicezk-cs created
poddisruptionbudget.policyzk-pdb created statefulset.appszk created
```

Use [`kubectl
get`](docsreferencegeneratedkubectlkubectl-commands#get) to watch the
StatefulSet controller create the StatefulSets Pods.

```shell kubectl get pods -w -l appzk ```

Once the `zk-2` Pod is Running and Ready, use `CTRL-C` to terminate
kubectl.

``` NAME READY STATUS RESTARTS AGE zk-0 01 Pending 0 0s zk-0 01
Pending 0 0s zk-0 01 ContainerCreating 0 0s zk-0 01 Running 0 19s zk-0
11 Running 0 40s zk-1 01 Pending 0 0s zk-1 01 Pending 0 0s zk-1 01
ContainerCreating 0 0s zk-1 01 Running 0 18s zk-1 11 Running 0 40s zk-2
01 Pending 0 0s zk-2 01 Pending 0 0s zk-2 01 ContainerCreating 0 0s zk-2
01 Running 0 19s zk-2 11 Running 0 40s ```

The StatefulSet controller creates three Pods, and each Pod has a
container with a
[ZooKeeper](httpsarchive.apache.orgdistzookeeperstable) server.

# # # Facilitating leader election

Because there is no terminating algorithm for electing a leader in an
anonymous network, Zab requires explicit membership configuration to
perform leader election. Each server in the ensemble needs to have a
unique identifier, all servers need to know the global set of
identifiers, and each identifier needs to be associated with a network
address.

Use [`kubectl
exec`](docsreferencegeneratedkubectlkubectl-commands#exec) to get the
hostnames of the Pods in the `zk` StatefulSet.

```shell for i in 0 1 2 do kubectl exec zk-i -- hostname done ```

The StatefulSet controller provides each Pod with a unique hostname
based on its ordinal index. The hostnames take the form of `-`.
Because the `replicas` field of the `zk` StatefulSet is set to
`3`, the Sets controller creates three Pods with their hostnames set
to `zk-0`, `zk-1`, and `zk-2`.

``` zk-0 zk-1 zk-2 ```

The servers in a ZooKeeper ensemble use natural numbers as unique
identifiers, and store each servers identifier in a file called `myid`
in the servers data directory.

To examine the contents of the `myid` file for each server use the
following command.

```shell for i in 0 1 2 do echo myid zk-ikubectl exec zk-i -- cat
varlibzookeeperdatamyid done ```

Because the identifiers are natural numbers and the ordinal indices are
non-negative integers, you can generate an identifier by adding 1 to the
ordinal.

``` myid zk-0 1 myid zk-1 2 myid zk-2 3 ```

To get the Fully Qualified Domain Name (FQDN) of each Pod in the `zk`
StatefulSet use the following command.

```shell for i in 0 1 2 do kubectl exec zk-i -- hostname -f done
```

The `zk-hs` Service creates a domain for all of the Pods,
`zk-hs.default.svc.cluster.local`.

``` zk-0.zk-hs.default.svc.cluster.local
zk-1.zk-hs.default.svc.cluster.local
zk-2.zk-hs.default.svc.cluster.local ```

The A records in [Kubernetes
DNS](docsconceptsservices-networkingdns-pod-service) resolve the FQDNs
to the Pods IP addresses. If Kubernetes reschedules the Pods, it will
update the A records with the Pods new IP addresses, but the A records
names will not change.

ZooKeeper stores its application configuration in a file named
`zoo.cfg`. Use `kubectl exec` to view the contents of the
`zoo.cfg` file in the `zk-0` Pod.

```shell kubectl exec zk-0 -- cat optzookeeperconfzoo.cfg ```

In the `server.1`, `server.2`, and `server.3` properties at the
bottom of the file, the `1`, `2`, and `3` correspond to the
identifiers in the ZooKeeper servers `myid` files. They are set to the
FQDNs for the Pods in the `zk` StatefulSet.

``` clientPort2181 dataDirvarlibzookeeperdata
dataLogDirvarlibzookeeperlog tickTime2000 initLimit10 syncLimit2000
maxClientCnxns60 minSessionTimeout 4000 maxSessionTimeout 40000
autopurge.snapRetainCount3 autopurge.purgeInterval0
server.1zk-0.zk-hs.default.svc.cluster.local28883888
server.2zk-1.zk-hs.default.svc.cluster.local28883888
server.3zk-2.zk-hs.default.svc.cluster.local28883888 ```

# # # Achieving consensus

Consensus protocols require that the identifiers of each participant be
unique. No two participants in the Zab protocol should claim the same
unique identifier. This is necessary to allow the processes in the
system to agree on which processes have committed which data. If two
Pods are launched with the same ordinal, two ZooKeeper servers would
both identify themselves as the same server.

```shell kubectl get pods -w -l appzk ```

``` NAME READY STATUS RESTARTS AGE zk-0 01 Pending 0 0s zk-0 01
Pending 0 0s zk-0 01 ContainerCreating 0 0s zk-0 01 Running 0 19s zk-0
11 Running 0 40s zk-1 01 Pending 0 0s zk-1 01 Pending 0 0s zk-1 01
ContainerCreating 0 0s zk-1 01 Running 0 18s zk-1 11 Running 0 40s zk-2
01 Pending 0 0s zk-2 01 Pending 0 0s zk-2 01 ContainerCreating 0 0s zk-2
01 Running 0 19s zk-2 11 Running 0 40s ```

The A records for each Pod are entered when the Pod becomes Ready.
Therefore, the FQDNs of the ZooKeeper servers will resolve to a single
endpoint, and that endpoint will be the unique ZooKeeper server claiming
the identity configured in its `myid` file.

``` zk-0.zk-hs.default.svc.cluster.local
zk-1.zk-hs.default.svc.cluster.local
zk-2.zk-hs.default.svc.cluster.local ```

This ensures that the `servers` properties in the ZooKeepers
`zoo.cfg` files represents a correctly configured ensemble.

``` server.1zk-0.zk-hs.default.svc.cluster.local28883888
server.2zk-1.zk-hs.default.svc.cluster.local28883888
server.3zk-2.zk-hs.default.svc.cluster.local28883888 ```

When the servers use the Zab protocol to attempt to commit a value, they
will either achieve consensus and commit the value (if leader election
has succeeded and at least two of the Pods are Running and Ready), or
they will fail to do so (if either of the conditions are not met). No
state will arise where one server acknowledges a write on behalf of
another.

# # # Sanity testing the ensemble

The most basic sanity test is to write data to one ZooKeeper server and
to read the data from another.

The command below executes the `zkCli.sh` script to write `world` to
the path `hello` on the `zk-0` Pod in the ensemble.

```shell kubectl exec zk-0 -- zkCli.sh create hello world ```

``` WATCHER

WatchedEvent stateSyncConnected typeNone pathnull Created hello ```

To get the data from the `zk-1` Pod use the following command.

```shell kubectl exec zk-1 -- zkCli.sh get hello ```

The data that you created on `zk-0` is available on all the servers in
the ensemble.

``` WATCHER

WatchedEvent stateSyncConnected typeNone pathnull world cZxid
0x100000002 ctime Thu Dec 08 151330 UTC 2016 mZxid 0x100000002 mtime Thu
Dec 08 151330 UTC 2016 pZxid 0x100000002 cversion 0 dataVersion 0
aclVersion 0 ephemeralOwner 0x0 dataLength 5 numChildren 0 ```

# # # Providing durable storage

As mentioned in the [ZooKeeper Basics](#zookeeper) section, ZooKeeper
commits all entries to a durable WAL, and periodically writes snapshots
in memory state, to storage media. Using WALs to provide durability is a
common technique for applications that use consensus protocols to
achieve a replicated state machine.

Use the [`kubectl
delete`](docsreferencegeneratedkubectlkubectl-commands#delete) command
to delete the `zk` StatefulSet.

```shell kubectl delete statefulset zk ```

``` statefulset.apps zk deleted ```

Watch the termination of the Pods in the StatefulSet.

```shell kubectl get pods -w -l appzk ```

When `zk-0` if fully terminated, use `CTRL-C` to terminate kubectl.

``` zk-2 11 Terminating 0 9m zk-0 11 Terminating 0 11m zk-1 11
Terminating 0 10m zk-2 01 Terminating 0 9m zk-2 01 Terminating 0 9m zk-2
01 Terminating 0 9m zk-1 01 Terminating 0 10m zk-1 01 Terminating 0 10m
zk-1 01 Terminating 0 10m zk-0 01 Terminating 0 11m zk-0 01 Terminating
0 11m zk-0 01 Terminating 0 11m ```

Reapply the manifest in `zookeeper.yaml`.

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationzookeeperzookeeper.yaml ```

This creates the `zk` StatefulSet object, but the other API objects in
the manifest are not modified because they already exist.

Watch the StatefulSet controller recreate the StatefulSets Pods.

```shell kubectl get pods -w -l appzk ```

Once the `zk-2` Pod is Running and Ready, use `CTRL-C` to terminate
kubectl.

``` NAME READY STATUS RESTARTS AGE zk-0 01 Pending 0 0s zk-0 01
Pending 0 0s zk-0 01 ContainerCreating 0 0s zk-0 01 Running 0 19s zk-0
11 Running 0 40s zk-1 01 Pending 0 0s zk-1 01 Pending 0 0s zk-1 01
ContainerCreating 0 0s zk-1 01 Running 0 18s zk-1 11 Running 0 40s zk-2
01 Pending 0 0s zk-2 01 Pending 0 0s zk-2 01 ContainerCreating 0 0s zk-2
01 Running 0 19s zk-2 11 Running 0 40s ```

Use the command below to get the value you entered during the [sanity
test](#sanity-testing-the-ensemble), from the `zk-2` Pod.

```shell kubectl exec zk-2 zkCli.sh get hello ```

Even though you terminated and recreated all of the Pods in the `zk`
StatefulSet, the ensemble still serves the original value.

``` WATCHER

WatchedEvent stateSyncConnected typeNone pathnull world cZxid
0x100000002 ctime Thu Dec 08 151330 UTC 2016 mZxid 0x100000002 mtime Thu
Dec 08 151330 UTC 2016 pZxid 0x100000002 cversion 0 dataVersion 0
aclVersion 0 ephemeralOwner 0x0 dataLength 5 numChildren 0 ```

The `volumeClaimTemplates` field of the `zk` StatefulSets `spec`
specifies a PersistentVolume provisioned for each Pod.

```yaml volumeClaimTemplates  - metadata name datadir annotations
volume.alpha.kubernetes.iostorage-class anything spec accessModes [
ReadWriteOnce ] resources requests storage 20Gi ```

The `StatefulSet` controller generates a `PersistentVolumeClaim` for
each Pod in the `StatefulSet`.

Use the following command to get the `StatefulSet`s
`PersistentVolumeClaims`.

```shell kubectl get pvc -l appzk ```

When the `StatefulSet` recreated its Pods, it remounts the Pods
PersistentVolumes.

``` NAME STATUS VOLUME CAPACITY ACCESSMODES AGE datadir-zk-0 Bound
pvc-bed742cd-bcb1-11e6-994f-42010a800002 20Gi RWO 1h datadir-zk-1 Bound
pvc-bedd27d2-bcb1-11e6-994f-42010a800002 20Gi RWO 1h datadir-zk-2 Bound
pvc-bee0817e-bcb1-11e6-994f-42010a800002 20Gi RWO 1h ```

The `volumeMounts` section of the `StatefulSet`s container
`template` mounts the PersistentVolumes in the ZooKeeper servers data
directories.

```yaml volumeMounts - name datadir mountPath varlibzookeeper ```

When a Pod in the `zk` `StatefulSet` is (re)scheduled, it will
always have the same `PersistentVolume` mounted to the ZooKeeper
servers data directory. Even when the Pods are rescheduled, all the
writes made to the ZooKeeper servers WALs, and all their snapshots,
remain durable.

# # Ensuring consistent configuration

As noted in the [Facilitating Leader
Election](#facilitating-leader-election) and [Achieving
Consensus](#achieving-consensus) sections, the servers in a ZooKeeper
ensemble require consistent configuration to elect a leader and form a
quorum. They also require consistent configuration of the Zab protocol
in order for the protocol to work correctly over a network. In our
example we achieve consistent configuration by embedding the
configuration directly into the manifest.

Get the `zk` StatefulSet.

```shell kubectl get sts zk -o yaml ```

```

command  - sh  - -c  - start-zookeeper --servers3
--data_dirvarlibzookeeperdata --data_log_dirvarlibzookeeperdatalog
--conf_diroptzookeeperconf --client_port2181 --election_port3888
--server_port2888 --tick_time2000 --init_limit10 --sync_limit5
--heap512M --max_client_cnxns60 --snap_retain_count3
--purge_interval12 --max_session_timeout40000
--min_session_timeout4000 --log_levelINFO

```

The command used to start the ZooKeeper servers passed the configuration
as command line parameter. You can also use environment variables to
pass configuration to the ensemble.

# # # Configuring logging

One of the files generated by the `zkGenConfig.sh` script controls
ZooKeepers logging. ZooKeeper uses
[Log4j](httpslogging.apache.orglog4j2.x), and, by default, it uses a
time and size based rolling file appender for its logging configuration.

Use the command below to get the logging configuration from one of Pods
in the `zk` `StatefulSet`.

```shell kubectl exec zk-0 cat usretczookeeperlog4j.properties ```

The logging configuration below will cause the ZooKeeper process to
write all of its logs to the standard output file stream.

``` zookeeper.root.loggerCONSOLE zookeeper.console.thresholdINFO
log4j.rootLoggerzookeeper.root.logger
log4j.appender.CONSOLEorg.apache.log4j.ConsoleAppender
log4j.appender.CONSOLE.Thresholdzookeeper.console.threshold
log4j.appender.CONSOLE.layoutorg.apache.log4j.PatternLayout
log4j.appender.CONSOLE.layout.ConversionPatterndISO8601 [myidXmyid] -
-5p [tC1L] - mn ```

This is the simplest possible way to safely log inside the container.
Because the applications write logs to standard out, Kubernetes will
handle log rotation for you. Kubernetes also implements a sane retention
policy that ensures application logs written to standard out and
standard error do not exhaust local storage media.

Use [`kubectl
logs`](docsreferencegeneratedkubectlkubectl-commands#logs) to retrieve
the last 20 log lines from one of the Pods.

```shell kubectl logs zk-0 --tail 20 ```

You can view application logs written to standard out or standard error
using `kubectl logs` and from the Kubernetes Dashboard.

``` 2016-12-06 193416,236 [myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152740 2016-12-06 193416,237 [myid1] - INFO
[Thread-1136NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152740 (no session established for client) 2016-12-06 193426,155
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152749 2016-12-06 193426,155
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152749 2016-12-06 193426,156 [myid1] - INFO
[Thread-1137NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152749 (no session established for client) 2016-12-06 193426,222
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152750 2016-12-06 193426,222
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152750 2016-12-06 193426,226 [myid1] - INFO
[Thread-1138NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152750 (no session established for client) 2016-12-06 193436,151
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152760 2016-12-06 193436,152
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152760 2016-12-06 193436,152 [myid1] - INFO
[Thread-1139NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152760 (no session established for client) 2016-12-06 193436,230
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152761 2016-12-06 193436,231
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152761 2016-12-06 193436,231 [myid1] - INFO
[Thread-1140NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152761 (no session established for client) 2016-12-06 193446,149
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152767 2016-12-06 193446,149
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152767 2016-12-06 193446,149 [myid1] - INFO
[Thread-1141NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152767 (no session established for client) 2016-12-06 193446,230
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxnFactory192] -
Accepted socket connection from 127.0.0.152768 2016-12-06 193446,230
[myid1] - INFO
[NIOServerCxn.Factory0.0.0.00.0.0.02181NIOServerCnxn827] - Processing
ruok command from 127.0.0.152768 2016-12-06 193446,230 [myid1] - INFO
[Thread-1142NIOServerCnxn1008] - Closed socket connection for client
127.0.0.152768 (no session established for client) ```

Kubernetes integrates with many logging solutions. You can choose a
logging solution that best fits your cluster and applications. For
cluster-level logging and aggregation, consider deploying a [sidecar
container](docsconceptscluster-administrationlogging#sidecar-container-with-logging-agent)
to rotate and ship your logs.

# # # Configuring a non-privileged user

The best practices to allow an application to run as a privileged user
inside of a container are a matter of debate. If your organization
requires that applications run as a non-privileged user you can use a
[SecurityContext](docstasksconfigure-pod-containersecurity-context) to
control the user that the entry point runs as.

The `zk` `StatefulSet`s Pod `template` contains a
`SecurityContext`.

```yaml securityContext runAsUser 1000 fsGroup 1000 ```

In the Pods containers, UID 1000 corresponds to the zookeeper user and
GID 1000 corresponds to the zookeeper group.

Get the ZooKeeper process information from the `zk-0` Pod.

```shell kubectl exec zk-0 -- ps -elf ```

As the `runAsUser` field of the `securityContext` object is set to
1000, instead of running as root, the ZooKeeper process runs as the
zookeeper user.

``` F S UID PID PPID C PRI NI ADDR SZ WCHAN STIME TTY TIME CMD 4 S
zookeep 1 0 0 80 0 - 1127 - 2046 000000 sh -c zkGenConfig.sh zkServer.sh
start-foreground 0 S zookeep 27 1 0 80 0 - 1155556 - 2046 000019
usrlibjvmjava-8-openjdk-amd64binjava -Dzookeeper.log.dirvarlogzookeeper
-Dzookeeper.root.loggerINFO,CONSOLE -cp
usrbin..buildclassesusrbin..buildlib*.jarusrbin..sharezookeeperzookeeper-3.4.9.jarusrbin..sharezookeeperslf4j-log4j12-1.6.1.jarusrbin..sharezookeeperslf4j-api-1.6.1.jarusrbin..sharezookeepernetty-3.10.5.Final.jarusrbin..sharezookeeperlog4j-1.2.16.jarusrbin..sharezookeeperjline-0.9.94.jarusrbin..srcjavalib*.jarusrbin..etczookeeper
-Xmx2G -Xms2G -Dcom.sun.management.jmxremote
-Dcom.sun.management.jmxremote.local.onlyfalse
org.apache.zookeeper.server.quorum.QuorumPeerMain
usrbin..etczookeeperzoo.cfg ```

By default, when the Pods PersistentVolumes is mounted to the ZooKeeper
servers data directory, it is only accessible by the root user. This
configuration prevents the ZooKeeper process from writing to its WAL and
storing its snapshots.

Use the command below to get the file permissions of the ZooKeeper data
directory on the `zk-0` Pod.

```shell kubectl exec -ti zk-0 -- ls -ld varlibzookeeperdata ```

Because the `fsGroup` field of the `securityContext` object is set
to 1000, the ownership of the Pods PersistentVolumes is set to the
zookeeper group, and the ZooKeeper process is able to read and write its
data.

``` drwxr-sr-x 3 zookeeper zookeeper 4096 Dec 5 2045
varlibzookeeperdata ```

# # Managing the ZooKeeper process

The [ZooKeeper
documentation](httpszookeeper.apache.orgdoccurrentzookeeperAdmin.html#sc_supervision)
mentions that You will want to have a supervisory process that manages
each of your ZooKeeper server processes (JVM). Utilizing a watchdog
(supervisory process) to restart failed processes in a distributed
system is a common pattern. When deploying an application in Kubernetes,
rather than using an external utility as a supervisory process, you
should use Kubernetes as the watchdog for your application.

# # # Updating the ensemble

The `zk` `StatefulSet` is configured to use the `RollingUpdate`
update strategy.

You can use `kubectl patch` to update the number of `cpus` allocated
to the servers.

```shell kubectl patch sts zk --typejson -p[op replace, path
spectemplatespeccontainers0resourcesrequestscpu, value0.3] ```

``` statefulset.appszk patched ```

Use `kubectl rollout status` to watch the status of the update.

```shell kubectl rollout status stszk ```

``` waiting for statefulset rolling update to complete 0 pods at
revision zk-5db4499664... Waiting for 1 pods to be ready... Waiting
for 1 pods to be ready... waiting for statefulset rolling update to
complete 1 pods at revision zk-5db4499664... Waiting for 1 pods to be
ready... Waiting for 1 pods to be ready... waiting for statefulset
rolling update to complete 2 pods at revision zk-5db4499664... Waiting
for 1 pods to be ready... Waiting for 1 pods to be ready...
statefulset rolling update complete 3 pods at revision zk-5db4499664...
```

This terminates the Pods, one at a time, in reverse ordinal order, and
recreates them with the new configuration. This ensures that quorum is
maintained during a rolling update.

Use the `kubectl rollout history` command to view a history or
previous configurations.

```shell kubectl rollout history stszk ```

The output is similar to this

``` statefulsets zk REVISION 1 2 ```

Use the `kubectl rollout undo` command to roll back the modification.

```shell kubectl rollout undo stszk ```

The output is similar to this

``` statefulset.appszk rolled back ```

# # # Handling process failure

[Restart
Policies](docsconceptsworkloadspodspod-lifecycle#restart-policy)
control how Kubernetes handles process failures for the entry point of
the container in a Pod. For Pods in a `StatefulSet`, the only
appropriate `RestartPolicy` is Always, and this is the default value.
For stateful applications you should **never** override the default
policy.

Use the following command to examine the process tree for the ZooKeeper
server running in the `zk-0` Pod.

```shell kubectl exec zk-0 -- ps -ef ```

The command used as the containers entry point has PID 1, and the
ZooKeeper process, a child of the entry point, has PID 27.

``` UID PID PPID C STIME TTY TIME CMD zookeep 1 0 0 1503 000000 sh -c
zkGenConfig.sh zkServer.sh start-foreground zookeep 27 1 0 1503 000003
usrlibjvmjava-8-openjdk-amd64binjava -Dzookeeper.log.dirvarlogzookeeper
-Dzookeeper.root.loggerINFO,CONSOLE -cp
usrbin..buildclassesusrbin..buildlib*.jarusrbin..sharezookeeperzookeeper-3.4.9.jarusrbin..sharezookeeperslf4j-log4j12-1.6.1.jarusrbin..sharezookeeperslf4j-api-1.6.1.jarusrbin..sharezookeepernetty-3.10.5.Final.jarusrbin..sharezookeeperlog4j-1.2.16.jarusrbin..sharezookeeperjline-0.9.94.jarusrbin..srcjavalib*.jarusrbin..etczookeeper
-Xmx2G -Xms2G -Dcom.sun.management.jmxremote
-Dcom.sun.management.jmxremote.local.onlyfalse
org.apache.zookeeper.server.quorum.QuorumPeerMain
usrbin..etczookeeperzoo.cfg ```

In another terminal watch the Pods in the `zk` `StatefulSet` with
the following command.

```shell kubectl get pod -w -l appzk ```

In another terminal, terminate the ZooKeeper process in Pod `zk-0`
with the following command.

```shell kubectl exec zk-0 -- pkill java ```

The termination of the ZooKeeper process caused its parent process to
terminate. Because the `RestartPolicy` of the container is Always, it
restarted the parent process.

``` NAME READY STATUS RESTARTS AGE zk-0 11 Running 0 21m zk-1 11
Running 0 20m zk-2 11 Running 0 19m NAME READY STATUS RESTARTS AGE zk-0
01 Error 0 29m zk-0 01 Running 1 29m zk-0 11 Running 1 29m ```

If your application uses a script (such as `zkServer.sh`) to launch
the process that implements the applications business logic, the script
must terminate with the child process. This ensures that Kubernetes will
restart the applications container when the process implementing the
applications business logic fails.

# # # Testing for liveness

Configuring your application to restart failed processes is not enough
to keep a distributed system healthy. There are scenarios where a
systems processes can be both alive and unresponsive, or otherwise
unhealthy. You should use liveness probes to notify Kubernetes that your
applications processes are unhealthy and it should restart them.

The Pod `template` for the `zk` `StatefulSet` specifies a liveness
probe.

```yaml livenessProbe exec command  - sh  - -c  - zookeeper-ready
2181 initialDelaySeconds 15 timeoutSeconds 5 ```

The probe calls a bash script that uses the ZooKeeper `ruok` four
letter word to test the servers health.

``` OK(echo ruok nc 127.0.0.1 1) if [ OK imok ] then exit 0 else
exit 1 fi ```

In one terminal window, use the following command to watch the Pods in
the `zk` StatefulSet.

```shell kubectl get pod -w -l appzk ```

In another window, using the following command to delete the
`zookeeper-ready` script from the file system of Pod `zk-0`.

```shell kubectl exec zk-0 -- rm optzookeeperbinzookeeper-ready
```

When the liveness probe for the ZooKeeper process fails, Kubernetes will
automatically restart the process for you, ensuring that unhealthy
processes in the ensemble are restarted.

```shell kubectl get pod -w -l appzk ```

``` NAME READY STATUS RESTARTS AGE zk-0 11 Running 0 1h zk-1 11
Running 0 1h zk-2 11 Running 0 1h NAME READY STATUS RESTARTS AGE zk-0 01
Running 0 1h zk-0 01 Running 1 1h zk-0 11 Running 1 1h ```

# # # Testing for readiness

Readiness is not the same as liveness. If a process is alive, it is
scheduled and healthy. If a process is ready, it is able to process
input. Liveness is a necessary, but not sufficient, condition for
readiness. There are cases, particularly during initialization and
termination, when a process can be alive but not ready.

If you specify a readiness probe, Kubernetes will ensure that your
applications processes will not receive network traffic until their
readiness checks pass.

For a ZooKeeper server, liveness implies readiness. Therefore, the
readiness probe from the `zookeeper.yaml` manifest is identical to the
liveness probe.

```yaml readinessProbe exec command  - sh  - -c  - zookeeper-ready
2181 initialDelaySeconds 15 timeoutSeconds 5 ```

Even though the liveness and readiness probes are identical, it is
important to specify both. This ensures that only healthy servers in the
ZooKeeper ensemble receive network traffic.

# # Tolerating Node failure

ZooKeeper needs a quorum of servers to successfully commit mutations to
data. For a three server ensemble, two servers must be healthy for
writes to succeed. In quorum based systems, members are deployed across
failure domains to ensure availability. To avoid an outage, due to the
loss of an individual machine, best practices preclude co-locating
multiple instances of the application on the same machine.

By default, Kubernetes may co-locate Pods in a `StatefulSet` on the
same node. For the three server ensemble you created, if two servers are
on the same node, and that node fails, the clients of your ZooKeeper
service will experience an outage until at least one of the Pods can be
rescheduled.

You should always provision additional capacity to allow the processes
of critical systems to be rescheduled in the event of node failures. If
you do so, then the outage will only last until the Kubernetes scheduler
reschedules one of the ZooKeeper servers. However, if you want your
service to tolerate node failures with no downtime, you should set
`podAntiAffinity`.

Use the command below to get the nodes for Pods in the `zk`
`StatefulSet`.

```shell for i in 0 1 2 do kubectl get pod zk-i --template
.spec.nodeName echo done ```

All of the Pods in the `zk` `StatefulSet` are deployed on different
nodes.

``` kubernetes-node-cxpk kubernetes-node-a5aq kubernetes-node-2g2d
```

This is because the Pods in the `zk` `StatefulSet` have a
`PodAntiAffinity` specified.

```yaml affinity podAntiAffinity
requiredDuringSchedulingIgnoredDuringExecution  - labelSelector
matchExpressions  - key app operator In values  - zk topologyKey
kubernetes.iohostname ```

The `requiredDuringSchedulingIgnoredDuringExecution` field tells the
Kubernetes Scheduler that it should never co-locate two Pods which have
`app` label as `zk` in the domain defined by the `topologyKey`.
The `topologyKey` `kubernetes.iohostname` indicates that the domain
is an individual node. Using different rules, labels, and selectors, you
can extend this technique to spread your ensemble across physical,
network, and power failure domains.

# # Surviving maintenance

In this section you will cordon and drain nodes. If you are using this
tutorial on a shared cluster, be sure that this will not adversely
affect other tenants.

The previous section showed you how to spread your Pods across nodes to
survive unplanned node failures, but you also need to plan for temporary
node failures that occur due to planned maintenance.

Use this command to get the nodes in your cluster.

```shell kubectl get nodes ```

This tutorial assumes a cluster with at least four nodes. If the cluster
has more than four, use [`kubectl
cordon`](docsreferencegeneratedkubectlkubectl-commands#cordon) to
cordon all but four nodes. Constraining to four nodes will ensure
Kubernetes encounters affinity and PodDisruptionBudget constraints when
scheduling zookeeper Pods in the following maintenance simulation.

```shell kubectl cordon ```

Use this command to get the `zk-pdb` `PodDisruptionBudget`.

```shell kubectl get pdb zk-pdb ```

The `max-unavailable` field indicates to Kubernetes that at most one
Pod from `zk` `StatefulSet` can be unavailable at any time.

``` NAME MIN-AVAILABLE MAX-UNAVAILABLE ALLOWED-DISRUPTIONS AGE zk-pdb
NA 1 1 ```

In one terminal, use this command to watch the Pods in the `zk`
`StatefulSet`.

```shell kubectl get pods -w -l appzk ```

In another terminal, use this command to get the nodes that the Pods are
currently scheduled on.

```shell for i in 0 1 2 do kubectl get pod zk-i --template
.spec.nodeName echo done ```

The output is similar to this

``` kubernetes-node-pb41 kubernetes-node-ixsl kubernetes-node-i4c4
```

Use [`kubectl
drain`](docsreferencegeneratedkubectlkubectl-commands#drain) to cordon
and drain the node on which the `zk-0` Pod is scheduled.

```shell kubectl drain (kubectl get pod zk-0 --template
.spec.nodeName) --ignore-daemonsets --force --delete-emptydir-data
```

The output is similar to this

``` node kubernetes-node-pb41 cordoned

WARNING Deleting pods not managed by ReplicationController, ReplicaSet,
Job, or DaemonSet fluentd-cloud-logging-kubernetes-node-pb41,
kube-proxy-kubernetes-node-pb41 Ignoring DaemonSet-managed pods
node-problem-detector-v0.1-o5elz pod zk-0 deleted node
kubernetes-node-pb41 drained ```

As there are four nodes in your cluster, `kubectl drain`, succeeds and
the `zk-0` is rescheduled to another node.

``` NAME READY STATUS RESTARTS AGE zk-0 11 Running 2 1h zk-1 11
Running 0 1h zk-2 11 Running 0 1h NAME READY STATUS RESTARTS AGE zk-0 11
Terminating 2 2h zk-0 01 Terminating 2 2h zk-0 01 Terminating 2 2h zk-0
01 Terminating 2 2h zk-0 01 Pending 0 0s zk-0 01 Pending 0 0s zk-0 01
ContainerCreating 0 0s zk-0 01 Running 0 51s zk-0 11 Running 0 1m ```

Keep watching the `StatefulSet`s Pods in the first terminal and drain
the node on which `zk-1` is scheduled.

```shell kubectl drain (kubectl get pod zk-1 --template
.spec.nodeName) --ignore-daemonsets --force --delete-emptydir-data
```

The output is similar to this

``` kubernetes-node-ixsl cordoned WARNING Deleting pods not managed
by ReplicationController, ReplicaSet, Job, or DaemonSet
fluentd-cloud-logging-kubernetes-node-ixsl,
kube-proxy-kubernetes-node-ixsl Ignoring DaemonSet-managed pods
node-problem-detector-v0.1-voc74 pod zk-1 deleted node
kubernetes-node-ixsl drained ```

The `zk-1` Pod cannot be scheduled because the `zk` `StatefulSet`
contains a `PodAntiAffinity` rule preventing co-location of the Pods,
and as only two nodes are schedulable, the Pod will remain in a Pending
state.

```shell kubectl get pods -w -l appzk ```

The output is similar to this

``` NAME READY STATUS RESTARTS AGE zk-0 11 Running 2 1h zk-1 11
Running 0 1h zk-2 11 Running 0 1h NAME READY STATUS RESTARTS AGE zk-0 11
Terminating 2 2h zk-0 01 Terminating 2 2h zk-0 01 Terminating 2 2h zk-0
01 Terminating 2 2h zk-0 01 Pending 0 0s zk-0 01 Pending 0 0s zk-0 01
ContainerCreating 0 0s zk-0 01 Running 0 51s zk-0 11 Running 0 1m zk-1
11 Terminating 0 2h zk-1 01 Terminating 0 2h zk-1 01 Terminating 0 2h
zk-1 01 Terminating 0 2h zk-1 01 Pending 0 0s zk-1 01 Pending 0 0s
```

Continue to watch the Pods of the StatefulSet, and drain the node on
which `zk-2` is scheduled.

```shell kubectl drain (kubectl get pod zk-2 --template
.spec.nodeName) --ignore-daemonsets --force --delete-emptydir-data
```

The output is similar to this

``` node kubernetes-node-i4c4 cordoned

WARNING Deleting pods not managed by ReplicationController, ReplicaSet,
Job, or DaemonSet fluentd-cloud-logging-kubernetes-node-i4c4,
kube-proxy-kubernetes-node-i4c4 Ignoring DaemonSet-managed pods
node-problem-detector-v0.1-dyrog WARNING Ignoring DaemonSet-managed pods
node-problem-detector-v0.1-dyrog Deleting pods not managed by
ReplicationController, ReplicaSet, Job, or DaemonSet
fluentd-cloud-logging-kubernetes-node-i4c4,
kube-proxy-kubernetes-node-i4c4 There are pending pods when an error
occurred Cannot evict pod as it would violate the pods disruption
budget. podzk-2 ```

Use `CTRL-C` to terminate kubectl.

You cannot drain the third node because evicting `zk-2` would violate
`zk-budget`. However, the node will remain cordoned.

Use `zkCli.sh` to retrieve the value you entered during the sanity
test from `zk-0`.

```shell kubectl exec zk-0 zkCli.sh get hello ```

The service is still available because its `PodDisruptionBudget` is
respected.

``` WatchedEvent stateSyncConnected typeNone pathnull world cZxid
0x200000002 ctime Wed Dec 07 000859 UTC 2016 mZxid 0x200000002 mtime Wed
Dec 07 000859 UTC 2016 pZxid 0x200000002 cversion 0 dataVersion 0
aclVersion 0 ephemeralOwner 0x0 dataLength 5 numChildren 0 ```

Use [`kubectl
uncordon`](docsreferencegeneratedkubectlkubectl-commands#uncordon) to
uncordon the first node.

```shell kubectl uncordon kubernetes-node-pb41 ```

The output is similar to this

``` node kubernetes-node-pb41 uncordoned ```

`zk-1` is rescheduled on this node. Wait until `zk-1` is Running and
Ready.

```shell kubectl get pods -w -l appzk ```

The output is similar to this

``` NAME READY STATUS RESTARTS AGE zk-0 11 Running 2 1h zk-1 11
Running 0 1h zk-2 11 Running 0 1h NAME READY STATUS RESTARTS AGE zk-0 11
Terminating 2 2h zk-0 01 Terminating 2 2h zk-0 01 Terminating 2 2h zk-0
01 Terminating 2 2h zk-0 01 Pending 0 0s zk-0 01 Pending 0 0s zk-0 01
ContainerCreating 0 0s zk-0 01 Running 0 51s zk-0 11 Running 0 1m zk-1
11 Terminating 0 2h zk-1 01 Terminating 0 2h zk-1 01 Terminating 0 2h
zk-1 01 Terminating 0 2h zk-1 01 Pending 0 0s zk-1 01 Pending 0 0s zk-1
01 Pending 0 12m zk-1 01 ContainerCreating 0 12m zk-1 01 Running 0 13m
zk-1 11 Running 0 13m ```

Attempt to drain the node on which `zk-2` is scheduled.

```shell kubectl drain (kubectl get pod zk-2 --template
.spec.nodeName) --ignore-daemonsets --force --delete-emptydir-data
```

The output is similar to this

``` node kubernetes-node-i4c4 already cordoned WARNING Deleting pods
not managed by ReplicationController, ReplicaSet, Job, or DaemonSet
fluentd-cloud-logging-kubernetes-node-i4c4,
kube-proxy-kubernetes-node-i4c4 Ignoring DaemonSet-managed pods
node-problem-detector-v0.1-dyrog pod heapster-v1.2.0-2604621511-wht1r
deleted pod zk-2 deleted node kubernetes-node-i4c4 drained ```

This time `kubectl drain` succeeds.

Uncordon the second node to allow `zk-2` to be rescheduled.

```shell kubectl uncordon kubernetes-node-ixsl ```

The output is similar to this

``` node kubernetes-node-ixsl uncordoned ```

You can use `kubectl drain` in conjunction with
`PodDisruptionBudgets` to ensure that your services remain available
during maintenance. If drain is used to cordon nodes and evict pods
prior to taking the node offline for maintenance, services that express
a disruption budget will have that budget respected. You should always
allocate additional capacity for critical services so that their Pods
can be immediately rescheduled.

# # heading cleanup

- Use `kubectl uncordon` to uncordon all the nodes in your cluster. -
You must delete the persistent storage media for the PersistentVolumes
used in this tutorial. Follow the necessary steps, based on your
environment, storage configuration, and provisioning method, to ensure
that all storage is reclaimed.

 FILE datakubernetes
website main content-en_docstutorialsstateless-application_index.md
 --- title Stateless
Applications weight 40 ---

 FILE datakubernetes
website main
content-en_docstutorialsstateless-applicationexpose-external-ip-address.md
 --- title Exposing an
External IP Address to Access an Application in a Cluster content_type
tutorial weight 10 ---

This page shows how to create a Kubernetes Service object that exposes
an external IP address.

# # heading prerequisites

* Install [kubectl](docstaskstools). * Use a cloud provider like
Google Kubernetes Engine or Amazon Web Services to create a Kubernetes
cluster. This tutorial creates an [external load
balancer](docstasksaccess-application-clustercreate-external-load-balancer),
which requires a cloud provider. * Configure `kubectl` to communicate
with your Kubernetes API server. For instructions, see the documentation
for your cloud provider.

# # heading objectives

* Run five instances of a Hello World application. * Create a Service
object that exposes an external IP address. * Use the Service object to
access the running application.

# # Creating a service for an application running in five pods

1. Run a Hello World application in your cluster

code_sample fileserviceload-balancer-example.yaml

```shell kubectl apply -f
httpsk8s.ioexamplesserviceload-balancer-example.yaml ``` The
preceding command creates a

and an associated . The ReplicaSet has five

each of which runs the Hello World application.

1. Display information about the Deployment

```shell kubectl get deployments hello-world kubectl describe
deployments hello-world ```

1. Display information about your ReplicaSet objects

```shell kubectl get replicasets kubectl describe replicasets ```

1. Create a Service object that exposes the deployment

```shell kubectl expose deployment hello-world --typeLoadBalancer
--namemy-service ```

1. Display information about the Service

```shell kubectl get services my-service ```

The output is similar to

```console NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE my-service
LoadBalancer 10.3.245.137 104.198.205.71 8080TCP 54s ```

The `typeLoadBalancer` service is backed by external cloud providers,
which is not covered in this example, please refer to [this
page](docsconceptsservices-networkingservice#loadbalancer) for the
details.

If the external IP address is shown as , wait for a minute and enter the
same command again.

1. Display detailed information about the Service

```shell kubectl describe services my-service ```

The output is similar to

```console Name my-service Namespace default Labels
app.kubernetes.ionameload-balancer-example Annotations Selector
app.kubernetes.ionameload-balancer-example Type LoadBalancer IP
10.3.245.137 LoadBalancer Ingress 104.198.205.71 Port 8080TCP NodePort
32377TCP Endpoints 10.0.0.68080,10.0.1.68080,10.0.1.78080 2 more...
Session Affinity None Events ```

Make a note of the external IP address (`LoadBalancer Ingress`)
exposed by your service. In this example, the external IP address is
104.198.205.71. Also note the value of `Port` and `NodePort`. In
this example, the `Port` is 8080 and the `NodePort` is 32377.

1. In the preceding output, you can see that the service has several
endpoints 10.0.0.68080,10.0.1.68080,10.0.1.78080 2 more. These are
internal addresses of the pods that are running the Hello World
application. To verify these are pod addresses, enter this command

```shell kubectl get pods --outputwide ```

The output is similar to

```console NAME ... IP NODE hello-world-2895499144-1jaz9 ...
10.0.1.6 gke-cluster-1-default-pool-e0b8d269-1afc
hello-world-2895499144-2e5uh ... 10.0.1.8
gke-cluster-1-default-pool-e0b8d269-1afc hello-world-2895499144-9m4h1
... 10.0.0.6 gke-cluster-1-default-pool-e0b8d269-5v7a
hello-world-2895499144-o4z13 ... 10.0.1.7
gke-cluster-1-default-pool-e0b8d269-1afc hello-world-2895499144-segjf
... 10.0.2.5 gke-cluster-1-default-pool-e0b8d269-cpuc ```

1. Use the external IP address (`LoadBalancer Ingress`) to access the
Hello World application

```shell curl http ```

where `` is the external IP address (`LoadBalancer Ingress`) of your
Service, and `` is the value of `Port` in your Service description.
If you are using minikube, typing `minikube service my-service` will
automatically open the Hello World application in a browser.

The response to a successful request is a hello message

```shell Hello, world! Version 2.0.0 Hostname 0bd46b45f32f ```

# # heading cleanup

To delete the Service, enter this command

```shell kubectl delete services my-service ```

To delete the Deployment, the ReplicaSet, and the Pods that are running
the Hello World application, enter this command

```shell kubectl delete deployment hello-world ```

# # heading whatsnext

Learn more about [connecting applications with
services](docstutorialsservicesconnect-applications-service).

 FILE datakubernetes
website main
content-en_docstutorialsstateless-applicationguestbook.md
 --- title Example
Deploying PHP Guestbook application with Redis reviewers - ahmetb -
jimangel content_type tutorial weight 20 card name tutorials weight 30
title Stateless Example PHP Guestbook with Redis
min-kubernetes-server-version v1.14 source
httpscloud.google.comkubernetes-enginedocstutorialsguestbook ---

This tutorial shows you how to build and deploy a simple _(not
production ready)_, multi-tier web application using Kubernetes and
[Docker](httpswww.docker.com). This example consists of the following
components

* A single-instance [Redis](httpswww.redis.io) to store guestbook
entries * Multiple web frontend instances

# # heading objectives

* Start up a Redis leader. * Start up two Redis followers. * Start up
the guestbook frontend. * Expose and view the Frontend Service. *
Clean up.

# # heading prerequisites

# # Start up the Redis Database

The guestbook application uses Redis to store its data.

# # # Creating the Redis Deployment

The manifest file, included below, specifies a Deployment controller
that runs a single replica Redis Pod.

code_sample fileapplicationguestbookredis-leader-deployment.yaml

1. Launch a terminal window in the directory you downloaded the
manifest files. 1. Apply the Redis Deployment from the
`redis-leader-deployment.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookredis-leader-deployment.yaml
```

1. Query the list of Pods to verify that the Redis Pod is running

```shell kubectl get pods ```

The response should be similar to this

``` NAME READY STATUS RESTARTS AGE redis-leader-fb76b4755-xjr2n 11
Running 0 13s ```

1. Run the following command to view the logs from the Redis leader Pod

```shell kubectl logs -f deploymentredis-leader ```

# # # Creating the Redis leader Service

The guestbook application needs to communicate to the Redis to write its
data. You need to apply a
[Service](docsconceptsservices-networkingservice) to proxy the traffic
to the Redis Pod. A Service defines a policy to access the Pods.

code_sample fileapplicationguestbookredis-leader-service.yaml

1. Apply the Redis Service from the following
`redis-leader-service.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookredis-leader-service.yaml ```

1. Query the list of Services to verify that the Redis Service is
running

```shell kubectl get service ```

The response should be similar to this

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE kubernetes ClusterIP
10.0.0.1 443TCP 1m redis-leader ClusterIP 10.103.78.24 6379TCP 16s
```

This manifest file creates a Service named `redis-leader` with a set
of labels that match the labels previously defined, so the Service
routes network traffic to the Redis Pod.

# # # Set up Redis followers

Although the Redis leader is a single Pod, you can make it highly
available and meet traffic demands by adding a few Redis followers, or
replicas.

code_sample fileapplicationguestbookredis-follower-deployment.yaml

1. Apply the Redis Deployment from the following
`redis-follower-deployment.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookredis-follower-deployment.yaml
```

1. Verify that the two Redis follower replicas are running by querying
the list of Pods

```shell kubectl get pods ```

The response should be similar to this

``` NAME READY STATUS RESTARTS AGE redis-follower-dddfbdcc9-82sfr 11
Running 0 37s redis-follower-dddfbdcc9-qrt5k 11 Running 0 38s
redis-leader-fb76b4755-xjr2n 11 Running 0 11m ```

# # # Creating the Redis follower service

The guestbook application needs to communicate with the Redis followers
to read data. To make the Redis followers discoverable, you must set up
another [Service](docsconceptsservices-networkingservice).

code_sample fileapplicationguestbookredis-follower-service.yaml

1. Apply the Redis Service from the following
`redis-follower-service.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookredis-follower-service.yaml
```

1. Query the list of Services to verify that the Redis Service is
running

```shell kubectl get service ```

The response should be similar to this

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE kubernetes ClusterIP
10.96.0.1 443TCP 3d19h redis-follower ClusterIP 10.110.162.42 6379TCP 9s
redis-leader ClusterIP 10.103.78.24 6379TCP 6m10s ```

This manifest file creates a Service named `redis-follower` with a set
of labels that match the labels previously defined, so the Service
routes network traffic to the Redis Pod.

# # Set up and Expose the Guestbook Frontend

Now that you have the Redis storage of your guestbook up and running,
start the guestbook web servers. Like the Redis followers, the frontend
is deployed using a Kubernetes Deployment.

The guestbook app uses a PHP frontend. It is configured to communicate
with either the Redis follower or leader Services, depending on whether
the request is a read or a write. The frontend exposes a JSON interface,
and serves a jQuery-Ajax-based UX.

# # # Creating the Guestbook Frontend Deployment

code_sample fileapplicationguestbookfrontend-deployment.yaml

1. Apply the frontend Deployment from the `frontend-deployment.yaml`
file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookfrontend-deployment.yaml ```

1. Query the list of Pods to verify that the three frontend replicas
are running

```shell kubectl get pods -l appguestbook -l tierfrontend ```

The response should be similar to this

``` NAME READY STATUS RESTARTS AGE frontend-85595f5bf9-5tqhb 11
Running 0 47s frontend-85595f5bf9-qbzwm 11 Running 0 47s
frontend-85595f5bf9-zchwc 11 Running 0 47s ```

# # # Creating the Frontend Service

The `Redis` Services you applied is only accessible within the
Kubernetes cluster because the default type for a Service is
[ClusterIP](docsconceptsservices-networkingservice#publishing-services-service-types).
`ClusterIP` provides a single IP address for the set of Pods the
Service is pointing to. This IP address is accessible only within the
cluster.

If you want guests to be able to access your guestbook, you must
configure the frontend Service to be externally visible, so a client can
request the Service from outside the Kubernetes cluster. However a
Kubernetes user can use `kubectl port-forward` to access the service
even though it uses a `ClusterIP`.

Some cloud providers, like Google Compute Engine or Google Kubernetes
Engine, support external load balancers. If your cloud provider supports
load balancers and you want to use it, uncomment `type LoadBalancer`.

code_sample fileapplicationguestbookfrontend-service.yaml

1. Apply the frontend Service from the `frontend-service.yaml` file

```shell kubectl apply -f
httpsk8s.ioexamplesapplicationguestbookfrontend-service.yaml ```

1. Query the list of Services to verify that the frontend Service is
running

```shell kubectl get services ```

The response should be similar to this

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE frontend ClusterIP
10.97.28.230 80TCP 19s kubernetes ClusterIP 10.96.0.1 443TCP 3d19h
redis-follower ClusterIP 10.110.162.42 6379TCP 5m48s redis-leader
ClusterIP 10.103.78.24 6379TCP 11m ```

# # # Viewing the Frontend Service via `kubectl port-forward`

1. Run the following command to forward port `8080` on your local
machine to port `80` on the service.

```shell kubectl port-forward svcfrontend 808080 ```

The response should be similar to this

``` Forwarding from 127.0.0.18080 - 80 Forwarding from [1]8080 - 80
```

1. load the page [httplocalhost8080](httplocalhost8080) in your
browser to view your guestbook.

# # # Viewing the Frontend Service via `LoadBalancer`

If you deployed the `frontend-service.yaml` manifest with type
`LoadBalancer` you need to find the IP address to view your Guestbook.

1. Run the following command to get the IP address for the frontend
Service.

```shell kubectl get service frontend ```

The response should be similar to this

``` NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE frontend
LoadBalancer 10.51.242.136 109.197.92.229 8032372TCP 1m ```

1. Copy the external IP address, and load the page in your browser to
view your guestbook.

Try adding some guestbook entries by typing in a message, and clicking
Submit. The message you typed appears in the frontend. This message
indicates that data is successfully added to Redis through the Services
you created earlier.

# # Scale the Web Frontend

You can scale up or down as needed because your servers are defined as a
Service that uses a Deployment controller.

1. Run the following command to scale up the number of frontend Pods

```shell kubectl scale deployment frontend --replicas5 ```

1. Query the list of Pods to verify the number of frontend Pods running

```shell kubectl get pods ```

The response should look similar to this

``` NAME READY STATUS RESTARTS AGE frontend-85595f5bf9-5df5m 11
Running 0 83s frontend-85595f5bf9-7zmg5 11 Running 0 83s
frontend-85595f5bf9-cpskg 11 Running 0 15m frontend-85595f5bf9-l2l54 11
Running 0 14m frontend-85595f5bf9-l9c8z 11 Running 0 14m
redis-follower-dddfbdcc9-82sfr 11 Running 0 97m
redis-follower-dddfbdcc9-qrt5k 11 Running 0 97m
redis-leader-fb76b4755-xjr2n 11 Running 0 108m ```

1. Run the following command to scale down the number of frontend Pods

```shell kubectl scale deployment frontend --replicas2 ```

1. Query the list of Pods to verify the number of frontend Pods running

```shell kubectl get pods ```

The response should look similar to this

``` NAME READY STATUS RESTARTS AGE frontend-85595f5bf9-cpskg 11
Running 0 16m frontend-85595f5bf9-l9c8z 11 Running 0 15m
redis-follower-dddfbdcc9-82sfr 11 Running 0 98m
redis-follower-dddfbdcc9-qrt5k 11 Running 0 98m
redis-leader-fb76b4755-xjr2n 11 Running 0 109m ```

# # heading cleanup

Deleting the Deployments and Services also deletes any running Pods. Use
labels to delete multiple resources with one command.

1. Run the following commands to delete all Pods, Deployments, and
Services.

```shell kubectl delete deployment -l appredis kubectl delete service
-l appredis kubectl delete deployment frontend kubectl delete service
frontend ```

The response should look similar to this

``` deployment.apps redis-follower deleted deployment.apps
redis-leader deleted deployment.apps frontend deleted service frontend
deleted ```

1. Query the list of Pods to verify that no Pods are running

```shell kubectl get pods ```

The response should look similar to this

``` No resources found in default namespace. ```

# # heading whatsnext

* Complete the [Kubernetes Basics](docstutorialskubernetes-basics)
Interactive Tutorials * Use Kubernetes to create a blog using
[Persistent Volumes for MySQL and
Wordpress](docstutorialsstateful-applicationmysql-wordpress-persistent-volume#visit-your-new-wordpress-blog)
* Read more about [connecting applications with
services](docstutorialsservicesconnect-applications-service) * Read
more about [using labels
effectively](docsconceptsoverviewworking-with-objectslabels#using-labels-effectively)
