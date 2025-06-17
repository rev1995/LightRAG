---
reviewers
- jsafrane
title Create static Pods
weight 220
content_type task
---

*Static Pods* are managed directly by the kubelet daemon on a specific node,
without the
observing them.
Unlike Pods that are managed by the control plane (for example, a
)
instead, the kubelet watches each static Pod (and restarts it if it fails).

Static Pods are always bound to one  on a specific node.

The kubelet automatically tries to create a
on the Kubernetes API server for each static Pod.
This means that the Pods running on a node are visible on the API server,
but cannot be controlled from there.
The Pod names will be suffixed with the node hostname with a leading hyphen.

If you are running clustered Kubernetes and are using static
Pods to run a Pod on every node, you should probably be using a
 instead.

The `spec` of a static Pod cannot refer to other API objects
(e.g., ,
,
, etc).

Static pods do not support [ephemeral containers](docsconceptsworkloadspodsephemeral-containers).

# #  heading prerequisites

This page assumes youre using  to run Pods,
and that your nodes are running the Fedora operating system.
Instructions for other distributions or Kubernetes installations may vary.

# # Create a static pod #static-pod-creation

You can configure a static Pod with either a
[file system hosted configuration file](docstasksconfigure-pod-containerstatic-pod#configuration-files)
or a [web hosted configuration file](docstasksconfigure-pod-containerstatic-pod#pods-created-via-http).

# # # Filesystem-hosted static Pod manifest #configuration-files

Manifests are standard Pod definitions in JSON or YAML format in a specific directory.
Use the `staticPodPath ` field in the
[kubelet configuration file](docsreferenceconfig-apikubelet-config.v1beta1),
which periodically scans the directory and createsdeletes static Pods as YAMLJSON files appeardisappear there.
Note that the kubelet will ignore files starting with dots when scanning the specified directory.

For example, this is how to start a simple web server as a static Pod

1. Choose a node where you want to run the static Pod. In this example, its `my-node1`.

    ```shell
    ssh my-node1
    ```

1. Choose a directory, say `etckubernetesmanifests` and place a web server
   Pod definition there, for example `etckubernetesmanifestsstatic-web.yaml`

   ```shell
   # Run this command on the node where kubelet is running
   mkdir -p etckubernetesmanifests
   cat etckubernetesmanifestsstatic-web.yaml
   apiVersion v1
   kind Pod
   metadata
     name static-web
     labels
       role myrole
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

1. Configure the kubelet on that node to set a `staticPodPath` value in the
   [kubelet configuration file](docsreferenceconfig-apikubelet-config.v1beta1).
   See [Set Kubelet Parameters Via A Configuration File](docstasksadminister-clusterkubelet-config-file)
   for more information.

   An alternative and deprecated method is to configure the kubelet on that node
   to look for static Pod manifests locally, using a command line argument.
   To use the deprecated approach, start the kubelet with the
   `--pod-manifest-pathetckubernetesmanifests` argument.

1. Restart the kubelet. On Fedora, you would run

   ```shell
   # Run this command on the node where the kubelet is running
   systemctl restart kubelet
   ```

# # # Web-hosted static pod manifest #pods-created-via-http

Kubelet periodically downloads a file specified by `--manifest-url` argument
and interprets it as a JSONYAML file that contains Pod definitions.
Similar to how [filesystem-hosted manifests](#configuration-files) work, the kubelet
refetches the manifest on a schedule. If there are changes to the list of static
Pods, the kubelet applies them.

To use this approach

1. Create a YAML file and store it on a web server so that you can pass the URL of that file to the kubelet.

    ```yaml
    apiVersion v1
    kind Pod
    metadata
      name static-web
      labels
        role myrole
    spec
      containers
        - name web
          image nginx
          ports
            - name web
              containerPort 80
              protocol TCP
    ```

1. Configure the kubelet on your selected node to use this web manifest by
   running it with `--manifest-url`.
   On Fedora, edit `etckuberneteskubelet` to include this line

   ```shell
   KUBELET_ARGS--cluster-dns10.254.0.10 --cluster-domainkube.local --manifest-url
   ```

1. Restart the kubelet. On Fedora, you would run

   ```shell
   # Run this command on the node where the kubelet is running
   systemctl restart kubelet
   ```

# # Observe static pod behavior #behavior-of-static-pods

When the kubelet starts, it automatically starts all defined static Pods. As you have
defined a static Pod and restarted the kubelet, the new static Pod should
already be running.

You can view running containers (including static Pods) by running (on the node)
```shell
# Run this command on the node where the kubelet is running
crictl ps
```

The output might be something like

```console
CONTAINER       IMAGE                                 CREATED           STATE      NAME    ATTEMPT    POD ID
129fd7d382018   docker.iolibrarynginxsha256...    11 minutes ago    Running    web     0          34533c6729106
```

`crictl` outputs the image URI and SHA-256 checksum. `NAME` will look more like
`docker.iolibrarynginxsha2560d17b565c37bcbd895e9d92315a05c1c3c9a29f762b011a10c54a66cd53c9b31`.

You can see the mirror Pod on the API server

```shell
kubectl get pods
```
```
NAME                  READY   STATUS    RESTARTS        AGE
static-web-my-node1   11     Running   0               2m
```

Make sure the kubelet has permission to create the mirror Pod in the API server.
If not, the creation request is rejected by the API server.

 from the static Pod are
propagated into the mirror Pod. You can use those labels as normal via
, etc.

If you try to use `kubectl` to delete the mirror Pod from the API server,
the kubelet _doesnt_ remove the static Pod

```shell
kubectl delete pod static-web-my-node1
```
```
pod static-web-my-node1 deleted
```
You can see that the Pod is still running
```shell
kubectl get pods
```
```
NAME                  READY   STATUS    RESTARTS   AGE
static-web-my-node1   11     Running   0          4s
```

Back on your node where the kubelet is running, you can try to stop the container manually.
Youll see that, after a time, the kubelet will notice and will restart the Pod
automatically

```shell
# Run these commands on the node where the kubelet is running
crictl stop 129fd7d382018 # replace with the ID of your container
sleep 20
crictl ps
```

```console
CONTAINER       IMAGE                                 CREATED           STATE      NAME    ATTEMPT    POD ID
89db4553e1eeb   docker.iolibrarynginxsha256...    19 seconds ago    Running    web     1          34533c6729106
```
Once you identify the right container, you can get the logs for that container with `crictl`

```shell
# Run these commands on the node where the container is running
crictl logs
```

```console
10.240.0.48 - - [16Nov2022124549 0000] GET  HTTP1.1 200 612 - curl7.47.0 -
10.240.0.48 - - [16Nov2022124550 0000] GET  HTTP1.1 200 612 - curl7.47.0 -
10.240.0.48 - - [16Nove2022124551 0000] GET  HTTP1.1 200 612 - curl7.47.0 -
```

To find more about how to debug using `crictl`, please visit
[_Debugging Kubernetes nodes with crictl_](docstasksdebugdebug-clustercrictl).

# # Dynamic addition and removal of static pods

The running kubelet periodically scans the configured directory
(`etckubernetesmanifests` in our example) for changes and
addsremoves Pods as files appeardisappear in this directory.

```shell
# This assumes you are using filesystem-hosted static Pod configuration
# Run these commands on the node where the container is running
#
mv etckubernetesmanifestsstatic-web.yaml tmp
sleep 20
crictl ps
# You see that no nginx container is running
mv tmpstatic-web.yaml  etckubernetesmanifests
sleep 20
crictl ps
```
```console
CONTAINER       IMAGE                                 CREATED           STATE      NAME    ATTEMPT    POD ID
f427638871c35   docker.iolibrarynginxsha256...    19 seconds ago    Running    web     1          34533c6729106
```
# #  heading whatsnext

* [Generate static Pod manifests for control plane components](docsreferencesetup-toolskubeadmimplementation-details#generate-static-pod-manifests-for-control-plane-components)
* [Generate static Pod manifest for local etcd](docsreferencesetup-toolskubeadmimplementation-details#generate-static-pod-manifest-for-local-etcd)
* [Debugging Kubernetes nodes with `crictl`](docstasksdebugdebug-clustercrictl)
* [Learn more about `crictl`](httpsgithub.comkubernetes-sigscri-tools).
* [Map `docker` CLI commands to `crictl`](docsreferencetoolsmap-crictl-dockercli).
* [Set up etcd instances as static pods managed by a kubelet](docssetupproduction-environmenttoolskubeadmsetup-ha-etcd-with-kubeadm)
