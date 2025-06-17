---
reviewers
- verb
- soltysh
title Debug Running Pods
content_type task
---

This page explains how to debug Pods running (or crashing) on a Node.

# #  heading prerequisites

* Your  should already be
  scheduled and running. If your Pod is not yet running, start with [Debugging
  Pods](docstasksdebugdebug-application).
* For some of the advanced debugging steps you need to know on which Node the
  Pod is running and have shell access to run commands on that Node. You dont
  need that access to run the standard debug steps that use `kubectl`.

# # Using `kubectl describe pod` to fetch details about pods

For this example well use a Deployment to create two pods, similar to the earlier example.

 code_sample fileapplicationnginx-with-request.yaml

Create deployment by running following command

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationnginx-with-request.yaml
```

```none
deployment.appsnginx-deployment created
```

Check pod status by following command

```shell
kubectl get pods
```

```none
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-67d4bdd6f5-cx2nz   11     Running   0          13s
nginx-deployment-67d4bdd6f5-w6kd7   11     Running   0          13s
```

We can retrieve a lot more information about each of these pods using `kubectl describe pod`. For example

```shell
kubectl describe pod nginx-deployment-67d4bdd6f5-w6kd7
```

```none
Name         nginx-deployment-67d4bdd6f5-w6kd7
Namespace    default
Priority     0
Node         kube-worker-1192.168.0.113
Start Time   Thu, 17 Feb 2022 165101 -0500
Labels       appnginx
              pod-template-hash67d4bdd6f5
Annotations
Status       Running
IP           10.88.0.3
IPs
  IP           10.88.0.3
  IP           2001db81
Controlled By  ReplicaSetnginx-deployment-67d4bdd6f5
Containers
  nginx
    Container ID   containerd5403af59a2b46ee5a23fb0ae4b1e077f7ca5c5fb7af16e1ab21c00e0e616462a
    Image          nginx
    Image ID       docker.iolibrarynginxsha2562834dc507516af02784808c5f48b7cbe38b8ed5d0f4837f16e78d00deb7e7767
    Port           80TCP
    Host Port      0TCP
    State          Running
      Started      Thu, 17 Feb 2022 165105 -0500
    Ready          True
    Restart Count  0
    Limits
      cpu     500m
      memory  128Mi
    Requests
      cpu        500m
      memory     128Mi
    Environment
    Mounts
      varrunsecretskubernetes.ioserviceaccount from kube-api-access-bgsgp (ro)
Conditions
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes
  kube-api-access-bgsgp
    Type                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds  3607
    ConfigMapName           kube-root-ca.crt
    ConfigMapOptional
    DownwardAPI             true
QoS Class                   Guaranteed
Node-Selectors
Tolerations                 node.kubernetes.ionot-readyNoExecute opExists for 300s
                             node.kubernetes.iounreachableNoExecute opExists for 300s
Events
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  34s   default-scheduler  Successfully assigned defaultnginx-deployment-67d4bdd6f5-w6kd7 to kube-worker-1
  Normal  Pulling    31s   kubelet            Pulling image nginx
  Normal  Pulled     30s   kubelet            Successfully pulled image nginx in 1.146417389s
  Normal  Created    30s   kubelet            Created container nginx
  Normal  Started    30s   kubelet            Started container nginx
```

Here you can see configuration information about the container(s) and Pod (labels, resource requirements, etc.), as well as status information about the container(s) and Pod (state, readiness, restart count, events, etc.).

The container state is one of Waiting, Running, or Terminated. Depending on the state, additional information will be provided -- here you can see that for a container in Running state, the system tells you when the container started.

Ready tells you whether the container passed its last readiness probe. (In this case, the container does not have a readiness probe configured the container is assumed to be ready if no readiness probe is configured.)

Restart Count tells you how many times the container has been restarted this information can be useful for detecting crash loops in containers that are configured with a restart policy of always.

Currently the only Condition associated with a Pod is the binary Ready condition, which indicates that the pod is able to service requests and should be added to the load balancing pools of all matching services.

Lastly, you see a log of recent events related to your Pod. From indicates the component that is logging the event. Reason and Message tell you what happened.

# # Example debugging Pending Pods

A common scenario that you can detect using events is when youve created a Pod that wont fit on any node. For example, the Pod might request more resources than are free on any node, or it might specify a label selector that doesnt match any nodes. Lets say we created the previous Deployment with 5 replicas (instead of 2) and requesting 600 millicores instead of 500, on a four-node cluster where each (virtual) machine has 1 CPU. In that case one of the Pods will not be able to schedule. (Note that because of the cluster addon pods such as fluentd, skydns, etc., that run on each node, if we requested 1000 millicores then none of the Pods would be able to schedule.)

```shell
kubectl get pods
```

```none
NAME                                READY     STATUS    RESTARTS   AGE
nginx-deployment-1006230814-6winp   11       Running   0          7m
nginx-deployment-1006230814-fmgu3   11       Running   0          7m
nginx-deployment-1370807587-6ekbw   11       Running   0          1m
nginx-deployment-1370807587-fg172   01       Pending   0          1m
nginx-deployment-1370807587-fz9sd   01       Pending   0          1m
```

To find out why the nginx-deployment-1370807587-fz9sd pod is not running, we can use `kubectl describe pod` on the pending Pod and look at its events

```shell
kubectl describe pod nginx-deployment-1370807587-fz9sd
```

```none
  Name		nginx-deployment-1370807587-fz9sd
  Namespace	default
  Node
  Labels		appnginx,pod-template-hash1370807587
  Status		Pending
  IP
  Controllers	ReplicaSetnginx-deployment-1370807587
  Containers
    nginx
      Image	nginx
      Port	80TCP
      QoS Tier
        memory	Guaranteed
        cpu	Guaranteed
      Limits
        cpu	1
        memory	128Mi
      Requests
        cpu	1
        memory	128Mi
      Environment Variables
  Volumes
    default-token-4bcbi
      Type	Secret (a volume populated by a Secret)
      SecretName	default-token-4bcbi
  Events
    FirstSeen	LastSeen	Count	From			        SubobjectPath	Type		Reason			    Message
    ---------	--------	-----	----			        -------------	--------	------			    -------
    1m		    48s		    7	    default-scheduler 			        Warning		FailedScheduling	pod (nginx-deployment-1370807587-fz9sd) failed to fit in any node
  fit failure on node (kubernetes-node-6ta5) Node didnt have enough resource CPU, requested 1000, used 1420, capacity 2000
  fit failure on node (kubernetes-node-wul5) Node didnt have enough resource CPU, requested 1000, used 1100, capacity 2000
```

Here you can see the event generated by the scheduler saying that the Pod failed to schedule for reason `FailedScheduling` (and possibly others).  The message tells us that there were not enough resources for the Pod on any of the nodes.

To correct this situation, you can use `kubectl scale` to update your Deployment to specify four or fewer replicas. (Or you could leave the one Pod pending, which is harmless.)

Events such as the ones you saw at the end of `kubectl describe pod` are persisted in etcd and provide high-level information on what is happening in the cluster. To list all events you can use

```shell
kubectl get events
```

but you have to remember that events are namespaced. This means that if youre interested in events for some namespaced object (e.g. what happened with Pods in namespace `my-namespace`) you need to explicitly provide a namespace to the command

```shell
kubectl get events --namespacemy-namespace
```

To see events from all namespaces, you can use the `--all-namespaces` argument.

In addition to `kubectl describe pod`, another way to get extra information about a pod (beyond what is provided by `kubectl get pod`) is to pass the `-o yaml` output format flag to `kubectl get pod`. This will give you, in YAML format, even more information than `kubectl describe pod`--essentially all of the information the system has about the Pod. Here you will see things like annotations (which are key-value metadata without the label restrictions, that is used internally by Kubernetes system components), restart policy, ports, and volumes.

```shell
kubectl get pod nginx-deployment-1006230814-6winp -o yaml
```

```yaml
apiVersion v1
kind Pod
metadata
  creationTimestamp 2022-02-17T215101Z
  generateName nginx-deployment-67d4bdd6f5-
  labels
    app nginx
    pod-template-hash 67d4bdd6f5
  name nginx-deployment-67d4bdd6f5-w6kd7
  namespace default
  ownerReferences
  - apiVersion appsv1
    blockOwnerDeletion true
    controller true
    kind ReplicaSet
    name nginx-deployment-67d4bdd6f5
    uid 7d41dfd4-84c0-4be4-88ab-cedbe626ad82
  resourceVersion 1364
  uid a6501da1-0447-4262-98eb-c03d4002222e
spec
  containers
  - image nginx
    imagePullPolicy Always
    name nginx
    ports
    - containerPort 80
      protocol TCP
    resources
      limits
        cpu 500m
        memory 128Mi
      requests
        cpu 500m
        memory 128Mi
    terminationMessagePath devtermination-log
    terminationMessagePolicy File
    volumeMounts
    - mountPath varrunsecretskubernetes.ioserviceaccount
      name kube-api-access-bgsgp
      readOnly true
  dnsPolicy ClusterFirst
  enableServiceLinks true
  nodeName kube-worker-1
  preemptionPolicy PreemptLowerPriority
  priority 0
  restartPolicy Always
  schedulerName default-scheduler
  securityContext
  serviceAccount default
  serviceAccountName default
  terminationGracePeriodSeconds 30
  tolerations
  - effect NoExecute
    key node.kubernetes.ionot-ready
    operator Exists
    tolerationSeconds 300
  - effect NoExecute
    key node.kubernetes.iounreachable
    operator Exists
    tolerationSeconds 300
  volumes
  - name kube-api-access-bgsgp
    projected
      defaultMode 420
      sources
      - serviceAccountToken
          expirationSeconds 3607
          path token
      - configMap
          items
          - key ca.crt
            path ca.crt
          name kube-root-ca.crt
      - downwardAPI
          items
          - fieldRef
              apiVersion v1
              fieldPath metadata.namespace
            path namespace
status
  conditions
  - lastProbeTime null
    lastTransitionTime 2022-02-17T215101Z
    status True
    type Initialized
  - lastProbeTime null
    lastTransitionTime 2022-02-17T215106Z
    status True
    type Ready
  - lastProbeTime null
    lastTransitionTime 2022-02-17T215106Z
    status True
    type ContainersReady
  - lastProbeTime null
    lastTransitionTime 2022-02-17T215101Z
    status True
    type PodScheduled
  containerStatuses
  - containerID containerd5403af59a2b46ee5a23fb0ae4b1e077f7ca5c5fb7af16e1ab21c00e0e616462a
    image docker.iolibrarynginxlatest
    imageID docker.iolibrarynginxsha2562834dc507516af02784808c5f48b7cbe38b8ed5d0f4837f16e78d00deb7e7767
    lastState
    name nginx
    ready true
    restartCount 0
    started true
    state
      running
        startedAt 2022-02-17T215105Z
  hostIP 192.168.0.113
  phase Running
  podIP 10.88.0.3
  podIPs
  - ip 10.88.0.3
  - ip 2001db81
  qosClass Guaranteed
  startTime 2022-02-17T215101Z
```

# # Examining pod logs #examine-pod-logs

First, look at the logs of the affected container

```shell
kubectl logs POD_NAME CONTAINER_NAME
```

If your container has previously crashed, you can access the previous containers crash log with

```shell
kubectl logs --previous POD_NAME CONTAINER_NAME
```

# # Debugging with container exec #container-exec

If the  includes
debugging utilities, as is the case with images built from Linux and Windows OS
base images, you can run commands inside a specific container with
`kubectl exec`

```shell
kubectl exec POD_NAME -c CONTAINER_NAME -- CMD ARG1 ARG2 ... ARGN
```

`-c CONTAINER_NAME` is optional. You can omit it for Pods that only contain a single container.

As an example, to look at the logs from a running Cassandra pod, you might run

```shell
kubectl exec cassandra -- cat varlogcassandrasystem.log
```

You can run a shell thats connected to your terminal using the `-i` and `-t`
arguments to `kubectl exec`, for example

```shell
kubectl exec -it cassandra -- sh
```

For more details, see [Get a Shell to a Running Container](
docstasksdebugdebug-applicationget-shell-running-container).

# # Debugging with an ephemeral debug container #ephemeral-container

are useful for interactive troubleshooting when `kubectl exec` is insufficient
because a container has crashed or a container image doesnt include debugging
utilities, such as with [distroless images](
httpsgithub.comGoogleContainerToolsdistroless).

# # # Example debugging using ephemeral containers #ephemeral-container-example

You can use the `kubectl debug` command to add ephemeral containers to a
running Pod. First, create a pod for the example

```shell
kubectl run ephemeral-demo --imageregistry.k8s.iopause3.1 --restartNever
```

The examples in this section use the `pause` container image because it does not
contain debugging utilities, but this method works with all container
images.

If you attempt to use `kubectl exec` to create a shell you will see an error
because there is no shell in this container image.

```shell
kubectl exec -it ephemeral-demo -- sh
```

```
OCI runtime exec failed exec failed container_linux.go346 starting container process caused exec sh executable file not found in PATH unknown
```

You can instead add a debugging container using `kubectl debug`. If you
specify the `-i``--interactive` argument, `kubectl` will automatically attach
to the console of the Ephemeral Container.

```shell
kubectl debug -it ephemeral-demo --imagebusybox1.28 --targetephemeral-demo
```

```
Defaulting debug container name to debugger-8xzrl.
If you dont see a command prompt, try pressing enter.
 #
```

This command adds a new busybox container and attaches to it. The `--target`
parameter targets the process namespace of another container. Its necessary
here because `kubectl run` does not enable [process namespace sharing](
docstasksconfigure-pod-containershare-process-namespace) in the pod it
creates.

The `--target` parameter must be supported by the . When not supported,
the Ephemeral Container may not be started, or it may be started with an
isolated process namespace so that `ps` does not reveal processes in other
containers.

You can view the state of the newly created ephemeral container using `kubectl describe`

```shell
kubectl describe pod ephemeral-demo
```

```
...
Ephemeral Containers
  debugger-8xzrl
    Container ID   dockerb888f9adfd15bd5739fefaa39e1df4dd3c617b9902082b1cfdc29c4028ffb2eb
    Image          busybox
    Image ID       docker-pullablebusyboxsha2561828edd60c5efd34b2bf5dd3282ec0cc04d47b2ff9caa0b6d4f07a21d1c08084
    Port
    Host Port
    State          Running
      Started      Wed, 12 Feb 2020 142542 0100
    Ready          False
    Restart Count  0
    Environment
    Mounts
...
```

Use `kubectl delete` to remove the Pod when youre finished

```shell
kubectl delete pod ephemeral-demo
```

# # Debugging using a copy of the Pod

Sometimes Pod configuration options make it difficult to troubleshoot in certain
situations. For example, you cant run `kubectl exec` to troubleshoot your
container if your container image does not include a shell or if your application
crashes on startup. In these situations you can use `kubectl debug` to create a
copy of the Pod with configuration values changed to aid debugging.

# # # Copying a Pod while adding a new container

Adding a new container can be useful when your application is running but not
behaving as you expect and youd like to add additional troubleshooting
utilities to the Pod.

For example, maybe your applications container images are built on `busybox`
but you need debugging utilities not included in `busybox`. You can simulate
this scenario using `kubectl run`

```shell
kubectl run myapp --imagebusybox1.28 --restartNever -- sleep 1d
```

Run this command to create a copy of `myapp` named `myapp-debug` that adds a
new Ubuntu container for debugging

```shell
kubectl debug myapp -it --imageubuntu --share-processes --copy-tomyapp-debug
```

```
Defaulting debug container name to debugger-w7xmf.
If you dont see a command prompt, try pressing enter.
rootmyapp-debug#
```

* `kubectl debug` automatically generates a container name if you dont choose
  one using the `--container` flag.
* The `-i` flag causes `kubectl debug` to attach to the new container by
  default.  You can prevent this by specifying `--attachfalse`. If your session
  becomes disconnected you can reattach using `kubectl attach`.
* The `--share-processes` allows the containers in this Pod to see processes
  from the other containers in the Pod. For more information about how this
  works, see [Share Process Namespace between Containers in a Pod](
  docstasksconfigure-pod-containershare-process-namespace).

Dont forget to clean up the debugging Pod when youre finished with it

```shell
kubectl delete pod myapp myapp-debug
```

# # # Copying a Pod while changing its command

Sometimes its useful to change the command for a container, for example to
add a debugging flag or because the application is crashing.

To simulate a crashing application, use `kubectl run` to create a container
that immediately exits

```
kubectl run --imagebusybox1.28 myapp -- false
```

You can see using `kubectl describe pod myapp` that this container is crashing

```
Containers
  myapp
    Image         busybox
    ...
    Args
      false
    State          Waiting
      Reason       CrashLoopBackOff
    Last State     Terminated
      Reason       Error
      Exit Code    1
```

You can use `kubectl debug` to create a copy of this Pod with the command
changed to an interactive shell

```
kubectl debug myapp -it --copy-tomyapp-debug --containermyapp -- sh
```

```
If you dont see a command prompt, try pressing enter.
 #
```

Now you have an interactive shell that you can use to perform tasks like
checking filesystem paths or running the container command manually.

* To change the command of a specific container you must
  specify its name using `--container` or `kubectl debug` will instead
  create a new container to run the command you specified.
* The `-i` flag causes `kubectl debug` to attach to the container by default.
  You can prevent this by specifying `--attachfalse`. If your session becomes
  disconnected you can reattach using `kubectl attach`.

Dont forget to clean up the debugging Pod when youre finished with it

```shell
kubectl delete pod myapp myapp-debug
```

# # # Copying a Pod while changing container images

In some situations you may want to change a misbehaving Pod from its normal
production container images to an image containing a debugging build or
additional utilities.

As an example, create a Pod using `kubectl run`

```
kubectl run myapp --imagebusybox1.28 --restartNever -- sleep 1d
```

Now use `kubectl debug` to make a copy and change its container image
to `ubuntu`

```
kubectl debug myapp --copy-tomyapp-debug --set-image*ubuntu
```

The syntax of `--set-image` uses the same `container_nameimage` syntax as
`kubectl set image`. `*ubuntu` means change the image of all containers
to `ubuntu`.

Dont forget to clean up the debugging Pod when youre finished with it

```shell
kubectl delete pod myapp myapp-debug
```

# # Debugging via a shell on the node #node-shell-session

If none of these approaches work, you can find the Node on which the Pod is
running and create a Pod running on the Node. To create
an interactive shell on a Node using `kubectl debug`, run

```shell
kubectl debug nodemynode -it --imageubuntu
```

```
Creating debugging pod node-debugger-mynode-pdx84 with container debugger on node mynode.
If you dont see a command prompt, try pressing enter.
rootek8s#
```

When creating a debugging session on a node, keep in mind that

* `kubectl debug` automatically generates the name of the new Pod based on
  the name of the Node.
* The root filesystem of the Node will be mounted at `host`.
* The container runs in the host IPC, Network, and PID namespaces, although
  the pod isnt privileged, so reading some process information may fail,
  and `chroot host` may fail.
* If you need a privileged pod, create it manually or use the `--profilesysadmin` flag.

Dont forget to clean up the debugging Pod when youre finished with it

```shell
kubectl delete pod node-debugger-mynode-pdx84
```

# # Debugging a Pod or Node while applying a profile #debugging-profiles

When using `kubectl debug` to debug a node via a debugging Pod, a Pod via an ephemeral container,
or a copied Pod, you can apply a profile to them.
By applying a profile, specific properties such as [securityContext](docstasksconfigure-pod-containersecurity-context)
are set, allowing for adaptation to various scenarios.
There are two types of profiles, static profile and custom profile.

# # # Applying a Static Profile #static-profile

A static profile is a set of predefined properties, and you can apply them using the `--profile` flag.
The available profiles are as follows

 Profile       Description
 ------------  ---------------------------------------------------------------
 legacy        A set of properties backwards compatibility with 1.22 behavior
 general       A reasonable set of generic properties for each debugging journey
 baseline      A set of properties compatible with [PodSecurityStandard baseline policy](docsconceptssecuritypod-security-standards#baseline)
 restricted    A set of properties compatible with [PodSecurityStandard restricted policy](docsconceptssecuritypod-security-standards#restricted)
 netadmin      A set of properties including Network Administrator privileges
 sysadmin      A set of properties including System Administrator (root) privileges

If you dont specify `--profile`, the `legacy` profile is used by default, but it is planned to be deprecated in the near future.
So it is recommended to use other profiles such as `general`.

Assume that you create a Pod and debug it.
First, create a Pod named `myapp` as an example

```shell
kubectl run myapp --imagebusybox1.28 --restartNever -- sleep 1d
```

Then, debug the Pod using an ephemeral container.
If the ephemeral container needs to have privilege, you can use the `sysadmin` profile

```shell
kubectl debug -it myapp --imagebusybox1.28 --targetmyapp --profilesysadmin
```

```
Targeting container myapp. If you dont see processes from this container it may be because the container runtime doesnt support this feature.
Defaulting debug container name to debugger-6kg4x.
If you dont see a command prompt, try pressing enter.
 #
```

Check the capabilities of the ephemeral container process by running the following command inside the container

```shell
 # grep Cap procstatus
```

```
...
CapPrm	000001ffffffffff
CapEff	000001ffffffffff
...
```

This means the container process is granted full capabilities as a privileged container by applying `sysadmin` profile.
See more details about [capabilities](docstasksconfigure-pod-containersecurity-context#set-capabilities-for-a-container).

You can also check that the ephemeral container was created as a privileged container

```shell
kubectl get pod myapp -o jsonpath.spec.ephemeralContainers[0].securityContext
```

```
privilegedtrue
```

Clean up the Pod when youre finished with it

```shell
kubectl delete pod myapp
```

# # # Applying Custom Profile #custom-profile

You can define a partial container spec for debugging as a custom profile in either YAML or JSON format,
and apply it using the `--custom` flag.

Custom profile only supports the modification of the container spec,
but modifications to `name`, `image`, `command`, `lifecycle` and `volumeDevices` fields of the container spec
are not allowed.
It does not support the modification of the Pod spec.

Create a Pod named myapp as an example

```shell
kubectl run myapp --imagebusybox1.28 --restartNever -- sleep 1d
```

Create a custom profile in YAML or JSON format.
Here, create a YAML format file named `custom-profile.yaml`

```yaml
env
- name ENV_VAR_1
  value value_1
- name ENV_VAR_2
  value value_2
securityContext
  capabilities
    add
    - NET_ADMIN
    - SYS_TIME

```

Run this command to debug the Pod using an ephemeral container with the custom profile

```shell
kubectl debug -it myapp --imagebusybox1.28 --targetmyapp --profilegeneral --customcustom-profile.yaml
```

You can check that the ephemeral container has been added to the target Pod with the custom profile applied

```shell
kubectl get pod myapp -o jsonpath.spec.ephemeralContainers[0].env
```

```
[nameENV_VAR_1,valuevalue_1,nameENV_VAR_2,valuevalue_2]
```

```shell
kubectl get pod myapp -o jsonpath.spec.ephemeralContainers[0].securityContext
```

```
capabilitiesadd[NET_ADMIN,SYS_TIME]
```

Clean up the Pod when youre finished with it

```shell
kubectl delete pod myapp
```