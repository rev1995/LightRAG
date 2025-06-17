---
title Update API Objects in Place Using kubectl patch
description Use kubectl patch to update Kubernetes API objects in place. Do a strategic merge patch or a JSON merge patch.
content_type task
weight 50
---

This task shows how to use `kubectl patch` to update an API object in place. The exercises
in this task demonstrate a strategic merge patch and a JSON merge patch.

# #  heading prerequisites

# # Use a strategic merge patch to update a Deployment

Heres the configuration file for a Deployment that has two replicas. Each replica
is a Pod that has one container

 code_sample fileapplicationdeployment-patch.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationdeployment-patch.yaml
```

View the Pods associated with your Deployment

```shell
kubectl get pods
```

The output shows that the Deployment has two Pods. The `11` indicates that
each Pod has one container

```
NAME                        READY     STATUS    RESTARTS   AGE
patch-demo-28633765-670qr   11       Running   0          23s
patch-demo-28633765-j5qs3   11       Running   0          23s
```

Make a note of the names of the running Pods. Later, you will see that these Pods
get terminated and replaced by new ones.

At this point, each Pod has one Container that runs the nginx image. Now suppose
you want each Pod to have two containers one that runs nginx and one that runs redis.

Create a file named `patch-file.yaml` that has this content

```yaml
spec
  template
    spec
      containers
      - name patch-demo-ctr-2
        image redis
```

Patch your Deployment

```shell
kubectl patch deployment patch-demo --patch-file patch-file.yaml
```

View the patched Deployment

```shell
kubectl get deployment patch-demo --output yaml
```

The output shows that the PodSpec in the Deployment has two Containers

```yaml
containers
- image redis
  imagePullPolicy Always
  name patch-demo-ctr-2
  ...
- image nginx
  imagePullPolicy Always
  name patch-demo-ctr
  ...
```

View the Pods associated with your patched Deployment

```shell
kubectl get pods
```

The output shows that the running Pods have different names from the Pods that
were running previously. The Deployment terminated the old Pods and created two
new Pods that comply with the updated Deployment spec. The `22` indicates that
each Pod has two Containers

```
NAME                          READY     STATUS    RESTARTS   AGE
patch-demo-1081991389-2wrn5   22       Running   0          1m
patch-demo-1081991389-jmg7b   22       Running   0          1m
```

Take a closer look at one of the patch-demo Pods

```shell
kubectl get pod  --output yaml
```

The output shows that the Pod has two Containers one running nginx and one running redis

```
containers
- image redis
  ...
- image nginx
  ...
```

# # # Notes on the strategic merge patch

The patch you did in the preceding exercise is called a *strategic merge patch*.
Notice that the patch did not replace the `containers` list. Instead it added a new
Container to the list. In other words, the list in the patch was merged with the
existing list. This is not always what happens when you use a strategic merge patch on a list.
In some cases, the list is replaced, not merged.

With a strategic merge patch, a list is either replaced or merged depending on its
patch strategy. The patch strategy is specified by the value of the `patchStrategy` key
in a field tag in the Kubernetes source code. For example, the `Containers` field of `PodSpec`
struct has a `patchStrategy` of `merge`

```go
type PodSpec struct
  ...
  Containers []Container `jsoncontainers patchStrategymerge patchMergeKeyname ...`
  ...

```

You can also see the patch strategy in the
[OpenApi spec](httpsraw.githubusercontent.comkuberneteskubernetesmasterapiopenapi-specswagger.json)

```yaml
io.k8s.api.core.v1.PodSpec
    ...,
    containers
        description List of containers belonging to the pod.  ....
    ,
    x-kubernetes-patch-merge-key name,
    x-kubernetes-patch-strategy merge

```

And you can see the patch strategy in the
[Kubernetes API documentation](docsreferencegeneratedkubernetes-api#podspec-v1-core).

Create a file named `patch-file-tolerations.yaml` that has this content

```yaml
spec
  template
    spec
      tolerations
      - effect NoSchedule
        key disktype
        value ssd
```

Patch your Deployment

```shell
kubectl patch deployment patch-demo --patch-file patch-file-tolerations.yaml
```

View the patched Deployment

```shell
kubectl get deployment patch-demo --output yaml
```

The output shows that the PodSpec in the Deployment has only one Toleration

```yaml
tolerations
- effect NoSchedule
  key disktype
  value ssd
```

Notice that the `tolerations` list in the PodSpec was replaced, not merged. This is because
the Tolerations field of PodSpec does not have a `patchStrategy` key in its field tag. So the
strategic merge patch uses the default patch strategy, which is `replace`.

```go
type PodSpec struct
  ...
  Tolerations []Toleration `jsontolerations,omitempty protobufbytes,22,opt,nametolerations`
  ...

```

# # Use a JSON merge patch to update a Deployment

A strategic merge patch is different from a
[JSON merge patch](httpstools.ietf.orghtmlrfc7386).
With a JSON merge patch, if you
want to update a list, you have to specify the entire new list. And the new list completely
replaces the existing list.

The `kubectl patch` command has a `type` parameter that you can set to one of these values

  Parameter valueMerge type
  jsonJSON Patch, RFC 6902
  mergeJSON Merge Patch, RFC 7386
  strategicStrategic merge patch

For a comparison of JSON patch and JSON merge patch, see
[JSON Patch and JSON Merge Patch](httpserosb.github.iopostjson-patch-vs-merge-patch).

The default value for the `type` parameter is `strategic`. So in the preceding exercise, you
did a strategic merge patch.

Next, do a JSON merge patch on your same Deployment. Create a file named `patch-file-2.yaml`
that has this content

```yaml
spec
  template
    spec
      containers
      - name patch-demo-ctr-3
        image gcr.iogoogle-sampleshello-app2.0
```

In your patch command, set `type` to `merge`

```shell
kubectl patch deployment patch-demo --type merge --patch-file patch-file-2.yaml
```

View the patched Deployment

```shell
kubectl get deployment patch-demo --output yaml
```

The `containers` list that you specified in the patch has only one Container.
The output shows that your list of one Container replaced the existing `containers` list.

```yaml
spec
  containers
  - image gcr.iogoogle-sampleshello-app2.0
    ...
    name patch-demo-ctr-3
```

List the running Pods

```shell
kubectl get pods
```

In the output, you can see that the existing Pods were terminated, and new Pods
were created. The `11` indicates that each new Pod is running only one Container.

```shell
NAME                          READY     STATUS    RESTARTS   AGE
patch-demo-1307768864-69308   11       Running   0          1m
patch-demo-1307768864-c86dc   11       Running   0          1m
```

# # Use strategic merge patch to update a Deployment using the retainKeys strategy

Heres the configuration file for a Deployment that uses the `RollingUpdate` strategy

 code_sample fileapplicationdeployment-retainkeys.yaml

Create the deployment

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationdeployment-retainkeys.yaml
```

At this point, the deployment is created and is using the `RollingUpdate` strategy.

Create a file named `patch-file-no-retainkeys.yaml` that has this content

```yaml
spec
  strategy
    type Recreate
```

Patch your Deployment

```shell
kubectl patch deployment retainkeys-demo --type strategic --patch-file patch-file-no-retainkeys.yaml
```

In the output, you can see that it is not possible to set `type` as `Recreate` when a value is defined for `spec.strategy.rollingUpdate`

```
The Deployment retainkeys-demo is invalid spec.strategy.rollingUpdate Forbidden may not be specified when strategy `type` is Recreate
```

The way to remove the value for `spec.strategy.rollingUpdate` when updating the value for `type` is to use the `retainKeys` strategy for the strategic merge.

Create another file named `patch-file-retainkeys.yaml` that has this content

```yaml
spec
  strategy
    retainKeys
    - type
    type Recreate
```

With this patch, we indicate that we want to retain only the `type` key of the `strategy` object. Thus, the `rollingUpdate` will be removed during the patch operation.

Patch your Deployment again with this new patch

```shell
kubectl patch deployment retainkeys-demo --type strategic --patch-file patch-file-retainkeys.yaml
```

Examine the content of the Deployment

```shell
kubectl get deployment retainkeys-demo --output yaml
```

The output shows that the strategy object in the Deployment does not contain the `rollingUpdate` key anymore

```yaml
spec
  strategy
    type Recreate
  template
```

# # # Notes on the strategic merge patch using the retainKeys strategy

The patch you did in the preceding exercise is called a *strategic merge patch with retainKeys strategy*. This method introduces a new directive `retainKeys` that has the following strategies

- It contains a list of strings.
- All fields needing to be preserved must be present in the `retainKeys` list.
- The fields that are present will be merged with live object.
- All of the missing fields will be cleared when patching.
- All fields in the `retainKeys` list must be a superset or the same as the fields present in the patch.

The `retainKeys` strategy does not work for all objects. It only works when the value of the `patchStrategy` key in a field tag in the Kubernetes source code contains `retainKeys`. For example, the `Strategy` field of the `DeploymentSpec` struct has a `patchStrategy` of `retainKeys`

```go
type DeploymentSpec struct
  ...
   patchStrategyretainKeys
  Strategy DeploymentStrategy `jsonstrategy,omitempty patchStrategyretainKeys ...`
  ...

```

You can also see the `retainKeys` strategy in the [OpenApi spec](httpsraw.githubusercontent.comkuberneteskubernetesmasterapiopenapi-specswagger.json)

```yaml
io.k8s.api.apps.v1.DeploymentSpec
    ...,
    strategy
        ref #definitionsio.k8s.api.apps.v1.DeploymentStrategy,
        description The deployment strategy to use to replace existing pods with new ones.,
        x-kubernetes-patch-strategy retainKeys
    ,
    ....

```

And you can see the `retainKeys` strategy in the
[Kubernetes API documentation](docsreferencegeneratedkubernetes-api#deploymentspec-v1-apps).

# # # Alternate forms of the kubectl patch command

The `kubectl patch` command takes YAML or JSON. It can take the patch as a file or
directly on the command line.

Create a file named `patch-file.json` that has this content

```json

   spec
      template
         spec
            containers [

                  name patch-demo-ctr-2,
                  image redis

            ]

```

The following commands are equivalent

```shell
kubectl patch deployment patch-demo --patch-file patch-file.yaml
kubectl patch deployment patch-demo --patch specn templaten  specn   containersn   - name patch-demo-ctr-2n     image redis

kubectl patch deployment patch-demo --patch-file patch-file.json
kubectl patch deployment patch-demo --patch spec template spec containers [name patch-demo-ctr-2,image redis]
```

# # # Update an objects replica count using `kubectl patch` with `--subresource` #scale-kubectl-patch

The flag `--subresource[subresource-name]` is used with kubectl commands like get, patch,
edit, apply and replace to fetch and update `status`, `scale` and `resize` subresource of the
resources you specify. You can specify a subresource for any of the Kubernetes API resources
(built-in and CRs) that have `status`, `scale` or `resize` subresource.

For example, a Deployment has a `status` subresource and a `scale` subresource, so you can
use `kubectl` to get or modify just the `status` subresource of a Deployment.

Heres a manifest for a Deployment that has two replicas

 code_sample fileapplicationdeployment.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationdeployment.yaml
```

View the Pods associated with your Deployment

```shell
kubectl get pods -l appnginx
```

In the output, you can see that Deployment has two Pods. For example

```
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-7fb96c846b-22567   11     Running   0          47s
nginx-deployment-7fb96c846b-mlgns   11     Running   0          47s
```

Now, patch that Deployment with `--subresource[subresource-name]` flag

```shell
kubectl patch deployment nginx-deployment --subresourcescale --typemerge -p specreplicas3
```

The output is

```shell
scale.autoscalingnginx-deployment patched
```

View the Pods associated with your patched Deployment

```shell
kubectl get pods -l appnginx
```

In the output, you can see one new pod is created, so now you have 3 running pods.

```
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-7fb96c846b-22567   11     Running   0          107s
nginx-deployment-7fb96c846b-lxfr2   11     Running   0          14s
nginx-deployment-7fb96c846b-mlgns   11     Running   0          107s
```

View the patched Deployment

```shell
kubectl get deployment nginx-deployment -o yaml
```

```yaml
...
spec
  replicas 3
  ...
status
  ...
  availableReplicas 3
  readyReplicas 3
  replicas 3
```

If you run `kubectl patch` and specify `--subresource` flag for resource that doesnt support that
particular subresource, the API server returns a 404 Not Found error.

# # Summary

In this exercise, you used `kubectl patch` to change the live configuration
of a Deployment object. You did not change the configuration file that you originally used to
create the Deployment object. Other commands for updating API objects include
[kubectl annotate](docsreferencegeneratedkubectlkubectl-commands#annotate),
[kubectl edit](docsreferencegeneratedkubectlkubectl-commands#edit),
[kubectl replace](docsreferencegeneratedkubectlkubectl-commands#replace),
[kubectl scale](docsreferencegeneratedkubectlkubectl-commands#scale),
and
[kubectl apply](docsreferencegeneratedkubectlkubectl-commands#apply).

Strategic merge patch is not supported for custom resources.

# #  heading whatsnext

* [Kubernetes Object Management](docsconceptsoverviewworking-with-objectsobject-management)
* [Managing Kubernetes Objects Using Imperative Commands](docstasksmanage-kubernetes-objectsimperative-command)
* [Imperative Management of Kubernetes Objects Using Configuration Files](docstasksmanage-kubernetes-objectsimperative-config)
* [Declarative Management of Kubernetes Objects Using Configuration Files](docstasksmanage-kubernetes-objectsdeclarative-config)
