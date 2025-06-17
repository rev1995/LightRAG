---
title Updating Configuration via a ConfigMap
content_type tutorial
weight 20
---

This page provides a step-by-step example of updating configuration within a Pod via a ConfigMap
and builds upon the [Configure a Pod to Use a ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap) task.
At the end of this tutorial, you will understand how to change the configuration for a running application.
This tutorial uses the `alpine` and `nginx` images as examples.

# #  heading prerequisites

You need to have the [curl](httpscurl.se) command-line tool for making HTTP requests from
the terminal or command prompt. If you do not have `curl` available, you can install it. Check the
documentation for your local operating system.

# #  heading objectives
* Update configuration via a ConfigMap mounted as a Volume
* Update environment variables of a Pod via a ConfigMap
* Update configuration via a ConfigMap in a multi-container Pod
* Update configuration via a ConfigMap in a Pod possessing a Sidecar Container

# # Update configuration via a ConfigMap mounted as a Volume #rollout-configmap-volume

Use the `kubectl create configmap` command to create a ConfigMap from
[literal values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell
kubectl create configmap sport --from-literalsportfootball
```

Below is an example of a Deployment manifest with the ConfigMap `sport` mounted as a
 into the Pods only container.
 code_sample filedeploymentsdeployment-with-configmap-as-volume.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-as-volume.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell
kubectl get pods --selectorapp.kubernetes.ionameconfigmap-volume
```

You should see an output similar to

```
NAME                                READY   STATUS    RESTARTS   AGE
configmap-volume-6b976dfdcf-qxvbm   11     Running   0          72s
configmap-volume-6b976dfdcf-skpvm   11     Running   0          72s
configmap-volume-6b976dfdcf-tbc6r   11     Running   0          72s
```

On each node where one of these Pods is running, the kubelet fetches the data for
that ConfigMap and translates it to files in a local volume.
The kubelet then mounts that volume into the container, as specified in the Pod template.
The code running in that container loads the information from the file
and uses it to print a report to stdout.
You can check this report by viewing the logs for one of the Pods in that Deployment

```shell
# Pick one Pod that belongs to the Deployment, and view its logs
kubectl logs deploymentsconfigmap-volume
```

You should see an output similar to

```
Found 3 pods, using podconfigmap-volume-76d9c5678f-x5rgj
Thu Jan  4 140646 UTC 2024 My preferred sport is football
Thu Jan  4 140656 UTC 2024 My preferred sport is football
Thu Jan  4 140706 UTC 2024 My preferred sport is football
Thu Jan  4 140716 UTC 2024 My preferred sport is football
Thu Jan  4 140726 UTC 2024 My preferred sport is football
```

Edit the ConfigMap

```shell
kubectl edit configmap sport
```

In the editor that appears, change the value of key `sport` from `football` to `cricket`. Save your changes.
The kubectl tool updates the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml
apiVersion v1
data
  sport cricket
kind ConfigMap
# You can leave the existing metadata as they are.
# The values youll see wont exactly match these.
metadata
  creationTimestamp 2024-01-04T140506Z
  name sport
  namespace default
  resourceVersion 1743935
  uid 024ee001-fe72-487e-872e-34d6464a8a23
```

You should see the following output

```
configmapsport edited
```

Tail (follow the latest entries in) the logs of one of the pods that belongs to this Deployment

```shell
kubectl logs deploymentsconfigmap-volume --follow
```

After few seconds, you should see the log output change as follows

```
Thu Jan  4 141136 UTC 2024 My preferred sport is football
Thu Jan  4 141146 UTC 2024 My preferred sport is football
Thu Jan  4 141156 UTC 2024 My preferred sport is football
Thu Jan  4 141206 UTC 2024 My preferred sport is cricket
Thu Jan  4 141216 UTC 2024 My preferred sport is cricket
```

When you have a ConfigMap that is mapped into a running Pod using either a
`configMap` volume or a `projected` volume, and you update that ConfigMap,
the running Pod sees the update almost immediately.
However, your application only sees the change if it is written to either poll for changes,
or watch for file updates.
An application that loads its configuration once at startup will not notice a change.

The total delay from the moment when the ConfigMap is updated to the moment when
new keys are projected to the Pod can be as long as kubelet sync period.
Also check [Mounted ConfigMaps are updated automatically](docstasksconfigure-pod-containerconfigure-pod-configmap#mounted-configmaps-are-updated-automatically).

# # Update environment variables of a Pod via a ConfigMap #rollout-configmap-env

Use the `kubectl create configmap` command to create a ConfigMap from
[literal values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell
kubectl create configmap fruits --from-literalfruitsapples
```

Below is an example of a Deployment manifest with an environment variable configured via the ConfigMap `fruits`.

 code_sample filedeploymentsdeployment-with-configmap-as-envvar.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-as-envvar.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell
kubectl get pods --selectorapp.kubernetes.ionameconfigmap-env-var
```

You should see an output similar to

```
NAME                                 READY   STATUS    RESTARTS   AGE
configmap-env-var-59cfc64f7d-74d7z   11     Running   0          46s
configmap-env-var-59cfc64f7d-c4wmj   11     Running   0          46s
configmap-env-var-59cfc64f7d-dpr98   11     Running   0          46s
```

The key-value pair in the ConfigMap is configured as an environment variable in the container of the Pod.
Check this by viewing the logs of one Pod that belongs to the Deployment.

```shell
kubectl logs deploymentconfigmap-env-var
```

You should see an output similar to

```
Found 3 pods, using podconfigmap-env-var-7c994f7769-l74nq
Thu Jan  4 160706 UTC 2024 The basket is full of apples
Thu Jan  4 160716 UTC 2024 The basket is full of apples
Thu Jan  4 160726 UTC 2024 The basket is full of apples
```

Edit the ConfigMap

```shell
kubectl edit configmap fruits
```

In the editor that appears, change the value of key `fruits` from `apples` to `mangoes`. Save your changes.
The kubectl tool updates the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml
apiVersion v1
data
  fruits mangoes
kind ConfigMap
# You can leave the existing metadata as they are.
# The values youll see wont exactly match these.
metadata
  creationTimestamp 2024-01-04T160419Z
  name fruits
  namespace default
  resourceVersion 1749472
```

You should see the following output

```
configmapfruits edited
```

Tail the logs of the Deployment and observe the output for few seconds

```shell
# As the text explains, the output does NOT change
kubectl logs deploymentsconfigmap-env-var --follow
```

Notice that the output remains **unchanged**, even though you edited the ConfigMap

```
Thu Jan  4 161256 UTC 2024 The basket is full of apples
Thu Jan  4 161306 UTC 2024 The basket is full of apples
Thu Jan  4 161316 UTC 2024 The basket is full of apples
Thu Jan  4 161326 UTC 2024 The basket is full of apples
```

Although the value of the key inside the ConfigMap has changed, the environment variable
in the Pod still shows the earlier value. This is because environment variables for a
process running inside a Pod are **not** updated when the source data changes if you
wanted to force an update, you would need to have Kubernetes replace your existing Pods.
The new Pods would then run with the updated information.

You can trigger that replacement. Perform a rollout for the Deployment, using
[`kubectl rollout`](docsreferencekubectlgeneratedkubectl_rollout)

```shell
# Trigger the rollout
kubectl rollout restart deployment configmap-env-var

# Wait for the rollout to complete
kubectl rollout status deployment configmap-env-var --watchtrue
```

Next, check the Deployment

```shell
kubectl get deployment configmap-env-var
```

You should see an output similar to

```
NAME                READY   UP-TO-DATE   AVAILABLE   AGE
configmap-env-var   33     3            3           12m
```

Check the Pods

```shell
kubectl get pods --selectorapp.kubernetes.ionameconfigmap-env-var
```

The rollout causes Kubernetes to make a new
for the Deployment that means the existing Pods eventually terminate, and new ones are created.
After few seconds, you should see an output similar to

```
NAME                                 READY   STATUS        RESTARTS   AGE
configmap-env-var-6d94d89bf5-2ph2l   11     Running       0          13s
configmap-env-var-6d94d89bf5-74twx   11     Running       0          8s
configmap-env-var-6d94d89bf5-d5vx8   11     Running       0          11s
```

Please wait for the older Pods to fully terminate before proceeding with the next steps.

View the logs for a Pod in this Deployment

```shell
# Pick one Pod that belongs to the Deployment, and view its logs
kubectl logs deploymentconfigmap-env-var
```

You should see an output similar to the below

```
Found 3 pods, using podconfigmap-env-var-6d9ff89fb6-bzcf6
Thu Jan  4 163035 UTC 2024 The basket is full of mangoes
Thu Jan  4 163045 UTC 2024 The basket is full of mangoes
Thu Jan  4 163055 UTC 2024 The basket is full of mangoes
```

This demonstrates the scenario of updating environment variables in a Pod that are derived
from a ConfigMap. Changes to the ConfigMap values are applied to the Pod during the subsequent
rollout. If Pods get created for another reason, such as scaling up the Deployment, then the new Pods
also use the latest configuration values if you dont trigger a rollout, then you might find that your
app is running with a mix of old and new environment variable values.

# # Update configuration via a ConfigMap in a multi-container Pod #rollout-configmap-multiple-containers

Use the `kubectl create configmap` command to create a ConfigMap from
[literal values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell
kubectl create configmap color --from-literalcolorred
```

Below is an example manifest for a Deployment that manages a set of Pods, each with two containers.
The two containers share an `emptyDir` volume that they use to communicate.
The first container runs a web server (`nginx`). The mount path for the shared volume in the
web server container is `usrsharenginxhtml`. The second helper container is based on `alpine`,
and for this container the `emptyDir` volume is mounted at `pod-data`. The helper container writes
a file in HTML that has its content based on a ConfigMap. The web server container serves the HTML via HTTP.

 code_sample filedeploymentsdeployment-with-configmap-two-containers.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-two-containers.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell
kubectl get pods --selectorapp.kubernetes.ionameconfigmap-two-containers
```

You should see an output similar to

```
NAME                                        READY   STATUS    RESTARTS   AGE
configmap-two-containers-565fb6d4f4-2xhxf   22     Running   0          20s
configmap-two-containers-565fb6d4f4-g5v4j   22     Running   0          20s
configmap-two-containers-565fb6d4f4-mzsmf   22     Running   0          20s
```

Expose the Deployment (the `kubectl` tool creates a
 for you)

```shell
kubectl expose deployment configmap-two-containers --nameconfigmap-service --port8080 --target-port80
```

Use `kubectl` to forward the port

```shell
# this stays running in the background
kubectl port-forward serviceconfigmap-service 80808080
```

Access the service.

```shell
curl httplocalhost8080
```

You should see an output similar to

```
Fri Jan  5 080822 UTC 2024 My preferred color is red
```

Edit the ConfigMap

```shell
kubectl edit configmap color
```

In the editor that appears, change the value of key `color` from `red` to `blue`. Save your changes.
The kubectl tool updates the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml
apiVersion v1
data
  color blue
kind ConfigMap
# You can leave the existing metadata as they are.
# The values youll see wont exactly match these.
metadata
  creationTimestamp 2024-01-05T081205Z
  name color
  namespace configmap
  resourceVersion 1801272
  uid 80d33e4a-cbb4-4bc9-ba8c-544c68e425d6
```

Loop over the service URL for few seconds.

```shell
# Cancel this when youre happy with it (Ctrl-C)
while true do curl --connect-timeout 7.5 httplocalhost8080 sleep 10 done
```

You should see the output change as follows

```
Fri Jan  5 081400 UTC 2024 My preferred color is red
Fri Jan  5 081402 UTC 2024 My preferred color is red
Fri Jan  5 081420 UTC 2024 My preferred color is red
Fri Jan  5 081422 UTC 2024 My preferred color is red
Fri Jan  5 081432 UTC 2024 My preferred color is blue
Fri Jan  5 081443 UTC 2024 My preferred color is blue
Fri Jan  5 081500 UTC 2024 My preferred color is blue
```

# # Update configuration via a ConfigMap in a Pod possessing a sidecar container #rollout-configmap-sidecar

The above scenario can be replicated by using a [Sidecar Container](docsconceptsworkloadspodssidecar-containers)
as a helper container to write the HTML file.
As a Sidecar Container is conceptually an Init Container, it is guaranteed to start before the main web server container.
This ensures that the HTML file is always available when the web server is ready to serve it.

If you are continuing from the previous scenario, you can reuse the ConfigMap named `color` for this scenario.
If you are executing this scenario independently, use the `kubectl create configmap` command to create a ConfigMap
from [literal values](docstasksconfigure-pod-containerconfigure-pod-configmap#create-configmaps-from-literal-values)

```shell
kubectl create configmap color --from-literalcolorblue
```

Below is an example manifest for a Deployment that manages a set of Pods, each with a main container and
a sidecar container. The two containers share an `emptyDir` volume that they use to communicate.
The main container runs a web server (NGINX). The mount path for the shared volume in the web server container
is `usrsharenginxhtml`. The second container is a Sidecar Container based on Alpine Linux which acts as
a helper container. For this container the `emptyDir` volume is mounted at `pod-data`. The Sidecar Container
writes a file in HTML that has its content based on a ConfigMap. The web server container serves the HTML via HTTP.

 code_sample filedeploymentsdeployment-with-configmap-and-sidecar-container.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesdeploymentsdeployment-with-configmap-and-sidecar-container.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell
kubectl get pods --selectorapp.kubernetes.ionameconfigmap-sidecar-container
```

You should see an output similar to

```
NAME                                           READY   STATUS    RESTARTS   AGE
configmap-sidecar-container-5fb59f558b-87rp7   22     Running   0          94s
configmap-sidecar-container-5fb59f558b-ccs7s   22     Running   0          94s
configmap-sidecar-container-5fb59f558b-wnmgk   22     Running   0          94s
```

Expose the Deployment (the `kubectl` tool creates a
 for you)

```shell
kubectl expose deployment configmap-sidecar-container --nameconfigmap-sidecar-service --port8081 --target-port80
```

Use `kubectl` to forward the port

```shell
# this stays running in the background
kubectl port-forward serviceconfigmap-sidecar-service 80818081
```

Access the service.

```shell
curl httplocalhost8081
```

You should see an output similar to

```
Sat Feb 17 130905 UTC 2024 My preferred color is blue
```

Edit the ConfigMap

```shell
kubectl edit configmap color
```

In the editor that appears, change the value of key `color` from `blue` to `green`. Save your changes.
The kubectl tool updates the ConfigMap accordingly (if you see an error, try again).

Heres an example of how that manifest could look after you edit it

```yaml
apiVersion v1
data
  color green
kind ConfigMap
# You can leave the existing metadata as they are.
# The values youll see wont exactly match these.
metadata
  creationTimestamp 2024-02-17T122030Z
  name color
  namespace default
  resourceVersion 1054
  uid e40bb34c-58df-4280-8bea-6ed16edccfaa
```

Loop over the service URL for few seconds.

```shell
# Cancel this when youre happy with it (Ctrl-C)
while true do curl --connect-timeout 7.5 httplocalhost8081 sleep 10 done
```

You should see the output change as follows

```
Sat Feb 17 131235 UTC 2024 My preferred color is blue
Sat Feb 17 131245 UTC 2024 My preferred color is blue
Sat Feb 17 131255 UTC 2024 My preferred color is blue
Sat Feb 17 131305 UTC 2024 My preferred color is blue
Sat Feb 17 131315 UTC 2024 My preferred color is green
Sat Feb 17 131325 UTC 2024 My preferred color is green
Sat Feb 17 131335 UTC 2024 My preferred color is green
```

# # Update configuration via an immutable ConfigMap that is mounted as a volume #rollout-configmap-immutable-volume

Immutable ConfigMaps are especially used for configuration that is constant and is **not** expected
to change over time. Marking a ConfigMap as immutable allows a performance improvement where the kubelet does not watch for changes.

If you do need to make a change, you should plan to either

- change the name of the ConfigMap, and switch to running Pods that reference the new name
- replace all the nodes in your cluster that have previously run a Pod that used the old value
- restart the kubelet on any node where the kubelet previously loaded the old ConfigMap

An example manifest for an [Immutable ConfigMap](docsconceptsconfigurationconfigmap#configmap-immutable) is shown below.
 code_sample fileconfigmapimmutable-configmap.yaml

Create the Immutable ConfigMap

```shell
kubectl apply -f httpsk8s.ioexamplesconfigmapimmutable-configmap.yaml
```

Below is an example of a Deployment manifest with the Immutable ConfigMap `company-name-20150801` mounted as a
 into the Pods only container.

 code_sample filedeploymentsdeployment-with-immutable-configmap-as-volume.yaml

Create the Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesdeploymentsdeployment-with-immutable-configmap-as-volume.yaml
```

Check the pods for this Deployment to ensure they are ready (matching by
)

```shell
kubectl get pods --selectorapp.kubernetes.ionameimmutable-configmap-volume
```

You should see an output similar to

```
NAME                                          READY   STATUS    RESTARTS   AGE
immutable-configmap-volume-78b6fbff95-5gsfh   11     Running   0          62s
immutable-configmap-volume-78b6fbff95-7vcj4   11     Running   0          62s
immutable-configmap-volume-78b6fbff95-vdslm   11     Running   0          62s
```

The Pods container refers to the data defined in the ConfigMap and uses it to print a report to stdout.
You can check this report by viewing the logs for one of the Pods in that Deployment

```shell
# Pick one Pod that belongs to the Deployment, and view its logs
kubectl logs deploymentsimmutable-configmap-volume
```

You should see an output similar to

```
Found 3 pods, using podimmutable-configmap-volume-78b6fbff95-5gsfh
Wed Mar 20 035234 UTC 2024 The name of the company is ACME, Inc.
Wed Mar 20 035244 UTC 2024 The name of the company is ACME, Inc.
Wed Mar 20 035254 UTC 2024 The name of the company is ACME, Inc.
```

Once a ConfigMap is marked as immutable, it is not possible to revert this change
nor to mutate the contents of the data or the binaryData field.
In order to modify the behavior of the Pods that use this configuration,
you will create a new immutable ConfigMap and edit the Deployment
to define a slightly different pod template, referencing the new ConfigMap.

Create a new immutable ConfigMap by using the manifest shown below

 code_sample fileconfigmapnew-immutable-configmap.yaml

```shell
kubectl apply -f httpsk8s.ioexamplesconfigmapnew-immutable-configmap.yaml
```

You should see an output similar to

```
configmapcompany-name-20240312 created
```

Check the newly created ConfigMap

```shell
kubectl get configmap
```

You should see an output displaying both the old and new ConfigMaps

```
NAME                    DATA   AGE
company-name-20150801   1      22m
company-name-20240312   1      24s
```

Modify the Deployment to reference the new ConfigMap.

Edit the Deployment

```shell
kubectl edit deployment immutable-configmap-volume
```

In the editor that appears, update the existing volume definition to use the new ConfigMap.

```yaml
volumes
- configMap
    defaultMode 420
    name company-name-20240312 # Update this field
  name config-volume
```

You should see the following output

```
deployment.appsimmutable-configmap-volume edited
```

This will trigger a rollout. Wait for all the previous Pods to terminate and the new Pods to be in a ready state.

Monitor the status of the Pods

```shell
kubectl get pods --selectorapp.kubernetes.ionameimmutable-configmap-volume
```

```
NAME                                          READY   STATUS        RESTARTS   AGE
immutable-configmap-volume-5fdb88fcc8-29v8n   11     Running       0          13s
immutable-configmap-volume-5fdb88fcc8-52ddd   11     Running       0          14s
immutable-configmap-volume-5fdb88fcc8-n5jx4   11     Running       0          15s
immutable-configmap-volume-78b6fbff95-5gsfh   11     Terminating   0          32m
immutable-configmap-volume-78b6fbff95-7vcj4   11     Terminating   0          32m
immutable-configmap-volume-78b6fbff95-vdslm   11     Terminating   0          32m
```

You should eventually see an output similar to

```
NAME                                          READY   STATUS    RESTARTS   AGE
immutable-configmap-volume-5fdb88fcc8-29v8n   11     Running   0          43s
immutable-configmap-volume-5fdb88fcc8-52ddd   11     Running   0          44s
immutable-configmap-volume-5fdb88fcc8-n5jx4   11     Running   0          45s
```

View the logs for a Pod in this Deployment

```shell
# Pick one Pod that belongs to the Deployment, and view its logs
kubectl logs deploymentimmutable-configmap-volume
```

You should see an output similar to the below

```
Found 3 pods, using podimmutable-configmap-volume-5fdb88fcc8-n5jx4
Wed Mar 20 042417 UTC 2024 The name of the company is Fiktivesunternehmen GmbH
Wed Mar 20 042427 UTC 2024 The name of the company is Fiktivesunternehmen GmbH
Wed Mar 20 042437 UTC 2024 The name of the company is Fiktivesunternehmen GmbH
```

Once all the deployments have migrated to use the new immutable ConfigMap, it is advised to delete the old one.

```shell
kubectl delete configmap company-name-20150801
```

# # Summary

Changes to a ConfigMap mounted as a Volume on a Pod are available seamlessly after the subsequent kubelet sync.

Changes to a ConfigMap that configures environment variables for a Pod are available after the subsequent rollout for the Pod.

Once a ConfigMap is marked as immutable, it is not possible to revert this change
(you cannot make an immutable ConfigMap mutable), and you also cannot make any change
to the contents of the `data` or the `binaryData` field. You can delete and recreate
the ConfigMap, or you can make a new different ConfigMap. When you delete a ConfigMap,
running containers and their Pods maintain a mount point to any volume that referenced
that existing ConfigMap.

# #  heading cleanup

Terminate the `kubectl port-forward` commands in case they are running.

Delete the resources created during the tutorial

```shell
kubectl delete deployment configmap-volume configmap-env-var configmap-two-containers configmap-sidecar-container immutable-configmap-volume
kubectl delete service configmap-service configmap-sidecar-service
kubectl delete configmap sport fruits color company-name-20240312

kubectl delete configmap company-name-20150801 # In case it was not handled during the task execution
```
