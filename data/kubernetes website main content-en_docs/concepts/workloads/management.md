---
title Managing Workloads
content_type concept
reviewers
- janetkuo
weight 40
---

Youve deployed your application and exposed it via a Service. Now what Kubernetes provides a
number of tools to help you manage your application deployment, including scaling and updating.

# # Organizing resource configurations

Many applications require multiple resources to be created, such as a Deployment along with a Service.
Management of multiple resources can be simplified by grouping them together in the same file
(separated by `---` in YAML). For example

 code_sample fileapplicationnginx-app.yaml

Multiple resources can be created the same way as a single resource

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationnginx-app.yaml
```

```none
servicemy-nginx-svc created
deployment.appsmy-nginx created
```

The resources will be created in the order they appear in the manifest. Therefore, its best to
specify the Service first, since that will ensure the scheduler can spread the pods associated
with the Service as they are created by the controller(s), such as Deployment.

`kubectl apply` also accepts multiple `-f` arguments

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationnginxnginx-svc.yaml
  -f httpsk8s.ioexamplesapplicationnginxnginx-deployment.yaml
```

It is a recommended practice to put resources related to the same microservice or application tier
into the same file, and to group all of the files associated with your application in the same
directory. If the tiers of your application bind to each other using DNS, you can deploy all of
the components of your stack together.

A URL can also be specified as a configuration source, which is handy for deploying directly from
manifests in your source control system

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationnginxnginx-deployment.yaml
```

```none
deployment.appsmy-nginx created
```

If you need to define more manifests, such as adding a ConfigMap, you can do that too.

# # # External tools

This section lists only the most common tools used for managing workloads on Kubernetes. To see a larger list, view
[Application definition and image build](httpslandscape.cncf.ioguide#app-definition-and-development--application-definition-image-build)
in the  Landscape.

# # # # Helm #external-tool-helm

 thirdparty-content singletrue

[Helm](httpshelm.sh) is a tool for managing packages of pre-configured
Kubernetes resources. These packages are known as _Helm charts_.

# # # # Kustomize #external-tool-kustomize

[Kustomize](httpskustomize.io) traverses a Kubernetes manifest to add, remove or update configuration options.
It is available both as a standalone binary and as a [native feature](docstasksmanage-kubernetes-objectskustomization)
of kubectl.

# # Bulk operations in kubectl

Resource creation isnt the only operation that `kubectl` can perform in bulk. It can also extract
resource names from configuration files in order to perform other operations, in particular to
delete the same resources you created

```shell
kubectl delete -f httpsk8s.ioexamplesapplicationnginx-app.yaml
```

```none
deployment.apps my-nginx deleted
service my-nginx-svc deleted
```

In the case of two resources, you can specify both resources on the command line using the
resourcename syntax

```shell
kubectl delete deploymentsmy-nginx servicesmy-nginx-svc
```

For larger numbers of resources, youll find it easier to specify the selector (label query)
specified using `-l` or `--selector`, to filter resources by their labels

```shell
kubectl delete deployment,services -l appnginx
```

```none
deployment.apps my-nginx deleted
service my-nginx-svc deleted
```

# # # Chaining and filtering

Because `kubectl` outputs resource names in the same syntax it accepts, you can chain operations
using `()` or `xargs`

```shell
kubectl get (kubectl create -f docsconceptscluster-administrationnginx -o name  grep service )
kubectl create -f docsconceptscluster-administrationnginx -o name  grep service  xargs -i kubectl get
```

The output might be similar to

```none
NAME           TYPE           CLUSTER-IP   EXTERNAL-IP   PORT(S)      AGE
my-nginx-svc   LoadBalancer   10.0.0.208        80TCP       0s
```

With the above commands, first you create resources under `examplesapplicationnginx` and print
the resources created with `-o name` output format (print each resource as resourcename).
Then you `grep` only the Service, and then print it with [`kubectl get`](docsreferencekubectlgeneratedkubectl_get).

# # # Recursive operations on local files

If you happen to organize your resources across several subdirectories within a particular
directory, you can recursively perform the operations on the subdirectories also, by specifying
`--recursive` or `-R` alongside the `--filename``-f` argument.

For instance, assume there is a directory `projectk8sdevelopment` that holds all of the
 needed for the development environment,
organized by resource type

```none
projectk8sdevelopment
 configmap
    my-configmap.yaml
 deployment
    my-deployment.yaml
 pvc
     my-pvc.yaml
```

By default, performing a bulk operation on `projectk8sdevelopment` will stop at the first level
of the directory, not processing any subdirectories. If you had tried to create the resources in
this directory using the following command, we would have encountered an error

```shell
kubectl apply -f projectk8sdevelopment
```

```none
error you must provide one or more resources by argument or filename (.json.yaml.ymlstdin)
```

Instead, specify the `--recursive` or `-R` command line argument along with the `--filename``-f` argument

```shell
kubectl apply -f projectk8sdevelopment --recursive
```

```none
configmapmy-config created
deployment.appsmy-deployment created
persistentvolumeclaimmy-pvc created
```

The `--recursive` argument works with any operation that accepts the `--filename``-f` argument such as
`kubectl create`, `kubectl get`, `kubectl delete`, `kubectl describe`, or even `kubectl rollout`.

The `--recursive` argument also works when multiple `-f` arguments are provided

```shell
kubectl apply -f projectk8snamespaces -f projectk8sdevelopment --recursive
```

```none
namespacedevelopment created
namespacestaging created
configmapmy-config created
deployment.appsmy-deployment created
persistentvolumeclaimmy-pvc created
```

If youre interested in learning more about `kubectl`, go ahead and read
[Command line tool (kubectl)](docsreferencekubectl).

# # Updating your application without an outage

At some point, youll eventually need to update your deployed application, typically by specifying
a new image or image tag. `kubectl` supports several update operations, each of which is applicable
to different scenarios.

You can run multiple copies of your app, and use a _rollout_ to gradually shift the traffic to
new healthy Pods. Eventually, all the running Pods would have the new software.

This section of the page guides you through how to create and update applications with Deployments.

Lets say you were running version 1.14.2 of nginx

```shell
kubectl create deployment my-nginx --imagenginx1.14.2
```

```none
deployment.appsmy-nginx created
```

Ensure that there is 1 replica

```shell
kubectl scale --replicas 1 deploymentsmy-nginx --subresourcescale --typemerge -p specreplicas 1
```

```none
deployment.appsmy-nginx scaled
```

and allow Kubernetes to add more temporary replicas during a rollout, by setting a _surge maximum_ of
100

```shell
kubectl patch --typemerge -p specstrategyrollingUpdatemaxSurge 100
```

```none
deployment.appsmy-nginx patched
```

To update to version 1.16.1, change `.spec.template.spec.containers[0].image` from `nginx1.14.2`
to `nginx1.16.1` using `kubectl edit`

```shell
kubectl edit deploymentmy-nginx
# Change the manifest to use the newer container image, then save your changes
```

Thats it! The Deployment will declaratively update the deployed nginx application progressively
behind the scene. It ensures that only a certain number of old replicas may be down while they are
being updated, and only a certain number of new replicas may be created above the desired number
of pods. To learn more details about how this happens,
visit [Deployment](docsconceptsworkloadscontrollersdeployment).

You can use rollouts with DaemonSets, Deployments, or StatefulSets.

# # # Managing rollouts

You can use [`kubectl rollout`](docsreferencekubectlgeneratedkubectl_rollout) to manage a
progressive update of an existing application.

For example

```shell
kubectl apply -f my-deployment.yaml

# wait for rollout to finish
kubectl rollout status deploymentmy-deployment --timeout 10m # 10 minute timeout
```

or

```shell
kubectl apply -f backing-stateful-component.yaml

# dont wait for rollout to finish, just check the status
kubectl rollout status statefulsetsbacking-stateful-component --watchfalse
```

You can also pause, resume or cancel a rollout.
Visit [`kubectl rollout`](docsreferencekubectlgeneratedkubectl_rollout) to learn more.

# # Canary deployments

Another scenario where multiple labels are needed is to distinguish deployments of different
releases or configurations of the same component. It is common practice to deploy a *canary* of a
new application release (specified via image tag in the pod template) side by side with the
previous release so that the new release can receive live production traffic before fully rolling
it out.

For instance, you can use a `track` label to differentiate different releases.

The primary, stable release would have a `track` label with value as `stable`

```none
name frontend
replicas 3
...
labels
   app guestbook
   tier frontend
   track stable
...
image gb-frontendv3
```

and then you can create a new release of the guestbook frontend that carries the `track` label
with different value (i.e. `canary`), so that two sets of pods would not overlap

```none
name frontend-canary
replicas 1
...
labels
   app guestbook
   tier frontend
   track canary
...
image gb-frontendv4
```

The frontend service would span both sets of replicas by selecting the common subset of their
labels (i.e. omitting the `track` label), so that the traffic will be redirected to both
applications

```yaml
selector
   app guestbook
   tier frontend
```

You can tweak the number of replicas of the stable and canary releases to determine the ratio of
each release that will receive live production traffic (in this case, 31).
Once youre confident, you can update the stable track to the new application release and remove
the canary one.

# # Updating annotations

Sometimes you would want to attach annotations to resources. Annotations are arbitrary
non-identifying metadata for retrieval by API clients such as tools or libraries.
This can be done with `kubectl annotate`. For example

```shell
kubectl annotate pods my-nginx-v4-9gw19 descriptionmy frontend running nginx
kubectl get pods my-nginx-v4-9gw19 -o yaml
```

```shell
apiVersion v1
kind pod
metadata
  annotations
    description my frontend running nginx
...
```

For more information, see [annotations](docsconceptsoverviewworking-with-objectsannotations)
and [kubectl annotate](docsreferencekubectlgeneratedkubectl_annotate).

# # Scaling your application

When load on your application grows or shrinks, use `kubectl` to scale your application.
For instance, to decrease the number of nginx replicas from 3 to 1, do

```shell
kubectl scale deploymentmy-nginx --replicas1
```

```none
deployment.appsmy-nginx scaled
```

Now you only have one pod managed by the deployment.

```shell
kubectl get pods -l appnginx
```

```none
NAME                        READY     STATUS    RESTARTS   AGE
my-nginx-2035384211-j5fhi   11       Running   0          30m
```

To have the system automatically choose the number of nginx replicas as needed,
ranging from 1 to 3, do

```shell
# This requires an existing source of container and Pod metrics
kubectl autoscale deploymentmy-nginx --min1 --max3
```

```none
horizontalpodautoscaler.autoscalingmy-nginx autoscaled
```

Now your nginx replicas will be scaled up and down as needed, automatically.

For more information, please see [kubectl scale](docsreferencekubectlgeneratedkubectl_scale),
[kubectl autoscale](docsreferencekubectlgeneratedkubectl_autoscale) and
[horizontal pod autoscaler](docstasksrun-applicationhorizontal-pod-autoscale) document.

# # In-place updates of resources

Sometimes its necessary to make narrow, non-disruptive updates to resources youve created.

# # # kubectl apply

It is suggested to maintain a set of configuration files in source control
(see [configuration as code](httpsmartinfowler.comblikiInfrastructureAsCode.html)),
so that they can be maintained and versioned along with the code for the resources they configure.
Then, you can use [`kubectl apply`](docsreferencekubectlgeneratedkubectl_apply)
to push your configuration changes to the cluster.

This command will compare the version of the configuration that youre pushing with the previous
version and apply the changes youve made, without overwriting any automated changes to properties
you havent specified.

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationnginxnginx-deployment.yaml
```

```none
deployment.appsmy-nginx configured
```

To learn more about the underlying mechanism, read [server-side apply](docsreferenceusing-apiserver-side-apply).

# # # kubectl edit

Alternatively, you may also update resources with [`kubectl edit`](docsreferencekubectlgeneratedkubectl_edit)

```shell
kubectl edit deploymentmy-nginx
```

This is equivalent to first `get` the resource, edit it in text editor, and then `apply` the
resource with the updated version

```shell
kubectl get deployment my-nginx -o yaml  tmpnginx.yaml
vi tmpnginx.yaml
# do some edit, and then save the file

kubectl apply -f tmpnginx.yaml
deployment.appsmy-nginx configured

rm tmpnginx.yaml
```

This allows you to do more significant changes more easily. Note that you can specify the editor
with your `EDITOR` or `KUBE_EDITOR` environment variables.

For more information, please see [kubectl edit](docsreferencekubectlgeneratedkubectl_edit).

# # # kubectl patch

You can use [`kubectl patch`](docsreferencekubectlgeneratedkubectl_patch) to update API objects in place.
This subcommand supports JSON patch,
JSON merge patch, and strategic merge patch.

See
[Update API Objects in Place Using kubectl patch](docstasksmanage-kubernetes-objectsupdate-api-object-kubectl-patch)
for more details.

# # Disruptive updates

In some cases, you may need to update resource fields that cannot be updated once initialized, or
you may want to make a recursive change immediately, such as to fix broken pods created by a
Deployment. To change such fields, use `replace --force`, which deletes and re-creates the
resource. In this case, you can modify your original configuration file

```shell
kubectl replace -f httpsk8s.ioexamplesapplicationnginxnginx-deployment.yaml --force
```

```none
deployment.appsmy-nginx deleted
deployment.appsmy-nginx replaced
```

# #  heading whatsnext

- Learn about [how to use `kubectl` for application introspection and debugging](docstasksdebugdebug-applicationdebug-running-pod).
