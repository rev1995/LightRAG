---
title Operator pattern
content_type concept
weight 30
---

Operators are software extensions to Kubernetes that make use of
[custom resources](docsconceptsextend-kubernetesapi-extensioncustom-resources)
to manage applications and their components. Operators follow
Kubernetes principles, notably the [control loop](docsconceptsarchitecturecontroller).

# # Motivation

The _operator pattern_ aims to capture the key aim of a human operator who
is managing a service or set of services. Human operators who look after
specific applications and services have deep knowledge of how the system
ought to behave, how to deploy it, and how to react if there are problems.

People who run workloads on Kubernetes often like to use automation to take
care of repeatable tasks. The operator pattern captures how you can write
code to automate a task beyond what Kubernetes itself provides.

# # Operators in Kubernetes

Kubernetes is designed for automation. Out of the box, you get lots of
built-in automation from the core of Kubernetes. You can use Kubernetes
to automate deploying and running workloads, *and* you can automate how
Kubernetes does that.

Kubernetes
concept lets you extend the clusters behaviour without modifying the code of Kubernetes
itself by linking  to
one or more custom resources. Operators are clients of the Kubernetes API that act as
controllers for a [Custom Resource](docsconceptsextend-kubernetesapi-extensioncustom-resources).

# # An example operator #example

Some of the things that you can use an operator to automate include

* deploying an application on demand
* taking and restoring backups of that applications state
* handling upgrades of the application code alongside related changes such
  as database schemas or extra configuration settings
* publishing a Service to applications that dont support Kubernetes APIs to
  discover them
* simulating failure in all or part of your cluster to test its resilience
* choosing a leader for a distributed application without an internal
  member election process

What might an operator look like in more detail Heres an example

1. A custom resource named SampleDB, that you can configure into the cluster.
2. A Deployment that makes sure a Pod is running that contains the
   controller part of the operator.
3. A container image of the operator code.
4. Controller code that queries the control plane to find out what SampleDB
   resources are configured.
5. The core of the operator is code to tell the API server how to make
   reality match the configured resources.
   * If you add a new SampleDB, the operator sets up PersistentVolumeClaims
     to provide durable database storage, a StatefulSet to run SampleDB and
     a Job to handle initial configuration.
   * If you delete it, the operator takes a snapshot, then makes sure that
     the StatefulSet and Volumes are also removed.
6. The operator also manages regular database backups. For each SampleDB
   resource, the operator determines when to create a Pod that can connect
   to the database and take backups. These Pods would rely on a ConfigMap
   and  or a Secret that has database connection details and credentials.
7. Because the operator aims to provide robust automation for the resource
   it manages, there would be additional supporting code. For this example,
   code checks to see if the database is running an old version and, if so,
   creates Job objects that upgrade it for you.

# # Deploying operators

The most common way to deploy an operator is to add the
Custom Resource Definition and its associated Controller to your cluster.
The Controller will normally run outside of the
,
much as you would run any containerized application.
For example, you can run the controller in your cluster as a Deployment.

# # Using an operator #using-operators

Once you have an operator deployed, youd use it by adding, modifying or
deleting the kind of resource that the operator uses. Following the above
example, you would set up a Deployment for the operator itself, and then

```shell
kubectl get SampleDB                   # find configured databases

kubectl edit SampleDBexample-database # manually change some settings
```

hellipand thats it! The operator will take care of applying the changes
as well as keeping the existing service in good shape.

# # Writing your own operator #writing-operator

If there isnt an operator in the ecosystem that implements the behavior you
want, you can code your own.

You also implement an operator (that is, a Controller) using any language  runtime
that can act as a [client for the Kubernetes API](docsreferenceusing-apiclient-libraries).

Following are a few libraries and tools you can use to write your own cloud native
operator.

 thirdparty-content

* [Charmed Operator Framework](httpsjuju.is)
* [Java Operator SDK](httpsgithub.comoperator-frameworkjava-operator-sdk)
* [Kopf](httpsgithub.comnolarkopf) (Kubernetes Operator Pythonic Framework)
* [kube-rs](httpskube.rs) (Rust)
* [kubebuilder](httpsbook.kubebuilder.io)
* [KubeOps](httpsbuehler.github.iodotnet-operator-sdk) (.NET operator SDK)
* [Mast](httpsdocs.ansi.servicesmastuser_guideoperator)
* [Metacontroller](httpsmetacontroller.github.iometacontrollerintro.html) along with WebHooks that
  you implement yourself
* [Operator Framework](httpsoperatorframework.io)
* [shell-operator](httpsgithub.comflantshell-operator)

# #  heading whatsnext

* Read the
  [Operator White Paper](httpsgithub.comcncftag-app-deliveryblob163962c4b1cd70d085107fc579e3e04c2e14d59coperator-wgwhitepaperOperator-WhitePaper_v1-0.md).
* Learn more about [Custom Resources](docsconceptsextend-kubernetesapi-extensioncustom-resources)
* Find ready-made operators on [OperatorHub.io](httpsoperatorhub.io) to suit your use case
* [Publish](httpsoperatorhub.io) your operator for other people to use
* Read [CoreOS original article](httpsweb.archive.orgweb20170129131616httpscoreos.comblogintroducing-operators.html)
  that introduced the operator pattern (this is an archived version of the original article).
* Read an [article](httpscloud.google.comblogproductscontainers-kubernetesbest-practices-for-building-kubernetes-operators-and-stateful-apps)
  from Google Cloud about best practices for building operators
