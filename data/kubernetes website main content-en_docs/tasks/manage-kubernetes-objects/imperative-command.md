---
title Managing Kubernetes Objects Using Imperative Commands
content_type task
weight 30
---

Kubernetes objects can quickly be created, updated, and deleted directly using
imperative commands built into the `kubectl` command-line tool. This document
explains how those commands are organized and how to use them to manage live objects.

# #  heading prerequisites

Install [`kubectl`](docstaskstools).

# # Trade-offs

The `kubectl` tool supports three kinds of object management

* Imperative commands
* Imperative object configuration
* Declarative object configuration

See [Kubernetes Object Management](docsconceptsoverviewworking-with-objectsobject-management)
for a discussion of the advantages and disadvantage of each kind of object management.

# # How to create objects

The `kubectl` tool supports verb-driven commands for creating some of the most common
object types. The commands are named to be recognizable to users unfamiliar with
the Kubernetes object types.

- `run` Create a new Pod to run a Container.
- `expose` Create a new Service object to load balance traffic across Pods.
- `autoscale` Create a new Autoscaler object to automatically horizontally scale a controller, such as a Deployment.

The `kubectl` tool also supports creation commands driven by object type.
These commands support more object types and are more explicit about
their intent, but require users to know the type of objects they intend
to create.

- `create  [] `

Some objects types have subtypes that you can specify in the `create` command.
For example, the Service object has several subtypes including ClusterIP,
LoadBalancer, and NodePort. Heres an example that creates a Service with
subtype NodePort

```shell
kubectl create service nodeport
```

In the preceding example, the `create service nodeport` command is called
a subcommand of the `create service` command.

You can use the `-h` flag to find the arguments and flags supported by
a subcommand

```shell
kubectl create service nodeport -h
```

# # How to update objects

The `kubectl` command supports verb-driven commands for some common update operations.
These commands are named to enable users unfamiliar with Kubernetes
objects to perform updates without knowing the specific fields
that must be set

- `scale` Horizontally scale a controller to add or remove Pods by updating the replica count of the controller.
- `annotate` Add or remove an annotation from an object.
- `label` Add or remove a label from an object.

The `kubectl` command also supports update commands driven by an aspect of the object.
Setting this aspect may set different fields for different object types

- `set` `` Set an aspect of an object.

In Kubernetes version 1.5, not every verb-driven command has an associated aspect-driven command.

The `kubectl` tool supports these additional ways to update a live object directly,
however they require a better understanding of the Kubernetes object schema.

- `edit` Directly edit the raw configuration of a live object by opening its configuration in an editor.
- `patch` Directly modify specific fields of a live object by using a patch string.
For more details on patch strings, see the patch section in
[API Conventions](httpsgit.k8s.iocommunitycontributorsdevelsig-architectureapi-conventions.md#patch-operations).

# # How to delete objects

You can use the `delete` command to delete an object from a cluster

- `delete `

You can use `kubectl delete` for both imperative commands and imperative object
configuration. The difference is in the arguments passed to the command. To use
`kubectl delete` as an imperative command, pass the object to be deleted as
an argument. Heres an example that passes a Deployment object named nginx

```shell
kubectl delete deploymentnginx
```

# # How to view an object

TODO(pwittrock) Uncomment this when implemented.

You can use `kubectl view` to print specific fields of an object.

- `view` Prints the value of a specific field of an object.

There are several commands for printing information about an object

- `get` Prints basic information about matching objects.  Use `get -h` to see a list of options.
- `describe` Prints aggregated detailed information about matching objects.
- `logs` Prints the stdout and stderr for a container running in a Pod.

# # Using `set` commands to modify objects before creation

There are some object fields that dont have a flag you can use
in a `create` command. In some of those cases, you can use a combination of
`set` and `create` to specify a value for the field before object
creation. This is done by piping the output of the `create` command to the
`set` command, and then back to the `create` command. Heres an example

```sh
kubectl create service clusterip my-svc --clusteripNone -o yaml --dry-runclient  kubectl set selector --local -f - environmentqa -o yaml  kubectl create -f -
```

1. The `kubectl create service -o yaml --dry-runclient` command creates the configuration for the Service, but prints it to stdout as YAML instead of sending it to the Kubernetes API server.
1. The `kubectl set selector --local -f - -o yaml` command reads the configuration from stdin, and writes the updated configuration to stdout as YAML.
1. The `kubectl create -f -` command creates the object using the configuration provided via stdin.

# # Using `--edit` to modify objects before creation

You can use `kubectl create --edit` to make arbitrary changes to an object
before it is created. Heres an example

```sh
kubectl create service clusterip my-svc --clusteripNone -o yaml --dry-runclient  tmpsrv.yaml
kubectl create --edit -f tmpsrv.yaml
```

1. The `kubectl create service` command creates the configuration for the Service and saves it to `tmpsrv.yaml`.
1. The `kubectl create --edit` command opens the configuration file for editing before it creates the object.

# #  heading whatsnext

* [Imperative Management of Kubernetes Objects Using Configuration Files](docstasksmanage-kubernetes-objectsimperative-config)
* [Declarative Management of Kubernetes Objects Using Configuration Files](docstasksmanage-kubernetes-objectsdeclarative-config)
* [Kubectl Command Reference](docsreferencegeneratedkubectlkubectl-commands)
* [Kubernetes API Reference](docsreferencegeneratedkubernetes-api)
