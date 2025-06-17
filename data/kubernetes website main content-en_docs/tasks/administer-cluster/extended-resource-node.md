---
title Advertise Extended Resources for a Node
content_type task
weight 70
---

This page shows how to specify extended resources for a Node.
Extended resources allow cluster administrators to advertise node-level
resources that would otherwise be unknown to Kubernetes.

# #  heading prerequisites

# # Get the names of your Nodes

```shell
kubectl get nodes
```

Choose one of your Nodes to use for this exercise.

# # Advertise a new extended resource on one of your Nodes

To advertise a new extended resource on a Node, send an HTTP PATCH request to
the Kubernetes API server. For example, suppose one of your Nodes has four dongles
attached. Heres an example of a PATCH request that advertises four dongle resources
for your Node.

```
PATCH apiv1nodesstatus HTTP1.1
Accept applicationjson
Content-Type applicationjson-patchjson
Host k8s-master8080

[

    op add,
    path statuscapacityexample.com1dongle,
    value 4

]
```

Note that Kubernetes does not need to know what a dongle is or what a dongle is for.
The preceding PATCH request tells Kubernetes that your Node has four things that
you call dongles.

Start a proxy, so that you can easily send requests to the Kubernetes API server

```shell
kubectl proxy
```

In another command window, send the HTTP PATCH request.
Replace `` with the name of your Node

```shell
curl --header Content-Type applicationjson-patchjson
  --request PATCH
  --data [op add, path statuscapacityexample.com1dongle, value 4]
  httplocalhost8001apiv1nodesstatus
```

In the preceding request, `1` is the encoding for the character  in
the patch path. The operation path value in JSON-Patch is interpreted as a
JSON-Pointer. For more details, see
[IETF RFC 6901](httpstools.ietf.orghtmlrfc6901), section 3.

The output shows that the Node has a capacity of 4 dongles

```
capacity
  cpu 2,
  memory 2049008Ki,
  example.comdongle 4,
```

Describe your Node

```
kubectl describe node
```

Once again, the output shows the dongle resource

```yaml
Capacity
  cpu 2
  memory 2049008Ki
  example.comdongle 4
```

Now, application developers can create Pods that request a certain
number of dongles. See
[Assign Extended Resources to a Container](docstasksconfigure-pod-containerextended-resource).

# # Discussion

Extended resources are similar to memory and CPU resources. For example,
just as a Node has a certain amount of memory and CPU to be shared by all components
running on the Node, it can have a certain number of dongles to be shared
by all components running on the Node. And just as application developers
can create Pods that request a certain amount of memory and CPU, they can
create Pods that request a certain number of dongles.

Extended resources are opaque to Kubernetes Kubernetes does not
know anything about what they are. Kubernetes knows only that a Node
has a certain number of them. Extended resources must be advertised in integer
amounts. For example, a Node can advertise four dongles, but not 4.5 dongles.

# # # Storage example

Suppose a Node has 800 GiB of a special kind of disk storage. You could
create a name for the special storage, say example.comspecial-storage.
Then you could advertise it in chunks of a certain size, say 100 GiB. In that case,
your Node would advertise that it has eight resources of type
example.comspecial-storage.

```yaml
Capacity
 ...
 example.comspecial-storage 8
```

If you want to allow arbitrary requests for special storage, you
could advertise special storage in chunks of size 1 byte. In that case, you would advertise
800Gi resources of type example.comspecial-storage.

```yaml
Capacity
 ...
 example.comspecial-storage  800Gi
```

Then a Container could request any number of bytes of special storage, up to 800Gi.

# # Clean up

Here is a PATCH request that removes the dongle advertisement from a Node.

```
PATCH apiv1nodesstatus HTTP1.1
Accept applicationjson
Content-Type applicationjson-patchjson
Host k8s-master8080

[

    op remove,
    path statuscapacityexample.com1dongle,

]
```

Start a proxy, so that you can easily send requests to the Kubernetes API server

```shell
kubectl proxy
```

In another command window, send the HTTP PATCH request.
Replace `` with the name of your Node

```shell
curl --header Content-Type applicationjson-patchjson
  --request PATCH
  --data [op remove, path statuscapacityexample.com1dongle]
  httplocalhost8001apiv1nodesstatus
```

Verify that the dongle advertisement has been removed

```
kubectl describe node   grep dongle
```

(you should not see any output)

# #  heading whatsnext

# # # For application developers

- [Assign Extended Resources to a Container](docstasksconfigure-pod-containerextended-resource)

# # # For cluster administrators

- [Configure Minimum and Maximum Memory Constraints for a Namespace](docstasksadminister-clustermanage-resourcesmemory-constraint-namespace)
- [Configure Minimum and Maximum CPU Constraints for a Namespace](docstasksadminister-clustermanage-resourcescpu-constraint-namespace)
