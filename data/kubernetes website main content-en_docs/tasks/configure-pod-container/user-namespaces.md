---
title Use a User Namespace With a Pod
reviewers
content_type task
weight 210
min-kubernetes-server-version v1.25
---

This page shows how to configure a user namespace for pods. This allows you to
isolate the user running inside the container from the one in the host.

A process running as root in a container can run as a different (non-root) user
in the host in other words, the process has full privileges for operations
inside the user namespace, but is unprivileged for operations outside the
namespace.

You can use this feature to reduce the damage a compromised container can do to
the host or other pods in the same node. There are [several security
vulnerabilities][KEP-vulns] rated either **HIGH** or **CRITICAL** that were not
exploitable when user namespaces is active. It is expected user namespace will
mitigate some future vulnerabilities too.

Without using a user namespace a container running as root, in the case of a
container breakout, has root privileges on the node. And if some capability were
granted to the container, the capabilities are valid on the host too. None of
this is true when user namespaces are used.

[KEP-vulns] httpsgithub.comkubernetesenhancementstree217d790720c5aef09b8bd4d6ca96284a0affe6c2kepssig-node127-user-namespaces#motivation

# #  heading prerequisites

 thirdparty-content singletrue

* The node OS needs to be Linux
* You need to exec commands in the host
* You need to be able to exec into pods
* You need to enable the `UserNamespacesSupport`
  [feature gate](docsreferencecommand-line-tools-referencefeature-gates)

The feature gate to enable user namespaces was previously named
`UserNamespacesStatelessPodsSupport`, when only stateless pods were supported.
Only Kubernetes v1.25 through to v1.27 recognise `UserNamespacesStatelessPodsSupport`.

The cluster that youre using **must** include at least one node that meets the
[requirements](docsconceptsworkloadspodsuser-namespaces#before-you-begin)
for using user namespaces with Pods.

If you have a mixture of nodes and only some of the nodes provide user namespace support for
Pods, you also need to ensure that the user namespace Pods are
[scheduled](docsconceptsscheduling-evictionassign-pod-node) to suitable nodes.

# # Run a Pod that uses a user namespace #create-pod

A user namespace for a pod is enabled setting the `hostUsers` field of `.spec`
to `false`. For example

 code_sample filepodsuser-namespaces-stateless.yaml

1. Create the pod on your cluster

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodsuser-namespaces-stateless.yaml
   ```

1. Add a debugging container and attach to it and run `readlink procselfnsuser`

   ```shell
   kubectl debug userns -it --imagebusybox
   ```

Run this command

```shell
readlink procselfnsuser
```

The output is similar to

```shell
user[4026531837]
```

Also run

```shell
cat procselfuid_map
```

The output is similar to
```shell
0  833617920      65536
```

Then, open a shell in the host and run the same commands.

The `readlink` command shows the user namespace the process is running in. It
should be different when it is run on the host and inside the container.

The last number of the `uid_map` file inside the container must be 65536, on the
host it must be a bigger number.

If you are running the kubelet inside a user namespace, you need to compare the
output from running the command in the pod to the output of running in the host

```shell
readlink procpidnsuser
```

replacing `pid` with the kubelet PID.
