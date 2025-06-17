---
reviewers
- rickypai
- thockin
title Adding entries to Pod etchosts with HostAliases
content_type task
weight 60
min-kubernetes-server-version 1.7
---

Adding entries to a Pods `etchosts` file provides Pod-level override of hostname resolution when DNS and other options are not applicable. You can add these custom entries with the HostAliases field in PodSpec.

The Kubernetes project recommends modifying DNS configuration using the `hostAliases` field
(part of the `.spec` for a Pod), and not by using an init container or other means to edit `etchosts`
directly.
Change made in other ways may be overwritten by the kubelet during Pod creation or restart.

# # Default hosts file content

Start an Nginx Pod which is assigned a Pod IP

```shell
kubectl run nginx --image nginx
```

```
podnginx created
```

Examine a Pod IP

```shell
kubectl get pods --outputwide
```

```
NAME     READY     STATUS    RESTARTS   AGE    IP           NODE
nginx    11       Running   0          13s    10.200.0.4   worker0
```

The hosts file content would look like this

```shell
kubectl exec nginx -- cat etchosts
```

```
# Kubernetes-managed hosts file.
127.0.0.1	localhost
1	localhost ip6-localhost ip6-loopback
fe000	ip6-localnet
fe000	ip6-mcastprefix
fe001	ip6-allnodes
fe002	ip6-allrouters
10.200.0.4	nginx
```

By default, the `hosts` file only includes IPv4 and IPv6 boilerplates like
`localhost` and its own hostname.

# # Adding additional entries with hostAliases

In addition to the default boilerplate, you can add additional entries to the
`hosts` file.
For example to resolve `foo.local`, `bar.local` to `127.0.0.1` and `foo.remote`,
`bar.remote` to `10.1.2.3`, you can configure HostAliases for a Pod under
`.spec.hostAliases`

 code_sample fileservicenetworkinghostaliases-pod.yaml

You can start a Pod with that configuration by running

```shell
kubectl apply -f httpsk8s.ioexamplesservicenetworkinghostaliases-pod.yaml
```

```
podhostaliases-pod created
```

Examine a Pods details to see its IPv4 address and its status

```shell
kubectl get pod --outputwide
```

```
NAME                           READY     STATUS      RESTARTS   AGE       IP              NODE
hostaliases-pod                01       Completed   0          6s        10.200.0.5      worker0
```

The `hosts` file content looks like this

```shell
kubectl logs hostaliases-pod
```

```
# Kubernetes-managed hosts file.
127.0.0.1	localhost
1	localhost ip6-localhost ip6-loopback
fe000	ip6-localnet
fe000	ip6-mcastprefix
fe001	ip6-allnodes
fe002	ip6-allrouters
10.200.0.5	hostaliases-pod

# Entries added by HostAliases.
127.0.0.1	foo.local	bar.local
10.1.2.3	foo.remote	bar.remote
```

with the additional entries specified at the bottom.

# # Why does the kubelet manage the hosts file #why-does-kubelet-manage-the-hosts-file

The kubelet manages the
`hosts` file for each container of the Pod to prevent the container runtime from
modifying the file after the containers have already been started.
Historically, Kubernetes always used Docker Engine as its container runtime, and Docker Engine would
then modify the `etchosts` file after each container had started.

Current Kubernetes can use a variety of container runtimes even so, the kubelet manages the
hosts file within each container so that the outcome is as intended regardless of which
container runtime you use.

Avoid making manual changes to the hosts file inside a container.

If you make manual changes to the hosts file,
those changes are lost when the container exits.
