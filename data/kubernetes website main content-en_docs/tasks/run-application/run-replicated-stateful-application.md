---
reviewers
- enisoc
- erictune
- foxish
- janetkuo
- kow3ns
- smarterclayton
title Run a Replicated Stateful Application
content_type tutorial
weight 30
---

This page shows how to run a replicated stateful application using a
.
This application is a replicated MySQL database. The example topology has a
single primary server and multiple replicas, using asynchronous row-based
replication.

**This is not a production configuration**. MySQL settings remain on insecure defaults to keep the focus
on general patterns for running stateful applications in Kubernetes.

# #  heading prerequisites

-
-
- This tutorial assumes you are familiar with
  [PersistentVolumes](docsconceptsstoragepersistent-volumes)
  and [StatefulSets](docsconceptsworkloadscontrollersstatefulset),
  as well as other core concepts like [Pods](docsconceptsworkloadspods),
  [Services](docsconceptsservices-networkingservice), and
  [ConfigMaps](docstasksconfigure-pod-containerconfigure-pod-configmap).
- Some familiarity with MySQL helps, but this tutorial aims to present
  general patterns that should be useful for other systems.
- You are using the default namespace or another namespace that does not contain any conflicting objects.
- You need to have a AMD64-compatible CPU.

# #  heading objectives

- Deploy a replicated MySQL topology with a StatefulSet.
- Send MySQL client traffic.
- Observe resistance to downtime.
- Scale the StatefulSet up and down.

# # Deploy MySQL

The example MySQL deployment consists of a ConfigMap, two Services,
and a StatefulSet.

# # # Create a ConfigMap #configmap

Create the ConfigMap from the following YAML configuration file

 code_sample fileapplicationmysqlmysql-configmap.yaml

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationmysqlmysql-configmap.yaml
```

This ConfigMap provides `my.cnf` overrides that let you independently control
configuration on the primary MySQL server and its replicas.
In this case, you want the primary server to be able to serve replication logs to replicas
and you want replicas to reject any writes that dont come via replication.

Theres nothing special about the ConfigMap itself that causes different
portions to apply to different Pods.
Each Pod decides which portion to look at as its initializing,
based on information provided by the StatefulSet controller.

# # # Create Services #services

Create the Services from the following YAML configuration file

 code_sample fileapplicationmysqlmysql-services.yaml

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationmysqlmysql-services.yaml
```

The headless Service provides a home for the DNS entries that the StatefulSet
 creates for each
Pod thats part of the set.
Because the headless Service is named `mysql`, the Pods are accessible by
resolving `.mysql` from within any other Pod in the same Kubernetes
cluster and namespace.

The client Service, called `mysql-read`, is a normal Service with its own
cluster IP that distributes connections across all MySQL Pods that report
being Ready. The set of potential endpoints includes the primary MySQL server and all
replicas.

Note that only read queries can use the load-balanced client Service.
Because there is only one primary MySQL server, clients should connect directly to the
primary MySQL Pod (through its DNS entry within the headless Service) to execute
writes.

# # # Create the StatefulSet #statefulset

Finally, create the StatefulSet from the following YAML configuration file

 code_sample fileapplicationmysqlmysql-statefulset.yaml

```shell
kubectl apply -f httpsk8s.ioexamplesapplicationmysqlmysql-statefulset.yaml
```

You can watch the startup progress by running

```shell
kubectl get pods -l appmysql --watch
```

After a while, you should see all 3 Pods become `Running`

```
NAME      READY     STATUS    RESTARTS   AGE
mysql-0   22       Running   0          2m
mysql-1   22       Running   0          1m
mysql-2   22       Running   0          1m
```

Press **CtrlC** to cancel the watch.

If you dont see any progress, make sure you have a dynamic PersistentVolume
provisioner enabled, as mentioned in the [prerequisites](#before-you-begin).

This manifest uses a variety of techniques for managing stateful Pods as part of
a StatefulSet. The next section highlights some of these techniques to explain
what happens as the StatefulSet creates Pods.

# # Understanding stateful Pod initialization

The StatefulSet controller starts Pods one at a time, in order by their
ordinal index.
It waits until each Pod reports being Ready before starting the next one.

In addition, the controller assigns each Pod a unique, stable name of the form
`-`, which results in Pods named `mysql-0`,
`mysql-1`, and `mysql-2`.

The Pod template in the above StatefulSet manifest takes advantage of these
properties to perform orderly startup of MySQL replication.

# # # Generating configuration

Before starting any of the containers in the Pod spec, the Pod first runs any
[init containers](docsconceptsworkloadspodsinit-containers)
in the order defined.

The first init container, named `init-mysql`, generates special MySQL config
files based on the ordinal index.

The script determines its own ordinal index by extracting it from the end of
the Pod name, which is returned by the `hostname` command.
Then it saves the ordinal (with a numeric offset to avoid reserved values)
into a file called `server-id.cnf` in the MySQL `conf.d` directory.
This translates the unique, stable identity provided by the StatefulSet
into the domain of MySQL server IDs, which require the same properties.

The script in the `init-mysql` container also applies either `primary.cnf` or
`replica.cnf` from the ConfigMap by copying the contents into `conf.d`.
Because the example topology consists of a single primary MySQL server and any number of
replicas, the script assigns ordinal `0` to be the primary server, and everyone
else to be replicas.
Combined with the StatefulSet controllers
[deployment order guarantee](docsconceptsworkloadscontrollersstatefulset#deployment-and-scaling-guarantees),
this ensures the primary MySQL server is Ready before creating replicas, so they can begin
replicating.

# # # Cloning existing data

In general, when a new Pod joins the set as a replica, it must assume the primary MySQL
server might already have data on it. It also must assume that the replication
logs might not go all the way back to the beginning of time.
These conservative assumptions are the key to allow a running StatefulSet
to scale up and down over time, rather than being fixed at its initial size.

The second init container, named `clone-mysql`, performs a clone operation on
a replica Pod the first time it starts up on an empty PersistentVolume.
That means it copies all existing data from another running Pod,
so its local state is consistent enough to begin replicating from the primary server.

MySQL itself does not provide a mechanism to do this, so the example uses a
popular open-source tool called Percona XtraBackup.
During the clone, the source MySQL server might suffer reduced performance.
To minimize impact on the primary MySQL server, the script instructs each Pod to clone
from the Pod whose ordinal index is one lower.
This works because the StatefulSet controller always ensures Pod `N` is
Ready before starting Pod `N1`.

# # # Starting replication

After the init containers complete successfully, the regular containers run.
The MySQL Pods consist of a `mysql` container that runs the actual `mysqld`
server, and an `xtrabackup` container that acts as a
[sidecar](blog201506the-distributed-system-toolkit-patterns).

The `xtrabackup` sidecar looks at the cloned data files and determines if
its necessary to initialize MySQL replication on the replica.
If so, it waits for `mysqld` to be ready and then executes the
`CHANGE MASTER TO` and `START SLAVE` commands with replication parameters
extracted from the XtraBackup clone files.

Once a replica begins replication, it remembers its primary MySQL server and
reconnects automatically if the server restarts or the connection dies.
Also, because replicas look for the primary server at its stable DNS name
(`mysql-0.mysql`), they automatically find the primary server even if it gets a new
Pod IP due to being rescheduled.

Lastly, after starting replication, the `xtrabackup` container listens for
connections from other Pods requesting a data clone.
This server remains up indefinitely in case the StatefulSet scales up, or in
case the next Pod loses its PersistentVolumeClaim and needs to redo the clone.

# # Sending client traffic

You can send test queries to the primary MySQL server (hostname `mysql-0.mysql`)
by running a temporary container with the `mysql5.7` image and running the
`mysql` client binary.

```shell
kubectl run mysql-client --imagemysql5.7 -i --rm --restartNever --
  mysql -h mysql-0.mysql ` with the name of the Node you found in the last step.

Draining a Node can impact other workloads and applications
running on the same node. Only perform the following step in a test
cluster.

```shell
# See above advice about impact on other workloads
kubectl drain  --force --delete-emptydir-data --ignore-daemonsets
```

Now you can watch as the Pod reschedules on a different Node

```shell
kubectl get pod mysql-2 -o wide --watch
```

It should look something like this

```
NAME      READY   STATUS          RESTARTS   AGE       IP            NODE
mysql-2   22     Terminating     0          15m       10.244.1.56   kubernetes-node-9l2t
[...]
mysql-2   02     Pending         0          0s                kubernetes-node-fjlm
mysql-2   02     Init02        0          0s                kubernetes-node-fjlm
mysql-2   02     Init12        0          20s       10.244.5.32   kubernetes-node-fjlm
mysql-2   02     PodInitializing 0          21s       10.244.5.32   kubernetes-node-fjlm
mysql-2   12     Running         0          22s       10.244.5.32   kubernetes-node-fjlm
mysql-2   22     Running         0          30s       10.244.5.32   kubernetes-node-fjlm
```

And again, you should see server ID `102` disappear from the
`SELECT server_id` loop output for a while and then return.

Now uncordon the Node to return it to a normal state

```shell
kubectl uncordon
```

# # Scaling the number of replicas

When you use MySQL replication, you can scale your read query capacity by
adding replicas.
For a StatefulSet, you can achieve this with a single command

```shell
kubectl scale statefulset mysql  --replicas5
```

Watch the new Pods come up by running

```shell
kubectl get pods -l appmysql --watch
```

Once theyre up, you should see server IDs `103` and `104` start appearing in
the `SELECT server_id` loop output.

You can also verify that these new servers have the data you added before they
existed

```shell
kubectl run mysql-client --imagemysql5.7 -i -t --rm --restartNever --
  mysql -h mysql-3.mysql -e SELECT * FROM test.messages
```

```
Waiting for pod defaultmysql-client to be running, status is Pending, pod ready false
---------
 message
---------
 hello
---------
pod mysql-client deleted
```

Scaling back down is also seamless

```shell
kubectl scale statefulset mysql --replicas3
```

Although scaling up creates new PersistentVolumeClaims
automatically, scaling down does not automatically delete these PVCs.

This gives you the choice to keep those initialized PVCs around to make
scaling back up quicker, or to extract data before deleting them.

You can see this by running

```shell
kubectl get pvc -l appmysql
```

Which shows that all 5 PVCs still exist, despite having scaled the
StatefulSet down to 3

```
NAME           STATUS    VOLUME                                     CAPACITY   ACCESSMODES   AGE
data-mysql-0   Bound     pvc-8acbf5dc-b103-11e6-93fa-42010a800002   10Gi       RWO           20m
data-mysql-1   Bound     pvc-8ad39820-b103-11e6-93fa-42010a800002   10Gi       RWO           20m
data-mysql-2   Bound     pvc-8ad69a6d-b103-11e6-93fa-42010a800002   10Gi       RWO           20m
data-mysql-3   Bound     pvc-50043c45-b1c5-11e6-93fa-42010a800002   10Gi       RWO           2m
data-mysql-4   Bound     pvc-500a9957-b1c5-11e6-93fa-42010a800002   10Gi       RWO           2m
```

If you dont intend to reuse the extra PVCs, you can delete them

```shell
kubectl delete pvc data-mysql-3
kubectl delete pvc data-mysql-4
```

# #  heading cleanup

1. Cancel the `SELECT server_id` loop by pressing **CtrlC** in its terminal,
   or running the following from another terminal

   ```shell
   kubectl delete pod mysql-client-loop --now
   ```

1. Delete the StatefulSet. This also begins terminating the Pods.

   ```shell
   kubectl delete statefulset mysql
   ```

1. Verify that the Pods disappear.
   They might take some time to finish terminating.

   ```shell
   kubectl get pods -l appmysql
   ```

   Youll know the Pods have terminated when the above returns

   ```
   No resources found.
   ```

1. Delete the ConfigMap, Services, and PersistentVolumeClaims.

   ```shell
   kubectl delete configmap,service,pvc -l appmysql
   ```

1. If you manually provisioned PersistentVolumes, you also need to manually
   delete them, as well as release the underlying resources.
   If you used a dynamic provisioner, it automatically deletes the
   PersistentVolumes when it sees that you deleted the PersistentVolumeClaims.
   Some dynamic provisioners (such as those for EBS and PD) also release the
   underlying resources upon deleting the PersistentVolumes.

# #  heading whatsnext

- Learn more about [scaling a StatefulSet](docstasksrun-applicationscale-stateful-set).
- Learn more about [debugging a StatefulSet](docstasksdebugdebug-applicationdebug-statefulset).
- Learn more about [deleting a StatefulSet](docstasksrun-applicationdelete-stateful-set).
- Learn more about [force deleting StatefulSet Pods](docstasksrun-applicationforce-delete-stateful-set-pod).
- Look in the [Helm Charts repository](httpsartifacthub.io)
  for other stateful application examples.
