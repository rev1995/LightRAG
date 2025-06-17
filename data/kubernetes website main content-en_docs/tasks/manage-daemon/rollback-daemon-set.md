---
reviewers
- janetkuo
title Perform a Rollback on a DaemonSet
content_type task
weight 20
min-kubernetes-server-version 1.7
---

This page shows how to perform a rollback on a .

# #  heading prerequisites

You should already know how to [perform a rolling update on a
 DaemonSet](docstasksmanage-daemonupdate-daemon-set).

# # Performing a rollback on a DaemonSet

# # # Step 1 Find the DaemonSet revision you want to roll back to

You can skip this step if you only want to roll back to the last revision.

List all revisions of a DaemonSet

```shell
kubectl rollout history daemonset
```

This returns a list of DaemonSet revisions

```
daemonsets
REVISION        CHANGE-CAUSE
1               ...
2               ...
...
```

* Change cause is copied from DaemonSet annotation `kubernetes.iochange-cause`
  to its revisions upon creation. You may specify `--recordtrue` in `kubectl`
  to record the command executed in the change cause annotation.

To see the details of a specific revision

```shell
kubectl rollout history daemonset  --revision1
```

This returns the details of that revision

```
daemonsets  with revision #1
Pod Template
Labels       foobar
Containers
app
 Image        ...
 Port         ...
 Environment  ...
 Mounts       ...
Volumes      ...
```

# # # Step 2 Roll back to a specific revision

```shell
# Specify the revision number you get from Step 1 in --to-revision
kubectl rollout undo daemonset  --to-revision
```

If it succeeds, the command returns

```
daemonset  rolled back
```

If `--to-revision` flag is not specified, kubectl picks the most recent revision.

# # # Step 3 Watch the progress of the DaemonSet rollback

`kubectl rollout undo daemonset` tells the server to start rolling back the
DaemonSet. The real rollback is done asynchronously inside the cluster
.

To watch the progress of the rollback

```shell
kubectl rollout status ds
```

When the rollback is complete, the output is similar to

```
daemonset  successfully rolled out
```

# # Understanding DaemonSet revisions

In the previous `kubectl rollout history` step, you got a list of DaemonSet
revisions. Each revision is stored in a resource named ControllerRevision.

To see what is stored in each revision, find the DaemonSet revision raw
resources

```shell
kubectl get controllerrevision -l
```

This returns a list of ControllerRevisions

```
NAME                               CONTROLLER                     REVISION   AGE
-   DaemonSet     1          1h
-   DaemonSet     2          1h
```

Each ControllerRevision stores the annotations and template of a DaemonSet
revision.

`kubectl rollout undo` takes a specific ControllerRevision and replaces
DaemonSet template with the template stored in the ControllerRevision.
`kubectl rollout undo` is equivalent to updating DaemonSet template to a
previous revision through other commands, such as `kubectl edit` or `kubectl
apply`.

DaemonSet revisions only roll forward. That is to say, after a
rollback completes, the revision number (`.revision` field) of the
ControllerRevision being rolled back to will advance. For example, if you
have revision 1 and 2 in the system, and roll back from revision 2 to revision
1, the ControllerRevision with `.revision 1` will become `.revision 3`.

# # Troubleshooting

* See [troubleshooting DaemonSet rolling
  update](docstasksmanage-daemonupdate-daemon-set#troubleshooting).
