---
title Configure Pod Initialization
content_type task
weight 170
---

This page shows how to use an Init Container to initialize a Pod before an
application Container runs.

# #  heading prerequisites

# # Create a Pod that has an Init Container

In this exercise you create a Pod that has one application Container and one
Init Container. The init container runs to completion before the application
container starts.

Here is the configuration file for the Pod

 code_sample filepodsinit-containers.yaml

In the configuration file, you can see that the Pod has a Volume that the init
container and the application container share.

The init container mounts the
shared Volume at `work-dir`, and the application container mounts the shared
Volume at `usrsharenginxhtml`. The init container runs the following command
and then terminates

```shell
wget -O work-dirindex.html httpinfo.cern.ch
```

Notice that the init container writes the `index.html` file in the root directory
of the nginx server.

Create the Pod

```shell
kubectl apply -f httpsk8s.ioexamplespodsinit-containers.yaml
```

Verify that the nginx container is running

```shell
kubectl get pod init-demo
```

The output shows that the nginx container is running

```
NAME        READY     STATUS    RESTARTS   AGE
init-demo   11       Running   0          1m
```

Get a shell into the nginx container running in the init-demo Pod

```shell
kubectl exec -it init-demo -- binbash
```

In your shell, send a GET request to the nginx server

```
rootnginx# apt-get update
rootnginx# apt-get install curl
rootnginx# curl localhost
```

The output shows that nginx is serving the web page that was written by the init container

```html

httpinfo.cern.ch

httpinfo.cern.ch - home of the first website
  ...
  Browse the first website
  ...
```

# #  heading whatsnext

* Learn more about
  [communicating between Containers running in the same Pod](docstasksaccess-application-clustercommunicate-containers-same-pod-shared-volume).
* Learn more about [Init Containers](docsconceptsworkloadspodsinit-containers).
* Learn more about [Volumes](docsconceptsstoragevolumes).
* Learn more about [Debugging Init Containers](docstasksdebugdebug-applicationdebug-init-containers)
