---
title Use a Service to Access an Application in a Cluster
content_type tutorial
weight 60
---

This page shows how to create a Kubernetes Service object that external
clients can use to access an application running in a cluster. The Service
provides load balancing for an application that has two running instances.

# #  heading prerequisites

# #  heading objectives

- Run two instances of a Hello World application.
- Create a Service object that exposes a node port.
- Use the Service object to access the running application.

# # Creating a service for an application running in two pods

Here is the configuration file for the application Deployment

 code_sample fileserviceaccesshello-application.yaml

1. Run a Hello World application in your cluster
   Create the application Deployment using the file above

   ```shell
   kubectl apply -f httpsk8s.ioexamplesserviceaccesshello-application.yaml
   ```

   The preceding command creates a

   and an associated
   .
   The ReplicaSet has two

   each of which runs the Hello World application.

1. Display information about the Deployment

   ```shell
   kubectl get deployments hello-world
   kubectl describe deployments hello-world
   ```

1. Display information about your ReplicaSet objects

   ```shell
   kubectl get replicasets
   kubectl describe replicasets
   ```

1. Create a Service object that exposes the deployment

   ```shell
   kubectl expose deployment hello-world --typeNodePort --nameexample-service
   ```

1. Display information about the Service

   ```shell
   kubectl describe services example-service
   ```

   The output is similar to this

   ```none
   Name                   example-service
   Namespace              default
   Labels                 runload-balancer-example
   Annotations
   Selector               runload-balancer-example
   Type                   NodePort
   IP                     10.32.0.16
   Port                    8080TCP
   TargetPort             8080TCP
   NodePort                31496TCP
   Endpoints              10.200.1.48080,10.200.2.58080
   Session Affinity       None
   Events
   ```

   Make a note of the NodePort value for the Service. For example,
   in the preceding output, the NodePort value is 31496.

1. List the pods that are running the Hello World application

   ```shell
   kubectl get pods --selectorrunload-balancer-example --outputwide
   ```

   The output is similar to this

   ```none
   NAME                           READY   STATUS    ...  IP           NODE
   hello-world-2895499144-bsbk5   11     Running   ...  10.200.1.4   worker1
   hello-world-2895499144-m1pwt   11     Running   ...  10.200.2.5   worker2
   ```

1. Get the public IP address of one of your nodes that is running
   a Hello World pod. How you get this address depends on how you set
   up your cluster. For example, if you are using Minikube, you can
   see the node address by running `kubectl cluster-info`. If you are
   using Google Compute Engine instances, you can use the
   `gcloud compute instances list` command to see the public addresses of your
   nodes.

1. On your chosen node, create a firewall rule that allows TCP traffic
   on your node port. For example, if your Service has a NodePort value of
   31568, create a firewall rule that allows TCP traffic on port 31568. Different
   cloud providers offer different ways of configuring firewall rules.

1. Use the node address and node port to access the Hello World application

   ```shell
   curl http
   ```

   where `` is the public IP address of your node,
   and `` is the NodePort value for your service. The
   response to a successful request is a hello message

   ```none
   Hello, world!
   Version 2.0.0
   Hostname hello-world-cdd4458f4-m47c8
   ```

# # Using a service configuration file

As an alternative to using `kubectl expose`, you can use a
[service configuration file](docsconceptsservices-networkingservice)
to create a Service.

# #  heading cleanup

To delete the Service, enter this command

    kubectl delete services example-service

To delete the Deployment, the ReplicaSet, and the Pods that are running
the Hello World application, enter this command

    kubectl delete deployment hello-world

# #  heading whatsnext

Follow the
[Connecting Applications with Services](docstutorialsservicesconnect-applications-service)
tutorial.
