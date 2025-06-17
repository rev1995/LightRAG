---
title Connect a Frontend to a Backend Using Services
content_type tutorial
weight 70
---

This task shows how to create a _frontend_ and a _backend_ microservice. The backend
microservice is a hello greeter. The frontend exposes the backend using nginx and a
Kubernetes  object.

# #  heading objectives

* Create and run a sample `hello` backend microservice using a
   object.
* Use a Service object to send traffic to the backend microservices multiple replicas.
* Create and run a `nginx` frontend microservice, also using a Deployment object.
* Configure the frontend microservice to send traffic to the backend microservice.
* Use a Service object of `typeLoadBalancer` to expose the frontend microservice
  outside the cluster.

# #  heading prerequisites

This task uses
[Services with external load balancers](docstasksaccess-application-clustercreate-external-load-balancer), which
require a supported environment. If your environment does not support this, you can use a Service of type
[NodePort](docsconceptsservices-networkingservice#type-nodeport) instead.

# # Creating the backend using a Deployment

The backend is a simple hello greeter microservice. Here is the configuration
file for the backend Deployment

 code_sample fileserviceaccessbackend-deployment.yaml

Create the backend Deployment

```shell
kubectl apply -f httpsk8s.ioexamplesserviceaccessbackend-deployment.yaml
```

View information about the backend Deployment

```shell
kubectl describe deployment backend
```

The output is similar to this

```
Name                           backend
Namespace                      default
CreationTimestamp              Mon, 24 Oct 2016 142102 -0700
Labels                         apphello
                                tierbackend
                                trackstable
Annotations                    deployment.kubernetes.iorevision1
Selector                       apphello,tierbackend,trackstable
Replicas                       3 desired  3 updated  3 total  3 available  0 unavailable
StrategyType                   RollingUpdate
MinReadySeconds                0
RollingUpdateStrategy          1 max unavailable, 1 max surge
Pod Template
  Labels       apphello
                tierbackend
                trackstable
  Containers
   hello
    Image              gcr.iogoogle-sampleshello-go-gke1.0
    Port               80TCP
    Environment
    Mounts
  Volumes
Conditions
  Type          Status  Reason
  ----          ------  ------
  Available     True    MinimumReplicasAvailable
  Progressing   True    NewReplicaSetAvailable
OldReplicaSets
NewReplicaSet                  hello-3621623197 (33 replicas created)
Events
...
```

# # Creating the `hello` Service object

The key to sending requests from a frontend to a backend is the backend
Service. A Service creates a persistent IP address and DNS name entry
so that the backend microservice can always be reached. A Service uses
 to find
the Pods that it routes traffic to.

First, explore the Service configuration file

 code_sample fileserviceaccessbackend-service.yaml

In the configuration file, you can see that the Service, named `hello` routes
traffic to Pods that have the labels `app hello` and `tier backend`.

Create the backend Service

```shell
kubectl apply -f httpsk8s.ioexamplesserviceaccessbackend-service.yaml
```

At this point, you have a `backend` Deployment running three replicas of your `hello`
application, and you have a Service that can route traffic to them. However, this
service is neither available nor resolvable outside the cluster.

# # Creating the frontend

Now that you have your backend running, you can create a frontend that is accessible
outside the cluster, and connects to the backend by proxying requests to it.

The frontend sends requests to the backend worker Pods by using the DNS name
given to the backend Service. The DNS name is `hello`, which is the value
of the `name` field in the `examplesserviceaccessbackend-service.yaml`
configuration file.

The Pods in the frontend Deployment run a nginx image that is configured
to proxy requests to the `hello` backend Service. Here is the nginx configuration file

 code_sample fileserviceaccessfrontend-nginx.conf

Similar to the backend, the frontend has a Deployment and a Service. An important
difference to notice between the backend and frontend services, is that the
configuration for the frontend Service has `type LoadBalancer`, which means that
the Service uses a load balancer provisioned by your cloud provider and will be
accessible from outside the cluster.

 code_sample fileserviceaccessfrontend-service.yaml

 code_sample fileserviceaccessfrontend-deployment.yaml

Create the frontend Deployment and Service

```shell
kubectl apply -f httpsk8s.ioexamplesserviceaccessfrontend-deployment.yaml
kubectl apply -f httpsk8s.ioexamplesserviceaccessfrontend-service.yaml
```

The output verifies that both resources were created

```
deployment.appsfrontend created
servicefrontend created
```

The nginx configuration is baked into the
[container image](examplesserviceaccessDockerfile). A better way to do this would
be to use a
[ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap),
so that you can change the configuration more easily.

# # Interact with the frontend Service

Once youve created a Service of type LoadBalancer, you can use this
command to find the external IP

```shell
kubectl get service frontend --watch
```

This displays the configuration for the `frontend` Service and watches for
changes. Initially, the external IP is listed as ``

```
NAME       TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)  AGE
frontend   LoadBalancer   10.51.252.116        80TCP   10s
```

As soon as an external IP is provisioned, however, the configuration updates
to include the new IP under the `EXTERNAL-IP` heading

```
NAME       TYPE           CLUSTER-IP      EXTERNAL-IP        PORT(S)  AGE
frontend   LoadBalancer   10.51.252.116   XXX.XXX.XXX.XXX    80TCP   1m
```

That IP can now be used to interact with the `frontend` service from outside the
cluster.

# # Send traffic through the frontend

The frontend and backend are now connected. You can hit the endpoint
by using the curl command on the external IP of your frontend Service.

```shell
curl httpEXTERNAL_IP # replace this with the EXTERNAL-IP you saw earlier
```

The output shows the message generated by the backend

```json
messageHello
```

# #  heading cleanup

To delete the Services, enter this command

```shell
kubectl delete services frontend backend
```

To delete the Deployments, the ReplicaSets and the Pods that are running the backend and frontend applications, enter this command

```shell
kubectl delete deployment frontend backend
```

# #  heading whatsnext

* Learn more about [Services](docsconceptsservices-networkingservice)
* Learn more about [ConfigMaps](docstasksconfigure-pod-containerconfigure-pod-configmap)
* Learn more about [DNS for Service and Pods](docsconceptsservices-networkingdns-pod-service)
