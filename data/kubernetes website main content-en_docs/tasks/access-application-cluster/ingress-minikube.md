---
title Set up Ingress on Minikube with the NGINX Ingress Controller
content_type task
weight 110
min-kubernetes-server-version 1.19
---

An [Ingress](docsconceptsservices-networkingingress) is an API object that defines rules
which allow external access to services in a cluster. An
[Ingress controller](docsconceptsservices-networkingingress-controllers)
fulfills the rules set in the Ingress.

This page shows you how to set up a simple Ingress which routes requests to Service web or
web2 depending on the HTTP URI.

# #  heading prerequisites

This tutorial assumes that you are using `minikube` to run a local Kubernetes cluster.
Visit [Install tools](docstaskstools#minikube) to learn how to install `minikube`.

This tutorial uses a container that requires the AMD64 architecture.
If you are using minikube on a computer with a different CPU architecture,
you could try using minikube with a driver that can emulate AMD64.
For example, the Docker Desktop driver can do this.

If you are using an older Kubernetes version, switch to the documentation for that version.

# # # Create a minikube cluster

If you havent already set up a cluster locally, run `minikube start` to create a cluster.

# # Enable the Ingress controller

1. To enable the NGINX Ingress controller, run the following command

   ```shell
   minikube addons enable ingress
   ```

1. Verify that the NGINX Ingress controller is running

   ```shell
   kubectl get pods -n ingress-nginx
   ```

   It can take up to a minute before you see these pods running OK.

   The output is similar to

   ```none
   NAME                                        READY   STATUS      RESTARTS    AGE
   ingress-nginx-admission-create-g9g49        01     Completed   0          11m
   ingress-nginx-admission-patch-rqp78         01     Completed   1          11m
   ingress-nginx-controller-59b45fb494-26npt   11     Running     0          11m
   ```

# # Deploy a hello, world app

1. Create a Deployment using the following command

   ```shell
   kubectl create deployment web --imagegcr.iogoogle-sampleshello-app1.0
   ```

   The output should be

   ```none
   deployment.appsweb created
   ```

   Verify that the Deployment is in a Ready state

   ```shell
   kubectl get deployment web
   ```

   The output should be similar to

   ```none
   NAME   READY   UP-TO-DATE   AVAILABLE   AGE
   web    11     1            1           53s
   ```

1. Expose the Deployment

   ```shell
   kubectl expose deployment web --typeNodePort --port8080
   ```

   The output should be

   ```none
   serviceweb exposed
   ```

1. Verify the Service is created and is available on a node port

   ```shell
   kubectl get service web
   ```

   The output is similar to

   ```none
   NAME      TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
   web       NodePort   10.104.133.249           808031637TCP   12m
   ```

1. Visit the Service via NodePort, using the [`minikube service`](httpsminikube.sigs.k8s.iodocshandbookaccessing#using-minikube-service-with-tunnel) command. Follow the instructions for your platform

    tab nameLinux

   ```shell
   minikube service web --url
   ```
   The output is similar to
   ```none
   http172.17.0.1531637
   ```
   Invoke the URL obtained in the output of the previous step
   ```shell
   curl http172.17.0.1531637
   ```
    tab
    tab nameMacOS
   ```shell
   # The command must be run in a separate terminal.
   minikube service web --url
   ```
   The output is similar to
   ```none
   http127.0.0.162445
   ! Because you are using a Docker driver on darwin, the terminal needs to be open to run it.
   ```
   From a different terminal, invoke the URL obtained in the output of the previous step
   ```shell
   curl http127.0.0.162445
   ```
    tab

   The output is similar to

   ```none
   Hello, world!
   Version 1.0.0
   Hostname web-55b8c6998d-8k564
   ```

   You can now access the sample application via the Minikube IP address and NodePort.
   The next step lets you access the application using the Ingress resource.

# # Create an Ingress

The following manifest defines an Ingress that sends traffic to your Service via
`hello-world.example`.

1. Create `example-ingress.yaml` from the following file

    code_sample fileservicenetworkingexample-ingress.yaml

1. Create the Ingress object by running the following command

   ```shell
   kubectl apply -f httpsk8s.ioexamplesservicenetworkingexample-ingress.yaml
   ```

   The output should be

   ```none
   ingress.networking.k8s.ioexample-ingress created
   ```

1. Verify the IP address is set

   ```shell
   kubectl get ingress
   ```

   This can take a couple of minutes.

   You should see an IPv4 address in the `ADDRESS` column for example

   ```none
   NAME              CLASS   HOSTS                 ADDRESS        PORTS   AGE
   example-ingress   nginx   hello-world.example   172.17.0.15    80      38s
   ```

1. Verify that the Ingress controller is directing traffic, by following the instructions for your platform

   The network is limited if using the Docker driver on MacOS (Darwin) and the Node IP is not reachable directly. To get ingress to work youll need to open a new terminal and run `minikube tunnel`.
   `sudo` permission is required for it, so provide the password when prompted.

    tab nameLinux

   ```shell
   curl --resolve hello-world.example80( minikube ip ) -i httphello-world.example
   ```
    tab
    tab nameMacOS

   ```shell
   minikube tunnel
   ```
   The output is similar to

   ```none
   Tunnel successfully started

   NOTE Please do not close this terminal as this process must stay alive for the tunnel to be accessible ...

   The serviceingress example-ingress requires privileged ports to be exposed [80 443]
   sudo permission will be asked for it.
   Starting tunnel for service example-ingress.
   ```

   From within a new terminal, invoke the following command
   ```shell
   curl --resolve hello-world.example80127.0.0.1 -i httphello-world.example
   ```

    tab

   You should see

   ```none
   Hello, world!
   Version 1.0.0
   Hostname web-55b8c6998d-8k564
   ```

1. Optionally, you can also visit `hello-world.example` from your browser.

   Add a line to the bottom of the `etchosts` file on
     your computer (you will need administrator access)

      tab nameLinux
   Look up the external IP address as reported by minikube
   ```none
     minikube ip
   ```

   ```none
     172.17.0.15 hello-world.example
   ```

   Change the IP address to match the output from `minikube ip`.

    tab
      tab nameMacOS
   ```none
   127.0.0.1 hello-world.example
   ```
      tab

     After you make this change, your web browser sends requests for
     `hello-world.example` URLs to Minikube.

# # Create a second Deployment

1. Create another Deployment using the following command

   ```shell
   kubectl create deployment web2 --imagegcr.iogoogle-sampleshello-app2.0
   ```

   The output should be

   ```none
   deployment.appsweb2 created
   ```
   Verify that the Deployment is in a Ready state

   ```shell
   kubectl get deployment web2
   ```

   The output should be similar to

   ```none
   NAME   READY   UP-TO-DATE   AVAILABLE   AGE
   web2   11     1            1           16s
   ```

1. Expose the second Deployment

   ```shell
   kubectl expose deployment web2 --port8080 --typeNodePort
   ```

   The output should be

   ```none
   serviceweb2 exposed
   ```

# # Edit the existing Ingress #edit-ingress

1. Edit the existing `example-ingress.yaml` manifest, and add the
   following lines at the end

    ```yaml
    - path v2
      pathType Prefix
      backend
        service
          name web2
          port
            number 8080
    ```

1. Apply the changes

   ```shell
   kubectl apply -f example-ingress.yaml
   ```

   You should see

   ```none
   ingress.networkingexample-ingress configured
   ```

# # Test your Ingress

1. Access the 1st version of the Hello World app.

    tab nameLinux

   ```shell
   curl --resolve hello-world.example80( minikube ip ) -i httphello-world.example
   ```
    tab
    tab nameMacOS

   ```shell
   minikube tunnel
   ```
   The output is similar to

   ```none
   Tunnel successfully started

   NOTE Please do not close this terminal as this process must stay alive for the tunnel to be accessible ...

   The serviceingress example-ingress requires privileged ports to be exposed [80 443]
   sudo permission will be asked for it.
   Starting tunnel for service example-ingress.
   ```

   From within a new terminal, invoke the following command
   ```shell
   curl --resolve hello-world.example80127.0.0.1 -i httphello-world.example
   ```

    tab

   The output is similar to

   ```none
   Hello, world!
   Version 1.0.0
   Hostname web-55b8c6998d-8k564
   ```

1. Access the 2nd version of the Hello World app.

    tab nameLinux

   ```shell
   curl --resolve hello-world.example80( minikube ip ) -i httphello-world.examplev2
   ```
    tab
    tab nameMacOS

   ```shell
   minikube tunnel
   ```
   The output is similar to

   ```none
   Tunnel successfully started

   NOTE Please do not close this terminal as this process must stay alive for the tunnel to be accessible ...

   The serviceingress example-ingress requires privileged ports to be exposed [80 443]
   sudo permission will be asked for it.
   Starting tunnel for service example-ingress.
   ```

   From within a new terminal, invoke the following command
   ```shell
   curl --resolve hello-world.example80127.0.0.1 -i httphello-world.examplev2
   ```

    tab

   The output is similar to

   ```none
   Hello, world!
   Version 2.0.0
   Hostname web2-75cd47646f-t8cjk
   ```

   If you did the optional step to update `etchosts`, you can also visit `hello-world.example` and
   `hello-world.examplev2` from your browser.

# #  heading whatsnext

* Read more about [Ingress](docsconceptsservices-networkingingress)
* Read more about [Ingress Controllers](docsconceptsservices-networkingingress-controllers)
* Read more about [Services](docsconceptsservices-networkingservice)
