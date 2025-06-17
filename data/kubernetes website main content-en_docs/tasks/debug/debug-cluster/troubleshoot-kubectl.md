---
title Troubleshooting kubectl
content_type task
weight 10
---

This documentation is about investigating and diagnosing
 related issues.
If you encounter issues accessing `kubectl` or connecting to your cluster, this
document outlines various common scenarios and potential solutions to help
identify and address the likely cause.

# #  heading prerequisites

* You need to have a Kubernetes cluster.
* You also need to have `kubectl` installed - see [install tools](docstaskstools#kubectl)

# # Verify kubectl setup

Make sure you have installed and configured `kubectl` correctly on your local machine.
Check the `kubectl` version to ensure it is up-to-date and compatible with your cluster.

Check kubectl version

```shell
kubectl version
```

Youll see a similar output

```console
Client Version version.InfoMajor1, Minor27, GitVersionv1.27.4,GitCommitfa3d7990104d7c1f16943a67f11b154b71f6a132, GitTreeStateclean,BuildDate2023-07-19T122054Z, GoVersiongo1.20.6, Compilergc, Platformlinuxamd64
Kustomize Version v5.0.1
Server Version version.InfoMajor1, Minor27, GitVersionv1.27.3,GitCommit25b4e43193bcda6c7328a6d147b1fb73a33f1598, GitTreeStateclean,BuildDate2023-06-14T094740Z, GoVersiongo1.20.5, Compilergc, Platformlinuxamd64

```

If you see `Unable to connect to the server dial tcp 8443 io timeout`,
instead of `Server Version`, you need to troubleshoot kubectl connectivity with your cluster.

Make sure you have installed the kubectl by following the
[official documentation for installing kubectl](docstaskstools#kubectl), and you have
properly configured the `PATH` environment variable.

# # Check kubeconfig

The `kubectl` requires a `kubeconfig` file to connect to a Kubernetes cluster. The
`kubeconfig` file is usually located under the `.kubeconfig` directory. Make sure
that you have a valid `kubeconfig` file. If you dont have a `kubeconfig` file, you can
obtain it from your Kubernetes administrator, or you can copy it from your Kubernetes
control planes `etckubernetesadmin.conf` directory. If you have deployed your
Kubernetes cluster on a cloud platform and lost your `kubeconfig` file, you can
re-generate it using your cloud providers tools. Refer the cloud providers
documentation for re-generating a `kubeconfig` file.

Check if the `KUBECONFIG` environment variable is configured correctly. You can set
`KUBECONFIG`environment variable or use the `--kubeconfig` parameter with the kubectl
to specify the directory of a `kubeconfig` file.

# # Check VPN connectivity

If you are using a Virtual Private Network (VPN) to access your Kubernetes cluster,
make sure that your VPN connection is active and stable. Sometimes, VPN disconnections
can lead to connection issues with the cluster. Reconnect to the VPN and try accessing
the cluster again.

# # Authentication and authorization

If you are using the token based authentication and the kubectl is returning an error
regarding the authentication token or authentication server address, validate the
Kubernetes authentication token and the authentication server address are configured
properly.

If kubectl is returning an error regarding the authorization, make sure that you are
using the valid user credentials. And you have the permission to access the resource
that you have requested.

# # Verify contexts

Kubernetes supports [multiple clusters and contexts](docstasksaccess-application-clusterconfigure-access-multiple-clusters).
Ensure that you are using the correct context to interact with your cluster.

List available contexts

```shell
kubectl config get-contexts
```

Switch to the appropriate context

```shell
kubectl config use-context
```

# # API server and load balancer

The  server is the
central component of a Kubernetes cluster. If the API server or the load balancer that
runs in front of your API servers is not reachable or not responding, you wont be able
to interact with the cluster.

Check the if the API servers host is reachable by using `ping` command. Check clusters
network connectivity and firewall. If your are using a cloud provider for deploying
the cluster, check your cloud providers health check status for the clusters
API server.

Verify the status of the load balancer (if used) to ensure it is healthy and forwarding
traffic to the API server.

# # TLS problems
* Additional tools required - `base64` and `openssl` version 3.0 or above.

The Kubernetes API server only serves HTTPS requests by default. In that case TLS problems
may occur due to various reasons, such as certificate expiry or chain of trust validity.

You can find the TLS certificate in the kubeconfig file, located in the `.kubeconfig`
directory. The `certificate-authority` attribute contains the CA certificate and the
`client-certificate` attribute contains the client certificate.

Verify the expiry of these certificates

```shell
kubectl config view --flatten --output jsonpath.clusters[0].cluster.certificate-authority-data  base64 -d  openssl x509 -noout -dates
```

output
```console
notBeforeFeb 13 055747 2024 GMT
notAfterFeb 10 060247 2034 GMT
```

```shell
kubectl config view --flatten --output jsonpath.users[0].user.client-certificate-data base64 -d  openssl x509 -noout -dates
```

output
```console
notBeforeFeb 13 055747 2024 GMT
notAfterFeb 12 060250 2025 GMT
```

# # Verify kubectl helpers

Some kubectl authentication helpers provide easy access to Kubernetes clusters. If you
have used such helpers and are facing connectivity issues, ensure that the necessary
configurations are still present.

Check kubectl configuration for authentication details

```shell
kubectl config view
```

If you previously used a helper tool (for example, `kubectl-oidc-login`), ensure that it is still
installed and configured correctly.
