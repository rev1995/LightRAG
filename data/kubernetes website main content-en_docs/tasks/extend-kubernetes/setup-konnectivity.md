---
title Set up Konnectivity service
content_type task
weight 70
---

The Konnectivity service provides a TCP level proxy for the control plane to cluster
communication.

# #  heading prerequisites

You need to have a Kubernetes cluster, and the kubectl command-line tool must
be configured to communicate with your cluster. It is recommended to run this
tutorial on a cluster with at least two nodes that are not acting as control
plane hosts. If you do not already have a cluster, you can create one by using
[minikube](httpsminikube.sigs.k8s.iodocstutorialsmulti_node).

# # Configure the Konnectivity service

The following steps require an egress configuration, for example

 code_sample fileadminkonnectivityegress-selector-configuration.yaml

You need to configure the API Server to use the Konnectivity service
and direct the network traffic to the cluster nodes

1. Make sure that
[Service Account Token Volume Projection](docstasksconfigure-pod-containerconfigure-service-account#serviceaccount-token-volume-projection)
feature enabled in your cluster. It is enabled by default since Kubernetes v1.20.
1. Create an egress configuration file such as `adminkonnectivityegress-selector-configuration.yaml`.
1. Set the `--egress-selector-config-file` flag of the API Server to the path of
your API Server egress configuration file.
1. If you use UDS connection, add volumes config to the kube-apiserver
   ```yaml
   spec
     containers
       volumeMounts
       - name konnectivity-uds
         mountPath etckuberneteskonnectivity-server
         readOnly false
     volumes
     - name konnectivity-uds
       hostPath
         path etckuberneteskonnectivity-server
         type DirectoryOrCreate
   ```

Generate or obtain a certificate and kubeconfig for konnectivity-server.
For example, you can use the OpenSSL command line tool to issue a X.509 certificate,
using the cluster CA certificate `etckubernetespkica.crt` from a control-plane host.

```bash
openssl req -subj CNsystemkonnectivity-server -new -newkey rsa2048 -nodes -out konnectivity.csr -keyout konnectivity.key
openssl x509 -req -in konnectivity.csr -CA etckubernetespkica.crt -CAkey etckubernetespkica.key -CAcreateserial -out konnectivity.crt -days 375 -sha256
SERVER(kubectl config view -o jsonpath.clusters..server)
kubectl --kubeconfig etckuberneteskonnectivity-server.conf config set-credentials systemkonnectivity-server --client-certificate konnectivity.crt --client-key konnectivity.key --embed-certstrue
kubectl --kubeconfig etckuberneteskonnectivity-server.conf config set-cluster kubernetes --server SERVER --certificate-authority etckubernetespkica.crt --embed-certstrue
kubectl --kubeconfig etckuberneteskonnectivity-server.conf config set-context systemkonnectivity-serverkubernetes --cluster kubernetes --user systemkonnectivity-server
kubectl --kubeconfig etckuberneteskonnectivity-server.conf config use-context systemkonnectivity-serverkubernetes
rm -f konnectivity.crt konnectivity.key konnectivity.csr
```

Next, you need to deploy the Konnectivity server and agents.
[kubernetes-sigsapiserver-network-proxy](httpsgithub.comkubernetes-sigsapiserver-network-proxy)
is a reference implementation.

Deploy the Konnectivity server on your control plane node. The provided
`konnectivity-server.yaml` manifest assumes
that the Kubernetes components are deployed as a  in your cluster. If not, you can deploy the Konnectivity
server as a DaemonSet.

 code_sample fileadminkonnectivitykonnectivity-server.yaml

Then deploy the Konnectivity agents in your cluster

 code_sample fileadminkonnectivitykonnectivity-agent.yaml

Last, if RBAC is enabled in your cluster, create the relevant RBAC rules

 code_sample fileadminkonnectivitykonnectivity-rbac.yaml
