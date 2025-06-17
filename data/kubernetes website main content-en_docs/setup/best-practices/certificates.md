---
title PKI certificates and requirements
reviewers
- sig-cluster-lifecycle
content_type concept
weight 50
---

Kubernetes requires PKI certificates for authentication over TLS.
If you install Kubernetes with [kubeadm](docsreferencesetup-toolskubeadm), the certificates
that your cluster requires are automatically generated.
You can also generate your own certificates -- for example, to keep your private keys more secure
by not storing them on the API server.
This page explains the certificates that your cluster requires.

# # How certificates are used by your cluster

Kubernetes requires PKI for the following operations

# # # Server certificates

* Server certificate for the API server endpoint
* Server certificate for the etcd server
* [Server certificates](docsreferenceaccess-authn-authzkubelet-tls-bootstrapping#client-and-serving-certificates)
  for each kubelet (every  runs a kubelet)
* Optional server certificate for the [front-proxy](docstasksextend-kubernetesconfigure-aggregation-layer)

# # # Client certificates

* Client certificates for each kubelet, used to authenticate to the API server as a client of
  the Kubernetes API
* Client certificate for each API server, used to authenticate to etcd
* Client certificate for the controller manager to securely communicate with the API server
* Client certificate for the scheduler to securely communicate with the API server
* Client certificates, one for each node, for kube-proxy to authenticate to the API server
* Optional client certificates for administrators of the cluster to authenticate to the API server
* Optional client certificate for the [front-proxy](docstasksextend-kubernetesconfigure-aggregation-layer)

# # # Kubelets server and client certificates

To establish a secure connection and authenticate itself to the kubelet, the API Server
requires a client certificate and key pair.

In this scenario, there are two approaches for certificate usage

* Shared Certificates The kube-apiserver can utilize the same certificate and key pair it uses
  to authenticate its clients. This means that the existing certificates, such as `apiserver.crt`
  and `apiserver.key`, can be used for communicating with the kubelet servers.

* Separate Certificates Alternatively, the kube-apiserver can generate a new client certificate
  and key pair to authenticate its communication with the kubelet servers. In this case,
  a distinct certificate named `kubelet-client.crt` and its corresponding private key,
  `kubelet-client.key` are created.

`front-proxy` certificates are required only if you run kube-proxy to support
[an extension API server](docstasksextend-kubernetessetup-extension-api-server).

etcd also implements mutual TLS to authenticate clients and peers.

# # Where certificates are stored

If you install Kubernetes with kubeadm, most certificates are stored in `etckubernetespki`.
All paths in this documentation are relative to that directory, with the exception of user account
certificates which kubeadm places in `etckubernetes`.

# # Configure certificates manually

If you dont want kubeadm to generate the required certificates, you can create them using a
single root CA or by providing all certificates. See [Certificates](docstasksadminister-clustercertificates)
for details on creating your own certificate authority. See
[Certificate Management with kubeadm](docstasksadminister-clusterkubeadmkubeadm-certs)
for more on managing certificates.

# # # Single root CA

You can create a single root CA, controlled by an administrator. This root CA can then create
multiple intermediate CAs, and delegate all further creation to Kubernetes itself.

Required CAs

 Path                    Default CN                 Description
-------------------------------------------------------------------------------------
 ca.crt,key              kubernetes-ca              Kubernetes general CA
 etcdca.crt,key         etcd-ca                    For all etcd-related functions
 front-proxy-ca.crt,key  kubernetes-front-proxy-ca  For the [front-end proxy](docstasksextend-kubernetesconfigure-aggregation-layer)

On top of the above CAs, it is also necessary to get a publicprivate key pair for service account
management, `sa.key` and `sa.pub`.
The following example illustrates the CA key and certificate files shown in the previous table

```
etckubernetespkica.crt
etckubernetespkica.key
etckubernetespkietcdca.crt
etckubernetespkietcdca.key
etckubernetespkifront-proxy-ca.crt
etckubernetespkifront-proxy-ca.key
```

# # # All certificates

If you dont wish to copy the CA private keys to your cluster, you can generate all certificates yourself.

Required certificates

 Default CN                     Parent CA                  O (in Subject)  kind              hosts (SAN)
-------------------------------------------------------------------------------------------------------------------------------------------------
 kube-etcd                      etcd-ca                                    server, client    ``, ``, `localhost`, `127.0.0.1`
 kube-etcd-peer                 etcd-ca                                    server, client    ``, ``, `localhost`, `127.0.0.1`
 kube-etcd-healthcheck-client   etcd-ca                                    client
 kube-apiserver-etcd-client     etcd-ca                                    client
 kube-apiserver                 kubernetes-ca                              server            ``, ``, ``[1]
 kube-apiserver-kubelet-client  kubernetes-ca              systemmasters  client
 front-proxy-client             kubernetes-front-proxy-ca                  client

Instead of using the super-user group `systemmasters` for `kube-apiserver-kubelet-client`
a less privileged group can be used. kubeadm uses the `kubeadmcluster-admins` group for
that purpose.

[1] any other IP or DNS name you contact your cluster on (as used by [kubeadm](docsreferencesetup-toolskubeadm)
the load balancer stable IP andor DNS name, `kubernetes`, `kubernetes.default`, `kubernetes.default.svc`,
`kubernetes.default.svc.cluster`, `kubernetes.default.svc.cluster.local`)

where `kind` maps to one or more of the x509 key usage, which is also documented in the
`.spec.usages` of a [CertificateSigningRequest](docsreferencekubernetes-apiauthentication-resourcescertificate-signing-request-v1#CertificateSigningRequest)
type

 kind    Key usage
-----------------------------------------------------------------------------------------
 server  digital signature, key encipherment, server auth
 client  digital signature, key encipherment, client auth

HostsSAN listed above are the recommended ones for getting a working cluster if required by a
specific setup, it is possible to add additional SANs on all the server certificates.

For kubeadm users only

* The scenario where you are copying to your cluster CA certificates without private keys is
  referred as external CA in the kubeadm documentation.
* If you are comparing the above list with a kubeadm generated PKI, please be aware that
  `kube-etcd`, `kube-etcd-peer` and `kube-etcd-healthcheck-client` certificates are not generated
  in case of external etcd.

# # # Certificate paths

Certificates should be placed in a recommended path (as used by [kubeadm](docsreferencesetup-toolskubeadm)).
Paths should be specified using the given argument regardless of location.

 DefaultCN  recommendedkeypath  recommendedcertpath  command  keyargument  certargument
 ---------  ------------------  -------------------  -------  -----------  ------------
 etcd-ca  etcdca.key  etcdca.crt  kube-apiserver   --etcd-cafile
 kube-apiserver-etcd-client  apiserver-etcd-client.key  apiserver-etcd-client.crt  kube-apiserver  --etcd-keyfile  --etcd-certfile
 kubernetes-ca  ca.key  ca.crt  kube-apiserver   --client-ca-file
 kubernetes-ca  ca.key  ca.crt  kube-controller-manager  --cluster-signing-key-file  --client-ca-file,--root-ca-file,--cluster-signing-cert-file
 kube-apiserver  apiserver.key  apiserver.crt kube-apiserver  --tls-private-key-file  --tls-cert-file
 kube-apiserver-kubelet-client  apiserver-kubelet-client.key  apiserver-kubelet-client.crt  kube-apiserver  --kubelet-client-key  --kubelet-client-certificate
 front-proxy-ca  front-proxy-ca.key  front-proxy-ca.crt  kube-apiserver   --requestheader-client-ca-file
 front-proxy-ca  front-proxy-ca.key  front-proxy-ca.crt  kube-controller-manager   --requestheader-client-ca-file
 front-proxy-client  front-proxy-client.key  front-proxy-client.crt  kube-apiserver  --proxy-client-key-file  --proxy-client-cert-file
 etcd-ca  etcdca.key  etcdca.crt  etcd   --trusted-ca-file,--peer-trusted-ca-file
 kube-etcd  etcdserver.key  etcdserver.crt  etcd  --key-file  --cert-file
 kube-etcd-peer  etcdpeer.key  etcdpeer.crt  etcd  --peer-key-file  --peer-cert-file
 etcd-ca  etcdca.crt  etcdctl   --cacert
 kube-etcd-healthcheck-client  etcdhealthcheck-client.key  etcdhealthcheck-client.crt  etcdctl  --key  --cert

Same considerations apply for the service account key pair

 private key path   public key path   command                  argument
----------------------------------------------------------------------------------------------------
  sa.key                              kube-controller-manager  --service-account-private-key-file
                    sa.pub            kube-apiserver           --service-account-key-file

The following example illustrates the file paths [from the previous tables](#certificate-paths)
you need to provide if you are generating all of your own keys and certificates

```
etckubernetespkietcdca.key
etckubernetespkietcdca.crt
etckubernetespkiapiserver-etcd-client.key
etckubernetespkiapiserver-etcd-client.crt
etckubernetespkica.key
etckubernetespkica.crt
etckubernetespkiapiserver.key
etckubernetespkiapiserver.crt
etckubernetespkiapiserver-kubelet-client.key
etckubernetespkiapiserver-kubelet-client.crt
etckubernetespkifront-proxy-ca.key
etckubernetespkifront-proxy-ca.crt
etckubernetespkifront-proxy-client.key
etckubernetespkifront-proxy-client.crt
etckubernetespkietcdserver.key
etckubernetespkietcdserver.crt
etckubernetespkietcdpeer.key
etckubernetespkietcdpeer.crt
etckubernetespkietcdhealthcheck-client.key
etckubernetespkietcdhealthcheck-client.crt
etckubernetespkisa.key
etckubernetespkisa.pub
```

# # Configure certificates for user accounts

You must manually configure these administrator accounts and service accounts

 Filename                 Credential name             Default CN                           O (in Subject)
------------------------------------------------------------------------------------------------------------------
 admin.conf               default-admin               kubernetes-admin                     ``
 super-admin.conf         default-super-admin         kubernetes-super-admin               systemmasters
 kubelet.conf             default-auth                systemnode`` (see note)  systemnodes
 controller-manager.conf  default-controller-manager  systemkube-controller-manager
 scheduler.conf           default-scheduler           systemkube-scheduler

The value of `` for `kubelet.conf` **must** match precisely the value of the node name
provided by the kubelet as it registers with the apiserver. For further details, read the
[Node Authorization](docsreferenceaccess-authn-authznode).

In the above example `` is implementation specific. Some tools sign the
certificate in the default `admin.conf` to be part of the `systemmasters` group.
`systemmasters` is a break-glass, super user group can bypass the authorization
layer of Kubernetes, such as RBAC. Also some tools do not generate a separate
`super-admin.conf` with a certificate bound to this super user group.

kubeadm generates two separate administrator certificates in kubeconfig files.
One is in `admin.conf` and has `Subject O  kubeadmcluster-admins, CN  kubernetes-admin`.
`kubeadmcluster-admins` is a custom group bound to the `cluster-admin` ClusterRole.
This file is generated on all kubeadm managed control plane machines.

Another is in `super-admin.conf` that has `Subject O  systemmasters, CN  kubernetes-super-admin`.
This file is generated only on the node where `kubeadm init` was called.

1. For each configuration, generate an x509 certificatekey pair with the
   given Common Name (CN) and Organization (O).

1. Run `kubectl` as follows for each configuration

   ```
   KUBECONFIG kubectl config set-cluster default-cluster --serverhttps6443 --certificate-authority  --embed-certs
   KUBECONFIG kubectl config set-credentials  --client-key .pem --client-certificate .pem --embed-certs
   KUBECONFIG kubectl config set-context default-system --cluster default-cluster --user
   KUBECONFIG kubectl config use-context default-system
   ```

These files are used as follows

 Filename                 Command                  Comment
-------------------------------------------------------------------------------------------------------------------------
 admin.conf               kubectl                  Configures administrator user for the cluster
 super-admin.conf         kubectl                  Configures super administrator user for the cluster
 kubelet.conf             kubelet                  One required for each node in the cluster.
 controller-manager.conf  kube-controller-manager  Must be added to manifest in `manifestskube-controller-manager.yaml`
 scheduler.conf           kube-scheduler           Must be added to manifest in `manifestskube-scheduler.yaml`

The following files illustrate full paths to the files listed in the previous table

```
etckubernetesadmin.conf
etckubernetessuper-admin.conf
etckuberneteskubelet.conf
etckubernetescontroller-manager.conf
etckubernetesscheduler.conf
```
