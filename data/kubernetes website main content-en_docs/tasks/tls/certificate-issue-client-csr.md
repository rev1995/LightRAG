---
title Issue a Certificate for a Kubernetes API Client Using A CertificateSigningRequest
api_metadata
- apiVersion certificates.k8s.iov1
  kind CertificateSigningRequest
  override_link_text CSR v1
weight 80

# Docs maintenance note
#
# If there is a future page docstaskstlscertificate-issue-client-manually then this page
# should link there, and the new page should link back to this one.
---

Kubernetes lets you use a public key infrastructure (PKI) to authenticate to your cluster
as a client.

A few steps are required in order to get a normal user to be able to
authenticate and invoke an API. First, this user must have an [X.509](httpswww.itu.intrecT-REC-X.509) certificate
issued by an authority that your Kubernetes cluster trusts. The client must then present that certificate to the Kubernetes API.

You use a [CertificateSigningRequest](conceptssecuritycertificate-signing-requests)
as part of this process, and either you or some other principal must approve the request.

You will create a private key, and then get a certificate issued, and finally configure
that private key for a client.

# #  heading prerequisites

*

* You need the `kubectl`, `openssl` and `base64` utilities.

This page assumes you are using Kubernetes  (RBAC).
If you have alternative or additional security mechanisms around authorization, you need to account for those as well.

# # Create private key

In this step, you create a private key. You need to keep this document secret anyone who has it can impersonate the user.

```shell
# Create a private key
openssl genrsa -out myuser.key 3072
```

# # Create an X.509 certificate signing request #create-x.509-certificatessigningrequest

This is not the same as the similarly-named CertificateSigningRequest API the file you generate here goes into the
CertificateSigningRequest.

It is important to set CN and O attribute of the CSR. CN is the name of the user and O is the group that this user will belong to.
You can refer to [RBAC](docsreferenceaccess-authn-authzrbac) for standard groups.

```shell
# Change the common name myuser to the actual username that you want to use
openssl req -new -key myuser.key -out myuser.csr -subj CNmyuser
```

# # Create a Kubernetes CertificateSigningRequest #create-k8s-certificatessigningrequest

Encode the CSR document using this command

```shell
cat myuser.csr  base64  tr -d n
```

Create a [CertificateSigningRequest](docsreferencekubernetes-apiauthentication-resourcescertificate-signing-request-v1)
and submit it to a Kubernetes Cluster via kubectl. Below is a snippet of shell that you can use to generate the
CertificateSigningRequest.

```shell
cat  myuser.crt
```

# # Configure the certificate into kubeconfig

The next step is to add this user into the kubeconfig file.

First, you need to add new credentials

```shell
kubectl config set-credentials myuser --client-keymyuser.key --client-certificatemyuser.crt --embed-certstrue

```

Then, you need to add the context

```shell
kubectl config set-context myuser --clusterkubernetes --usermyuser
```

To test it

```shell
kubectl --context myuser auth whoami
```

You should see output confirming that you are myuser.

# # Create Role and RoleBinding

If you dont use Kubernetes RBAC, skip this step and make the appropriate changes for the authorization mechanism
your cluster actually uses.

With the certificate created it is time to define the Role and RoleBinding for
this user to access Kubernetes cluster resources.

This is a sample command to create a Role for this new user

```shell
kubectl create role developer --verbcreate --verbget --verblist --verbupdate --verbdelete --resourcepods
```

This is a sample command to create a RoleBinding for this new user

```shell
kubectl create rolebinding developer-binding-myuser --roledeveloper --usermyuser
```

# #  heading whatsnext

* Read [Manage TLS Certificates in a Cluster](docstaskstlsmanaging-tls-in-a-cluster)
* For details of X.509 itself, refer to [RFC 5280](httpstools.ietf.orghtmlrfc5280#section-3.1) section 3.1
* For information on the syntax of PKCS#10 certificate signing requests, refer to [RFC 2986](httpstools.ietf.orghtmlrfc2986)
* Read about [ClusterTrustBundles](docsreferenceaccess-authn-authzcertificate-signing-requests#cluster-trust-bundles)
