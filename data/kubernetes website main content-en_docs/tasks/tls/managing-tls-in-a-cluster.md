---
title Manage TLS Certificates in a Cluster
content_type task
reviewers
- mikedanese
- beacham
- liggit
---

Kubernetes provides a `certificates.k8s.io` API, which lets you provision TLS
certificates signed by a Certificate Authority (CA) that you control. These CA
and certificates can be used by your workloads to establish trust.

`certificates.k8s.io` API uses a protocol that is similar to the [ACME
draft](httpsgithub.comietf-wg-acmeacme).

Certificates created using the `certificates.k8s.io` API are signed by a
[dedicated CA](#configuring-your-cluster-to-provide-signing). It is possible to configure your cluster to use the cluster root
CA for this purpose, but you should never rely on this. Do not assume that
these certificates will validate against the cluster root CA.

# #  heading prerequisites

You need the `cfssl` tool. You can download `cfssl` from
[httpsgithub.comcloudflarecfsslreleases](httpsgithub.comcloudflarecfsslreleases).

Some steps in this page use the `jq` tool. If you dont have `jq`, you can
install it via your operating systems software sources, or fetch it from
[httpsjqlang.github.iojq](httpsjqlang.github.iojq).

# # Trusting TLS in a cluster

Trusting the [custom CA](#configuring-your-cluster-to-provide-signing) from an application running as a pod usually requires
some extra application configuration. You will need to add the CA certificate
bundle to the list of CA certificates that the TLS client or server trusts. For
example, you would do this with a golang TLS config by parsing the certificate
chain and adding the parsed certificates to the `RootCAs` field in the
[`tls.Config`](httpspkg.go.devcryptotls#Config) struct.

Even though the custom CA certificate may be included in the filesystem (in the
ConfigMap `kube-root-ca.crt`),
you should not use that certificate authority for any purpose other than to verify internal
Kubernetes endpoints. An example of an internal Kubernetes endpoint is the
Service named `kubernetes` in the default namespace.

If you want to use a custom certificate authority for your workloads, you should generate
that CA separately, and distribute its CA certificate using a
[ConfigMap](docstasksconfigure-pod-containerconfigure-pod-configmap) that your pods
have access to read.

# # Requesting a certificate

The following section demonstrates how to create a TLS certificate for a
Kubernetes service accessed through DNS.

This tutorial uses CFSSL Cloudflares PKI and TLS toolkit [click here](httpsblog.cloudflare.comintroducing-cfssl) to know more.

# # Create a certificate signing request

Generate a private key and certificate signing request (or CSR) by running
the following command

```shell
cat
Annotations
CreationTimestamp      Tue, 01 Feb 2022 114915 -0500
Requesting User        yournameexample.com
Signer                 example.comserving
Status                 Pending
Subject
        Common Name    my-pod.my-namespace.pod.cluster.local
        Serial Number
Subject Alternative Names
        DNS Names      my-pod.my-namespace.pod.cluster.local
                        my-svc.my-namespace.svc.cluster.local
        IP Addresses   192.0.2.24
                        10.0.34.2
Events
```

# # Get the CertificateSigningRequest approved #get-the-certificate-signing-request-approved

Approving the [certificate signing request](docsreferenceaccess-authn-authzcertificate-signing-requests)
is either done by an automated approval process or on a one off basis by a cluster
administrator. If youre authorized to approve a certificate request, you can do that
manually using `kubectl` for example

```shell
kubectl certificate approve my-svc.my-namespace
```

```none
certificatesigningrequest.certificates.k8s.iomy-svc.my-namespace approved
```

You should now see the following

```shell
kubectl get csr
```

```none
NAME                  AGE   SIGNERNAME            REQUESTOR              REQUESTEDDURATION   CONDITION
my-svc.my-namespace   10m   example.comserving   yournameexample.com                 Approved
```

This means the certificate request has been approved and is waiting for the
requested signer to sign it.

# # Sign the CertificateSigningRequest #sign-the-certificate-signing-request

Next, youll play the part of a certificate signer, issue the certificate, and upload it to the API.

A signer would typically watch the CertificateSigningRequest API for objects with its `signerName`,
check that they have been approved, sign certificates for those requests,
and update the API object status with the issued certificate.

# # # Create a Certificate Authority

You need an authority to provide the digital signature on the new certificate.

First, create a signing certificate by running the following

```shell
cat
This uses the command line tool [`jq`](httpsjqlang.github.iojq) to populate the base64-encoded
content in the `.status.certificate` field.
If you do not have `jq`, you can also save the JSON output to a file, populate this field manually, and
upload the resulting file.

Once the CSR is approved and the signed certificate is uploaded, run

```shell
kubectl get csr
```

The output is similar to
```none
NAME                  AGE   SIGNERNAME            REQUESTOR              REQUESTEDDURATION   CONDITION
my-svc.my-namespace   20m   example.comserving   yournameexample.com                 Approved,Issued
```

# # Download the certificate and use it

Now, as the requesting user, you can download the issued certificate
and save it to a `server.crt` file by running the following

```shell
kubectl get csr my-svc.my-namespace -o jsonpath.status.certificate
     base64 --decode  server.crt
```

Now you can populate `server.crt` and `server-key.pem` in a

that you could later mount into a Pod (for example, to use with a webserver
that serves HTTPS).

```shell
kubectl create secret tls server --cert server.crt --key server-key.pem
```

```none
secretserver created
```

Finally, you can populate `ca.pem` into a
and use it as the trust root to verify the serving certificate

```shell
kubectl create configmap example-serving-ca --from-file ca.crtca.pem
```

```none
configmapexample-serving-ca created
```

# # Approving CertificateSigningRequests #approving-certificate-signing-requests

A Kubernetes administrator (with appropriate permissions) can manually approve
(or deny) CertificateSigningRequests by using the `kubectl certificate
approve` and `kubectl certificate deny` commands. However if you intend
to make heavy usage of this API, you might consider writing an automated
certificates controller.

The ability to approve CSRs decides who trusts whom within your environment. The
ability to approve CSRs should not be granted broadly or lightly.

You should make sure that you confidently understand both the verification requirements
that fall on the approver **and** the repercussions of issuing a specific certificate
before you grant the `approve` permission.

Whether a machine or a human using kubectl as above, the role of the _approver_ is
to verify that the CSR satisfies two requirements

1. The subject of the CSR controls the private key used to sign the CSR. This
   addresses the threat of a third party masquerading as an authorized subject.
   In the above example, this step would be to verify that the pod controls the
   private key used to generate the CSR.
2. The subject of the CSR is authorized to act in the requested context. This
   addresses the threat of an undesired subject joining the cluster. In the
   above example, this step would be to verify that the pod is allowed to
   participate in the requested service.

If and only if these two requirements are met, the approver should approve
the CSR and otherwise should deny the CSR.

For more information on certificate approval and access control, read
the [Certificate Signing Requests](docsreferenceaccess-authn-authzcertificate-signing-requests)
reference page.

# # Configuring your cluster to provide signing

This page assumes that a signer is set up to serve the certificates API. The
Kubernetes controller manager provides a default implementation of a signer. To
enable it, pass the `--cluster-signing-cert-file` and
`--cluster-signing-key-file` parameters to the controller manager with paths to
your Certificate Authoritys keypair.
