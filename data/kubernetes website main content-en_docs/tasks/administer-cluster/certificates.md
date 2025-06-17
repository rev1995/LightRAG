---
title Generate Certificates Manually
content_type task
weight 30
---

When using client certificate authentication, you can generate certificates
manually through [`easyrsa`](httpsgithub.comOpenVPNeasy-rsa), [`openssl`](httpsgithub.comopensslopenssl) or [`cfssl`](httpsgithub.comcloudflarecfssl).

# # # easyrsa

**easyrsa** can manually generate certificates for your cluster.

1. Download, unpack, and initialize the patched version of `easyrsa3`.

   ```shell
   curl -LO httpsdl.k8s.ioeasy-rsaeasy-rsa.tar.gz
   tar xzf easy-rsa.tar.gz
   cd easy-rsa-mastereasyrsa3
   .easyrsa init-pki
   ```
1. Generate a new certificate authority (CA). `--batch` sets automatic mode
   `--req-cn` specifies the Common Name (CN) for the CAs new root certificate.

   ```shell
   .easyrsa --batch --req-cnMASTER_IP`date s` build-ca nopass
   ```

1. Generate server certificate and key.

   The argument `--subject-alt-name` sets the possible IPs and DNS names the API server will
   be accessed with. The `MASTER_CLUSTER_IP` is usually the first IP from the service CIDR
   that is specified as the `--service-cluster-ip-range` argument for both the API server and
   the controller manager component. The argument `--days` is used to set the number of days
   after which the certificate expires.
   The sample below also assumes that you are using `cluster.local` as the default
   DNS domain name.

   ```shell
   .easyrsa --subject-alt-nameIPMASTER_IP,
   IPMASTER_CLUSTER_IP,
   DNSkubernetes,
   DNSkubernetes.default,
   DNSkubernetes.default.svc,
   DNSkubernetes.default.svc.cluster,
   DNSkubernetes.default.svc.cluster.local
   --days10000
   build-server-full server nopass
   ```

1. Copy `pkica.crt`, `pkiissuedserver.crt`, and `pkiprivateserver.key` to your directory.

1. Fill in and add the following parameters into the API server start parameters

   ```shell
   --client-ca-fileyourdirectoryca.crt
   --tls-cert-fileyourdirectoryserver.crt
   --tls-private-key-fileyourdirectoryserver.key
   ```

# # # openssl

**openssl** can manually generate certificates for your cluster.

1. Generate a ca.key with 2048bit

   ```shell
   openssl genrsa -out ca.key 2048
   ```

1. According to the ca.key generate a ca.crt (use `-days` to set the certificate effective time)

   ```shell
   openssl req -x509 -new -nodes -key ca.key -subj CNMASTER_IP -days 10000 -out ca.crt
   ```

1. Generate a server.key with 2048bit

   ```shell
   openssl genrsa -out server.key 2048
   ```

1. Create a config file for generating a Certificate Signing Request (CSR).

   Be sure to substitute the values marked with angle brackets (e.g. ``)
   with real values before saving this to a file (e.g. `csr.conf`).
   Note that the value for `MASTER_CLUSTER_IP` is the service cluster IP for the
   API server as described in previous subsection.
   The sample below also assumes that you are using `cluster.local` as the default
   DNS domain name.

   ```ini
   [ req ]
   default_bits  2048
   prompt  no
   default_md  sha256
   req_extensions  req_ext
   distinguished_name  dn

   [ dn ]
   C
   ST
   L
   O
   OU
   CN

   [ req_ext ]
   subjectAltName  alt_names

   [ alt_names ]
   DNS.1  kubernetes
   DNS.2  kubernetes.default
   DNS.3  kubernetes.default.svc
   DNS.4  kubernetes.default.svc.cluster
   DNS.5  kubernetes.default.svc.cluster.local
   IP.1
   IP.2

   [ v3_ext ]
   authorityKeyIdentifierkeyid,issueralways
   basicConstraintsCAFALSE
   keyUsagekeyEncipherment,dataEncipherment
   extendedKeyUsageserverAuth,clientAuth
   subjectAltNamealt_names
   ```

1. Generate the certificate signing request based on the config file

   ```shell
   openssl req -new -key server.key -out server.csr -config csr.conf
   ```

1. Generate the server certificate using the ca.key, ca.crt and server.csr

   ```shell
   openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key
       -CAcreateserial -out server.crt -days 10000
       -extensions v3_ext -extfile csr.conf -sha256
   ```

1. View the certificate signing request

   ```shell
   openssl req  -noout -text -in .server.csr
   ```

1. View the certificate

   ```shell
   openssl x509  -noout -text -in .server.crt
   ```

Finally, add the same parameters into the API server start parameters.

# # # cfssl

**cfssl** is another tool for certificate generation.

1. Download, unpack and prepare the command line tools as shown below.

   Note that you may need to adapt the sample commands based on the hardware
   architecture and cfssl version you are using.

   ```shell
   curl -L httpsgithub.comcloudflarecfsslreleasesdownloadv1.5.0cfssl_1.5.0_linux_amd64 -o cfssl
   chmod x cfssl
   curl -L httpsgithub.comcloudflarecfsslreleasesdownloadv1.5.0cfssljson_1.5.0_linux_amd64 -o cfssljson
   chmod x cfssljson
   curl -L httpsgithub.comcloudflarecfsslreleasesdownloadv1.5.0cfssl-certinfo_1.5.0_linux_amd64 -o cfssl-certinfo
   chmod x cfssl-certinfo
   ```

1. Create a directory to hold the artifacts and initialize cfssl

   ```shell
   mkdir cert
   cd cert
   ..cfssl print-defaults config  config.json
   ..cfssl print-defaults csr  csr.json
   ```

1. Create a JSON config file for generating the CA file, for example, `ca-config.json`

   ```json

     signing
       default
         expiry 8760h
       ,
       profiles
         kubernetes
           usages [
             signing,
             key encipherment,
             server auth,
             client auth
           ],
           expiry 8760h

   ```

1. Create a JSON config file for CA certificate signing request (CSR), for example,
   `ca-csr.json`. Be sure to replace the values marked with angle brackets with
   real values you want to use.

   ```json

     CN kubernetes,
     key
       algo rsa,
       size 2048
     ,
     names[
       C ,
       ST ,
       L ,
       O ,
       OU
     ]

   ```

1. Generate CA key (`ca-key.pem`) and certificate (`ca.pem`)

   ```shell
   ..cfssl gencert -initca ca-csr.json  ..cfssljson -bare ca
   ```

1. Create a JSON config file for generating keys and certificates for the API
   server, for example, `server-csr.json`. Be sure to replace the values in angle brackets with
   real values you want to use. The `` is the service cluster
   IP for the API server as described in previous subsection.
   The sample below also assumes that you are using `cluster.local` as the default
   DNS domain name.

   ```json

     CN kubernetes,
     hosts [
       127.0.0.1,
       ,
       ,
       kubernetes,
       kubernetes.default,
       kubernetes.default.svc,
       kubernetes.default.svc.cluster,
       kubernetes.default.svc.cluster.local
     ],
     key
       algo rsa,
       size 2048
     ,
     names [
       C ,
       ST ,
       L ,
       O ,
       OU
     ]

   ```

1. Generate the key and certificate for the API server, which are by default
   saved into file `server-key.pem` and `server.pem` respectively

   ```shell
   ..cfssl gencert -caca.pem -ca-keyca-key.pem
        --configca-config.json -profilekubernetes
        server-csr.json  ..cfssljson -bare server
   ```

# # Distributing Self-Signed CA Certificate

A client node may refuse to recognize a self-signed CA certificate as valid.
For a non-production deployment, or for a deployment that runs behind a company
firewall, you can distribute a self-signed CA certificate to all clients and
refresh the local list for valid certificates.

On each client, perform the following operations

```shell
sudo cp ca.crt usrlocalshareca-certificateskubernetes.crt
sudo update-ca-certificates
```

```none
Updating certificates in etcsslcerts...
1 added, 0 removed done.
Running hooks in etcca-certificatesupdate.d....
done.
```

# # Certificates API

You can use the `certificates.k8s.io` API to provision
x509 certificates to use for authentication as documented
in the [Managing TLS in a cluster](docstaskstlsmanaging-tls-in-a-cluster)
task page.
