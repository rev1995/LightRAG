---
title Verify Signed Kubernetes Artifacts
content_type task
min-kubernetes-server-version v1.26
weight 420
---

# #  heading prerequisites

You will need to have the following tools installed

- `cosign` ([install guide](httpsdocs.sigstore.devcosignsystem_configinstallation))
- `curl` (often provided by your operating system)
- `jq` ([download jq](httpsjqlang.github.iojqdownload))

# # Verifying binary signatures

The Kubernetes release process signs all binary artifacts (tarballs, SPDX files,
standalone binaries) by using cosigns keyless signing. To verify a particular
binary, retrieve it together with its signature and certificate

```bash
URLhttpsdl.k8s.ioreleasevbinlinuxamd64
BINARYkubectl

FILES(
    BINARY
    BINARY.sig
    BINARY.cert
)

for FILE in FILES[] do
    curl -sSfL --retry 3 --retry-delay 3 URLFILE -o FILE
done
```

Then verify the blob by using `cosign verify-blob`

```shell
cosign verify-blob BINARY
  --signature BINARY.sig
  --certificate BINARY.cert
  --certificate-identity krel-stagingk8s-releng-prod.iam.gserviceaccount.com
  --certificate-oidc-issuer httpsaccounts.google.com
```

Cosign 2.0 requires the `--certificate-identity` and `--certificate-oidc-issuer` options.

To learn more about keyless signing, please refer to [Keyless Signatures](httpsdocs.sigstore.devcosignsigningoverview).

Previous versions of Cosign required that you set `COSIGN_EXPERIMENTAL1`.

For additional information, please refer to the [sigstore Blog](httpsblog.sigstore.devcosign-2-0-released)

# # Verifying image signatures

For a complete list of images that are signed please refer
to [Releases](releasesdownload).

Pick one image from this list and verify its signature using
the `cosign verify` command

```shell
cosign verify registry.k8s.iokube-apiserver-amd64v
  --certificate-identity krel-trustk8s-releng-prod.iam.gserviceaccount.com
  --certificate-oidc-issuer httpsaccounts.google.com
   jq .
```

# # # Verifying images for all control plane components

To verify all signed control plane images for the latest stable version
(v), please run the following commands

```shell
curl -Ls httpssbom.k8s.io(curl -Ls httpsdl.k8s.ioreleasestable.txt)release
   grep SPDXID SPDXRef-Package-registry.k8s.io
   grep -v sha256  cut -d- -f3-  sed s-  sed s-v1v1
   sort  images.txt
inputimages.txt
while IFS read -r image
do
  cosign verify image
    --certificate-identity krel-trustk8s-releng-prod.iam.gserviceaccount.com
    --certificate-oidc-issuer httpsaccounts.google.com
     jq .
done  input
```

Once you have verified an image, you can specify the image by its digest in your Pod
manifests as per this example

```console
registry-urlimage-namesha25645b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2
```

For more information, please refer
to the [Image Pull Policy](docsconceptscontainersimages#image-pull-policy)
section.

# # Verifying Image Signatures with Admission Controller

For non-control plane images (for example
[conformance image](httpsgithub.comkuberneteskubernetesblobmastertestconformanceimageREADME.md)),
signatures can also be verified at deploy time using
[sigstore policy-controller](httpsdocs.sigstore.devpolicy-controlleroverview)
admission controller.

Here are some helpful resources to get started with `policy-controller`

- [Installation](httpsgithub.comsigstorehelm-chartstreemainchartspolicy-controller)
- [Configuration Options](httpsgithub.comsigstorepolicy-controllertreemainconfig)

# # Verify the Software Bill Of Materials

You can verify the Kubernetes Software Bill of Materials (SBOM) by using the
sigstore certificate and signature, or the corresponding SHA files

```shell
# Retrieve the latest available Kubernetes release version
VERSION(curl -Ls httpsdl.k8s.ioreleasestable.txt)

# Verify the SHA512 sum
curl -Ls httpssbom.k8s.ioVERSIONrelease -o VERSION.spdx
echo (curl -Ls httpssbom.k8s.ioVERSIONrelease.sha512) VERSION.spdx  sha512sum --check

# Verify the SHA256 sum
echo (curl -Ls httpssbom.k8s.ioVERSIONrelease.sha256) VERSION.spdx  sha256sum --check

# Retrieve sigstore signature and certificate
curl -Ls httpssbom.k8s.ioVERSIONrelease.sig -o VERSION.spdx.sig
curl -Ls httpssbom.k8s.ioVERSIONrelease.cert -o VERSION.spdx.cert

# Verify the sigstore signature
cosign verify-blob
    --certificate VERSION.spdx.cert
    --signature VERSION.spdx.sig
    --certificate-identity krel-stagingk8s-releng-prod.iam.gserviceaccount.com
    --certificate-oidc-issuer httpsaccounts.google.com
    VERSION.spdx
```
