---
title Security Checklist
description
  Baseline checklist for ensuring security in Kubernetes clusters.
content_type concept
weight 100
---

This checklist aims at providing a basic list of guidance with links to more
comprehensive documentation on each topic. It does not claim to be exhaustive
and is meant to evolve.

On how to read and use this document

- The order of topics does not reflect an order of priority.
- Some checklist items are detailed in the paragraph below the list of each section.

Checklists are **not** sufficient for attaining a good security posture on their
own. A good security posture requires constant attention and improvement, but a
checklist can be the first step on the never-ending journey towards security
preparedness. Some of the recommendations in this checklist may be too
restrictive or too lax for your specific security needs. Since Kubernetes
security is not one size fits all, each category of checklist items should be
evaluated on its merits.

# # Authentication  Authorization

- [ ] `systemmasters` group is not used for user or component authentication after bootstrapping.
- [ ] The kube-controller-manager is running with `--use-service-account-credentials`
  enabled.
- [ ] The root certificate is protected (either an offline CA, or a managed
  online CA with effective access controls).
- [ ] Intermediate and leaf certificates have an expiry date no more than 3
  years in the future.
- [ ] A process exists for periodic access review, and reviews occur no more
  than 24 months apart.
- [ ] The [Role Based Access Control Good Practices](docsconceptssecurityrbac-good-practices)
  are followed for guidance related to authentication and authorization.

After bootstrapping, neither users nor components should authenticate to the
Kubernetes API as `systemmasters`. Similarly, running all of
kube-controller-manager as `systemmasters` should be avoided. In fact,
`systemmasters` should only be used as a break-glass mechanism, as opposed to
an admin user.

# # Network security

- [ ] CNI plugins in use support network policies.
- [ ] Ingress and egress network policies are applied to all workloads in the
  cluster.
- [ ] Default network policies within each namespace, selecting all pods, denying
  everything, are in place.
- [ ] If appropriate, a service mesh is used to encrypt all communications inside of the cluster.
- [ ] The Kubernetes API, kubelet API and etcd are not exposed publicly on Internet.
- [ ] Access from the workloads to the cloud metadata API is filtered.
- [ ] Use of LoadBalancer and ExternalIPs is restricted.

A number of [Container Network Interface (CNI) plugins](docsconceptsextend-kubernetescompute-storage-netnetwork-plugins)
plugins provide the functionality to
restrict network resources that pods may communicate with. This is most commonly done
through [Network Policies](docsconceptsservices-networkingnetwork-policies)
which provide a namespaced resource to define rules. Default network policies
that block all egress and ingress, in each namespace, selecting all pods, can be
useful to adopt an allow list approach to ensure that no workloads are missed.

Not all CNI plugins provide encryption in transit. If the chosen plugin lacks this
feature, an alternative solution could be to use a service mesh to provide that
functionality.

The etcd datastore of the control plane should have controls to limit access and
not be publicly exposed on the Internet. Furthermore, mutual TLS (mTLS) should
be used to communicate securely with it. The certificate authority for this
should be unique to etcd.

External Internet access to the Kubernetes API server should be restricted to
not expose the API publicly. Be careful, as many managed Kubernetes distributions
are publicly exposing the API server by default. You can then use a bastion host
to access the server.

The [kubelet](docsreferencecommand-line-tools-referencekubelet) API access
should be restricted and not exposed publicly, the default authentication and
authorization settings, when no configuration file specified with the `--config`
flag, are overly permissive.

If a cloud provider is used for hosting Kubernetes, the access from pods to the cloud
metadata API `169.254.169.254` should also be restricted or blocked if not needed
because it may leak information.

For restricted LoadBalancer and ExternalIPs use, see
[CVE-2020-8554 Man in the middle using LoadBalancer or ExternalIPs](httpsgithub.comkuberneteskubernetesissues97076)
and the [DenyServiceExternalIPs admission controller](docsreferenceaccess-authn-authzadmission-controllers#denyserviceexternalips)
for further information.

# # Pod security

- [ ] RBAC rights to `create`, `update`, `patch`, `delete` workloads is only granted if necessary.
- [ ] Appropriate Pod Security Standards policy is applied for all namespaces and enforced.
- [ ] Memory limit is set for the workloads with a limit equal or inferior to the request.
- [ ] CPU limit might be set on sensitive workloads.
- [ ] For nodes that support it, Seccomp is enabled with appropriate syscalls
  profile for programs.
- [ ] For nodes that support it, AppArmor or SELinux is enabled with appropriate
  profile for programs.

RBAC authorization is crucial but
[cannot be granular enough to have authorization on the Pods resources](docsconceptssecurityrbac-good-practices#workload-creation)
(or on any resource that manages Pods). The only granularity is the API verbs
on the resource itself, for example, `create` on Pods. Without
additional admission, the authorization to create these resources allows direct
unrestricted access to the schedulable nodes of a cluster.

The [Pod Security Standards](docsconceptssecuritypod-security-standards)
define three different policies, privileged, baseline and restricted that limit
how fields can be set in the `PodSpec` regarding security.
These standards can be enforced at the namespace level with the new
[Pod Security](docsconceptssecuritypod-security-admission) admission,
enabled by default, or by third-party admission webhook. Please note that,
contrary to the removed PodSecurityPolicy admission it replaces,
[Pod Security](docsconceptssecuritypod-security-admission)
admission can be easily combined with admission webhooks and external services.

Pod Security admission `restricted` policy, the most restrictive policy of the
[Pod Security Standards](docsconceptssecuritypod-security-standards) set,
[can operate in several modes](docsconceptssecuritypod-security-admission#pod-security-admission-labels-for-namespaces),
`warn`, `audit` or `enforce` to gradually apply the most appropriate
[security context](docstasksconfigure-pod-containersecurity-context)
according to security best practices. Nevertheless, pods
[security context](docstasksconfigure-pod-containersecurity-context)
should be separately investigated to limit the privileges and access pods may
have on top of the predefined security standards, for specific use cases.

For a hands-on tutorial on [Pod Security](docsconceptssecuritypod-security-admission),
see the blog post
[Kubernetes 1.23 Pod Security Graduates to Beta](blog20211209pod-security-admission-beta).

[Memory and CPU limits](docsconceptsconfigurationmanage-resources-containers)
should be set in order to restrict the memory and CPU resources a pod can
consume on a node, and therefore prevent potential DoS attacks from malicious or
breached workloads. Such policy can be enforced by an admission controller.
Please note that CPU limits will throttle usage and thus can have unintended
effects on auto-scaling features or efficiency i.e. running the process in best
effort with the CPU resource available.

Memory limit superior to request can expose the whole node to OOM issues.

# # # Enabling Seccomp

Seccomp stands for secure computing mode and has been a feature of the Linux kernel since version 2.6.12.
It can be used to sandbox the privileges of a process, restricting the calls it is able to make
from userspace into the kernel. Kubernetes lets you automatically apply seccomp profiles loaded onto
a node to your Pods and containers.

Seccomp can improve the security of your workloads by reducing the Linux kernel syscall attack
surface available inside containers. The seccomp filter mode leverages BPF to create an allow or
deny list of specific syscalls, named profiles.

Since Kubernetes 1.27, you can enable the use of `RuntimeDefault` as the default seccomp profile
for all workloads. A [security tutorial](docstutorialssecurityseccomp) is available on this
topic. In addition, the
[Kubernetes Security Profiles Operator](httpsgithub.comkubernetes-sigssecurity-profiles-operator)
is a project that facilitates the management and use of seccomp in clusters.

Seccomp is only available on Linux nodes.

# # # Enabling AppArmor or SELinux

# # # # AppArmor

[AppArmor](docstutorialssecurityapparmor) is a Linux kernel security module that can
provide an easy way to implement Mandatory Access Control (MAC) and better
auditing through system logs. A default AppArmor profile is enforced on nodes that support it, or a custom profile can be configured.
Like seccomp, AppArmor is also configured
through profiles, where each profile is either running in enforcing mode, which
blocks access to disallowed resources or complain mode, which only reports
violations. AppArmor profiles are enforced on a per-container basis, with an
annotation, allowing for processes to gain just the right privileges.

AppArmor is only available on Linux nodes, and enabled in
[some Linux distributions](httpsgitlab.comapparmorapparmor-wikishome#distributions-and-ports).

# # # # SELinux

[SELinux](httpsgithub.comSELinuxProjectselinux-notebookblobmainsrcselinux_overview.md) is also a
Linux kernel security module that can provide a mechanism for supporting access
control security policies, including Mandatory Access Controls (MAC). SELinux
labels can be assigned to containers or pods
[via their `securityContext` section](docstasksconfigure-pod-containersecurity-context#assign-selinux-labels-to-a-container).

SELinux is only available on Linux nodes, and enabled in
[some Linux distributions](httpsen.wikipedia.orgwikiSecurity-Enhanced_Linux#Implementations).

# # Logs and auditing

- [ ] Audit logs, if enabled, are protected from general access.

# # Pod placement

- [ ] Pod placement is done in accordance with the tiers of sensitivity of the
  application.
- [ ] Sensitive applications are running isolated on nodes or with specific
  sandboxed runtimes.

Pods that are on different tiers of sensitivity, for example, an application pod
and the Kubernetes API server, should be deployed onto separate nodes. The
purpose of node isolation is to prevent an application container breakout to
directly providing access to applications with higher level of sensitivity to easily
pivot within the cluster. This separation should be enforced to prevent pods
accidentally being deployed onto the same node. This could be enforced with the
following features

[Node Selectors](docsconceptsscheduling-evictionassign-pod-node)
 Key-value pairs, as part of the pod specification, that specify which nodes to
deploy onto. These can be enforced at the namespace and cluster level with the
[PodNodeSelector](docsreferenceaccess-authn-authzadmission-controllers#podnodeselector)
admission controller.

[PodTolerationRestriction](docsreferenceaccess-authn-authzadmission-controllers#podtolerationrestriction)
 An admission controller that allows administrators to restrict permitted
[tolerations](docsconceptsscheduling-evictiontaint-and-toleration) within a
namespace. Pods within a namespace may only utilize the tolerations specified on
the namespace object annotation keys that provide a set of default and allowed
tolerations.

[RuntimeClass](docsconceptscontainersruntime-class)
 RuntimeClass is a feature for selecting the container runtime configuration.
The container runtime configuration is used to run a Pods containers and can
provide more or less isolation from the host at the cost of performance
overhead.

# # Secrets

- [ ] ConfigMaps are not used to hold confidential data.
- [ ] Encryption at rest is configured for the Secret API.
- [ ] If appropriate, a mechanism to inject secrets stored in third-party storage
  is deployed and available.
- [ ] Service account tokens are not mounted in pods that dont require them.
- [ ] [Bound service account token volume](docsreferenceaccess-authn-authzservice-accounts-admin#bound-service-account-token-volume)
  is in-use instead of non-expiring tokens.

Secrets required for pods should be stored within Kubernetes Secrets as opposed
to alternatives such as ConfigMap. Secret resources stored within etcd should
be [encrypted at rest](docstasksadminister-clusterencrypt-data).

Pods needing secrets should have these automatically mounted through volumes,
preferably stored in memory like with the [`emptyDir.medium` option](docsconceptsstoragevolumes#emptydir).
Mechanism can be used to also inject secrets from third-party storages as
volume, like the [Secrets Store CSI Driver](httpssecrets-store-csi-driver.sigs.k8s.io).
This should be done preferentially as compared to providing the pods service
account RBAC access to secrets. This would allow adding secrets into the pod as
environment variables or files. Please note that the environment variable method
might be more prone to leakage due to crash dumps in logs and the
non-confidential nature of environment variable in Linux, as opposed to the
permission mechanism on files.

Service account tokens should not be mounted into pods that do not require them. This can be configured by setting
[`automountServiceAccountToken`](docstasksconfigure-pod-containerconfigure-service-account#use-the-default-service-account-to-access-the-api-server)
to `false` either within the service account to apply throughout the namespace
or specifically for a pod. For Kubernetes v1.22 and above, use
[Bound Service Accounts](docsreferenceaccess-authn-authzservice-accounts-admin#bound-service-account-token-volume)
for time-bound service account credentials.

# # Images

- [ ] Minimize unnecessary content in container images.
- [ ] Container images are configured to be run as unprivileged user.
- [ ] References to container images are made by sha256 digests (rather than
tags) or the provenance of the image is validated by verifying the images
digital signature at deploy time [via admission control](docstasksadminister-clusterverify-signed-artifacts#verifying-image-signatures-with-admission-controller).
- [ ] Container images are regularly scanned during creation and in deployment, and
  known vulnerable software is patched.

Container image should contain the bare minimum to run the program they
package. Preferably, only the program and its dependencies, building the image
from the minimal possible base. In particular, image used in production should not
contain shells or debugging utilities, as an
[ephemeral debug container](docstasksdebugdebug-applicationdebug-running-pod#ephemeral-container)
can be used for troubleshooting.

Build images to directly start with an unprivileged user by using the
[`USER` instruction in Dockerfile](httpsdocs.docker.comdevelopdevelop-imagesdockerfile_best-practices#user).
The [Security Context](docstasksconfigure-pod-containersecurity-context#set-the-security-context-for-a-pod)
allows a container image to be started with a specific user and group with
`runAsUser` and `runAsGroup`, even if not specified in the image manifest.
However, the file permissions in the image layers might make it impossible to just
start the process with a new unprivileged user without image modification.

Avoid using image tags to reference an image, especially the `latest` tag, the
image behind a tag can be easily modified in a registry. Prefer using the
complete `sha256` digest which is unique to the image manifest. This policy can be
enforced via an [ImagePolicyWebhook](docsreferenceaccess-authn-authzadmission-controllers#imagepolicywebhook).
Image signatures can also be automatically [verified with an admission controller](docstasksadminister-clusterverify-signed-artifacts#verifying-image-signatures-with-admission-controller)
at deploy time to validate their authenticity and integrity.

Scanning a container image can prevent critical vulnerabilities from being
deployed to the cluster alongside the container image. Image scanning should be
completed before deploying a container image to a cluster and is usually done
as part of the deployment process in a CICD pipeline. The purpose of an image
scan is to obtain information about possible vulnerabilities and their
prevention in the container image, such as a
[Common Vulnerability Scoring System (CVSS)](httpswww.first.orgcvss)
score. If the result of the image scans is combined with the pipeline
compliance rules, only properly patched container images will end up in
Production.

# # Admission controllers

- [ ] An appropriate selection of admission controllers is enabled.
- [ ] A pod security policy is enforced by the Pod Security Admission orand a
  webhook admission controller.
- [ ] The admission chain plugins and webhooks are securely configured.

Admission controllers can help improve the security of the cluster. However,
they can present risks themselves as they extend the API server and
[should be properly secured](blog20220119secure-your-admission-controllers-and-webhooks).

The following lists present a number of admission controllers that could be
considered to enhance the security posture of your cluster and application. It
includes controllers that may be referenced in other parts of this document.

This first group of admission controllers includes plugins
[enabled by default](docsreferenceaccess-authn-authzadmission-controllers#which-plugins-are-enabled-by-default),
consider to leave them enabled unless you know what you are doing

[`CertificateApproval`](docsreferenceaccess-authn-authzadmission-controllers#certificateapproval)
 Performs additional authorization checks to ensure the approving user has
permission to approve certificate request.

[`CertificateSigning`](docsreferenceaccess-authn-authzadmission-controllers#certificatesigning)
 Performs additional authorization checks to ensure the signing user has
permission to sign certificate requests.

[`CertificateSubjectRestriction`](docsreferenceaccess-authn-authzadmission-controllers#certificatesubjectrestriction)
 Rejects any certificate request that specifies a group (or organization
attribute) of `systemmasters`.

[`LimitRanger`](docsreferenceaccess-authn-authzadmission-controllers#limitranger)
 Enforces the LimitRange API constraints.

[`MutatingAdmissionWebhook`](docsreferenceaccess-authn-authzadmission-controllers#mutatingadmissionwebhook)
 Allows the use of custom controllers through webhooks, these controllers may
mutate requests that they review.

[`PodSecurity`](docsreferenceaccess-authn-authzadmission-controllers#podsecurity)
 Replacement for Pod Security Policy, restricts security contexts of deployed
Pods.

[`ResourceQuota`](docsreferenceaccess-authn-authzadmission-controllers#resourcequota)
 Enforces resource quotas to prevent over-usage of resources.

[`ValidatingAdmissionWebhook`](docsreferenceaccess-authn-authzadmission-controllers#validatingadmissionwebhook)
 Allows the use of custom controllers through webhooks, these controllers do
not mutate requests that it reviews.

The second group includes plugins that are not enabled by default but are in general
availability state and are recommended to improve your security posture

[`DenyServiceExternalIPs`](docsreferenceaccess-authn-authzadmission-controllers#denyserviceexternalips)
 Rejects all net-new usage of the `Service.spec.externalIPs` field. This is a mitigation for
[CVE-2020-8554 Man in the middle using LoadBalancer or ExternalIPs](httpsgithub.comkuberneteskubernetesissues97076).

[`NodeRestriction`](docsreferenceaccess-authn-authzadmission-controllers#noderestriction)
 Restricts kubelets permissions to only modify the pods API resources they own
or the node API resource that represent themselves. It also prevents kubelet
from using the `node-restriction.kubernetes.io` annotation, which can be used
by an attacker with access to the kubelets credentials to influence pod
placement to the controlled node.

The third group includes plugins that are not enabled by default but could be
considered for certain use cases

[`AlwaysPullImages`](docsreferenceaccess-authn-authzadmission-controllers#alwayspullimages)
 Enforces the usage of the latest version of a tagged image and ensures that the deployer
has permissions to use the image.

[`ImagePolicyWebhook`](docsreferenceaccess-authn-authzadmission-controllers#imagepolicywebhook)
 Allows enforcing additional controls for images through webhooks.

# # Whats next

- [Privilege escalation via Pod creation](docsreferenceaccess-authn-authzauthorization#privilege-escalation-via-pod-creation)
  warns you about a specific access control risk check how youre managing that
  threat.
  - If you use Kubernetes RBAC, read
    [RBAC Good Practices](docsconceptssecurityrbac-good-practices) for
    further information on authorization.
- [Securing a Cluster](docstasksadminister-clustersecuring-a-cluster) for
  information on protecting a cluster from accidental or malicious access.
- [Cluster Multi-tenancy guide](docsconceptssecuritymulti-tenancy) for
  configuration options recommendations and best practices on multi-tenancy.
- [Blog post A Closer Look at NSACISA Kubernetes Hardening Guidance](blog20211005nsa-cisa-kubernetes-hardening-guidance#building-secure-container-images)
  for complementary resource on hardening Kubernetes clusters.
