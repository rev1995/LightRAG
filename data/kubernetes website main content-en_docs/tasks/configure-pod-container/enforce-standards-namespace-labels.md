---
title Enforce Pod Security Standards with Namespace Labels
reviewers
- tallclair
- liggitt
content_type task
weight 250
---

Namespaces can be labeled to enforce the [Pod Security Standards](docsconceptssecuritypod-security-standards). The three policies
[privileged](docsconceptssecuritypod-security-standards#privileged), [baseline](docsconceptssecuritypod-security-standards#baseline)
and [restricted](docsconceptssecuritypod-security-standards#restricted) broadly cover the security spectrum
and are implemented by the [Pod Security](docsconceptssecuritypod-security-admission) .

# #  heading prerequisites

Pod Security Admission was available by default in Kubernetes v1.23, as
a beta. From version 1.25 onwards, Pod Security Admission is generally
available.

 version-check

# # Requiring the `baseline` Pod Security Standard with namespace labels

This manifest defines a Namespace `my-baseline-namespace` that

- _Blocks_ any pods that dont satisfy the `baseline` policy requirements.
- Generates a user-facing warning and adds an audit annotation to any created pod that does not
  meet the `restricted` policy requirements.
- Pins the versions of the `baseline` and `restricted` policies to v.

```yaml
apiVersion v1
kind Namespace
metadata
  name my-baseline-namespace
  labels
    pod-security.kubernetes.ioenforce baseline
    pod-security.kubernetes.ioenforce-version v

    # We are setting these to our _desired_ `enforce` level.
    pod-security.kubernetes.ioaudit restricted
    pod-security.kubernetes.ioaudit-version v
    pod-security.kubernetes.iowarn restricted
    pod-security.kubernetes.iowarn-version v
```

# # Add labels to existing namespaces with `kubectl label`

When an `enforce` policy (or version) label is added or changed, the admission plugin will test
each pod in the namespace against the new policy. Violations are returned to the user as warnings.

It is helpful to apply the `--dry-run` flag when initially evaluating security profile changes for
namespaces. The Pod Security Standard checks will still be run in _dry run_ mode, giving you
information about how the new policy would treat existing pods, without actually updating a policy.

```shell
kubectl label --dry-runserver --overwrite ns --all
    pod-security.kubernetes.ioenforcebaseline
```

# # # Applying to all namespaces

If youre just getting started with the Pod Security Standards, a suitable first step would be to
configure all namespaces with audit annotations for a stricter level such as `baseline`

```shell
kubectl label --overwrite ns --all
  pod-security.kubernetes.ioauditbaseline
  pod-security.kubernetes.iowarnbaseline
```

Note that this is not setting an enforce level, so that namespaces that havent been explicitly
evaluated can be distinguished. You can list namespaces without an explicitly set enforce level
using this command

```shell
kubectl get namespaces --selector!pod-security.kubernetes.ioenforce
```

# # # Applying to a single namespace

You can update a specific namespace as well. This command adds the `enforcerestricted`
policy to `my-existing-namespace`, pinning the restricted policy version to v.

```shell
kubectl label --overwrite ns my-existing-namespace
  pod-security.kubernetes.ioenforcerestricted
  pod-security.kubernetes.ioenforce-versionv
```
