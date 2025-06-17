---
title Field Selectors
content_type concept
weight 70
---

_Field selectors_ let you select Kubernetes  based on the
value of one or more resource fields. Here are some examples of field selector queries

* `metadata.namemy-service`
* `metadata.namespace!default`
* `status.phasePending`

This `kubectl` command selects all Pods for which the value of the [`status.phase`](docsconceptsworkloadspodspod-lifecycle#pod-phase) field is `Running`

```shell
kubectl get pods --field-selector status.phaseRunning
```

Field selectors are essentially resource *filters*. By default, no selectorsfilters are applied, meaning that all resources of the specified type are selected. This makes the `kubectl` queries `kubectl get pods` and `kubectl get pods --field-selector ` equivalent.

# # Supported fields

Supported field selectors vary by Kubernetes resource type. All resource types support the `metadata.name` and `metadata.namespace` fields. Using unsupported field selectors produces an error. For example

```shell
kubectl get ingress --field-selector foo.barbaz
```
```
Error from server (BadRequest) Unable to find ingresses that match label selector , field selector foo.barbaz foo.bar is not a known field selector only metadata.name, metadata.namespace
```

# # # List of supported fields

 Kind                       Fields
 -------------------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Pod                        `spec.nodeName``spec.restartPolicy``spec.schedulerName``spec.serviceAccountName``spec.hostNetwork``status.phase``status.podIP``status.nominatedNodeName`
 Event                      `involvedObject.kind``involvedObject.namespace``involvedObject.name``involvedObject.uid``involvedObject.apiVersion``involvedObject.resourceVersion``involvedObject.fieldPath``reason``reportingComponent``source``type`
 Secret                     `type`
 Namespace                  `status.phase`
 ReplicaSet                 `status.replicas`
 ReplicationController      `status.replicas`
 Job                        `status.successful`
 Node                       `spec.unschedulable`
 CertificateSigningRequest  `spec.signerName`

# # # Custom resources fields

All custom resource types support the `metadata.name` and `metadata.namespace` fields.

Additionally, the `spec.versions[*].selectableFields` field of a
declares which other fields in a custom resource may be used in field selectors. See [selectable fields for custom resources](docstasksextend-kubernetescustom-resourcescustom-resource-definitions#crd-selectable-fields)
for more information about how to use field selectors with CustomResourceDefinitions.

# # Supported operators

You can use the ``, ``, and `!` operators with field selectors (`` and `` mean the same thing). This `kubectl` command, for example, selects all Kubernetes Services that arent in the `default` namespace

```shell
kubectl get services  --all-namespaces --field-selector metadata.namespace!default
```

[Set-based operators](docsconceptsoverviewworking-with-objectslabels#set-based-requirement)
(`in`, `notin`, `exists`) are not supported for field selectors.

# # Chained selectors

As with [label](docsconceptsoverviewworking-with-objectslabels) and other selectors, field selectors can be chained together as a comma-separated list. This `kubectl` command selects all Pods for which the `status.phase` does not equal `Running` and the `spec.restartPolicy` field equals `Always`

```shell
kubectl get pods --field-selectorstatus.phase!Running,spec.restartPolicyAlways
```

# # Multiple resource types

You can use field selectors across multiple resource types. This `kubectl` command selects all Statefulsets and Services that are not in the `default` namespace

```shell
kubectl get statefulsets,services --all-namespaces --field-selector metadata.namespace!default
```
