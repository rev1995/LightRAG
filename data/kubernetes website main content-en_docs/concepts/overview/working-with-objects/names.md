---
reviewers
- mikedanese
- thockin
title Object Names and IDs
content_type concept
weight 30
---

Each  in your cluster has a [_Name_](#names) that is unique for that type of resource.
Every Kubernetes object also has a [_UID_](#uids) that is unique across your whole cluster.

For example, you can only have one Pod named `myapp-1234` within the same [namespace](docsconceptsoverviewworking-with-objectsnamespaces), but you can have one Pod and one Deployment that are each named `myapp-1234`.

For non-unique user-provided attributes, Kubernetes provides [labels](docsconceptsoverviewworking-with-objectslabels) and [annotations](docsconceptsoverviewworking-with-objectsannotations).

# # Names

**Names must be unique across all [API versions](docsconceptsoverviewkubernetes-api#api-groups-and-versioning)
of the same resource. API resources are distinguished by their API group, resource type, namespace
(for namespaced resources), and name. In other words, API version is irrelevant in this context.**

In cases when objects represent a physical entity, like a Node representing a physical host, when the host is re-created under the same name without deleting and re-creating the Node, Kubernetes treats the new host as the old one, which may lead to inconsistencies.

The server may generate a name when `generateName` is provided instead of `name` in a resource create request.
When `generateName` is used, the provided value is used as a name prefix, which server appends a generated suffix
to. Even though the name is generated, it may conflict with existing names resulting in a HTTP 409 response. This
became far less likely to happen in Kubernetes v1.31 and later, since the server will make up to 8 attempt to generate a
unique name before returning a HTTP 409 response.

Below are four types of commonly used name constraints for resources.

# # # DNS Subdomain Names

Most resource types require a name that can be used as a DNS subdomain name
as defined in [RFC 1123](httpstools.ietf.orghtmlrfc1123).
This means the name must

- contain no more than 253 characters
- contain only lowercase alphanumeric characters, - or .
- start with an alphanumeric character
- end with an alphanumeric character

# # # RFC 1123 Label Names #dns-label-names

Some resource types require their names to follow the DNS
label standard as defined in [RFC 1123](httpstools.ietf.orghtmlrfc1123).
This means the name must

- contain at most 63 characters
- contain only lowercase alphanumeric characters or -
- start with an alphanumeric character
- end with an alphanumeric character

# # # RFC 1035 Label Names

Some resource types require their names to follow the DNS
label standard as defined in [RFC 1035](httpstools.ietf.orghtmlrfc1035).
This means the name must

- contain at most 63 characters
- contain only lowercase alphanumeric characters or -
- start with an alphabetic character
- end with an alphanumeric character

The only difference between the RFC 1035 and RFC 1123
label standards is that RFC 1123 labels are allowed to
start with a digit, whereas RFC 1035 labels can start
with a lowercase alphabetic character only.

# # # Path Segment Names

Some resource types require their names to be able to be safely encoded as a
path segment. In other words, the name may not be . or .. and the name may
not contain  or .

Heres an example manifest for a Pod named `nginx-demo`.

```yaml
apiVersion v1
kind Pod
metadata
  name nginx-demo
spec
  containers
  - name nginx
    image nginx1.14.2
    ports
    - containerPort 80
```

Some resource types have additional restrictions on their names.

# # UIDs

Kubernetes UIDs are universally unique identifiers (also known as UUIDs).
UUIDs are standardized as ISOIEC 9834-8 and as ITU-T X.667.

# #  heading whatsnext

* Read about [labels](docsconceptsoverviewworking-with-objectslabels) and [annotations](docsconceptsoverviewworking-with-objectsannotations) in Kubernetes.
* See the [Identifiers and Names in Kubernetes](httpsgit.k8s.iodesign-proposals-archivearchitectureidentifiers.md) design document.
