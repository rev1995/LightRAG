---
title Recommended Labels
content_type concept
weight 100
---

You can visualize and manage Kubernetes objects with more tools than kubectl and
the dashboard. A common set of labels allows tools to work interoperably, describing
objects in a common manner that all tools can understand.

In addition to supporting tooling, the recommended labels describe applications
in a way that can be queried.

The metadata is organized around the concept of an _application_. Kubernetes is not
a platform as a service (PaaS) and doesnt have or enforce a formal notion of an application.
Instead, applications are informal and described with metadata. The definition of
what an application contains is loose.

These are recommended labels. They make it easier to manage applications
but arent required for any core tooling.

Shared labels and annotations share a common prefix `app.kubernetes.io`. Labels
without a prefix are private to users. The shared prefix ensures that shared labels
do not interfere with custom user labels.

# # Labels

In order to take full advantage of using these labels, they should be applied
on every resource object.

 Key                                  Description            Example   Type
 -----------------------------------  ---------------------  --------  ----
 `app.kubernetes.ioname`             The name of the application  `mysql`  string
 `app.kubernetes.ioinstance`         A unique name identifying the instance of an application  `mysql-abcxyz`  string
 `app.kubernetes.ioversion`          The current version of the application (e.g., a [SemVer 1.0](httpssemver.orgspecv1.0.0.html), revision hash, etc.)  `5.7.21`  string
 `app.kubernetes.iocomponent`        The component within the architecture  `database`  string
 `app.kubernetes.iopart-of`          The name of a higher level application this one is part of  `wordpress`  string
 `app.kubernetes.iomanaged-by`       The tool being used to manage the operation of an application  `Helm`  string

To illustrate these labels in action, consider the following  object

```yaml
# This is an excerpt
apiVersion appsv1
kind StatefulSet
metadata
  labels
    app.kubernetes.ioname mysql
    app.kubernetes.ioinstance mysql-abcxyz
    app.kubernetes.ioversion 5.7.21
    app.kubernetes.iocomponent database
    app.kubernetes.iopart-of wordpress
    app.kubernetes.iomanaged-by Helm
```

# # Applications And Instances Of Applications

An application can be installed one or more times into a Kubernetes cluster and,
in some cases, the same namespace. For example, WordPress can be installed more
than once where different websites are different installations of WordPress.

The name of an application and the instance name are recorded separately. For
example, WordPress has a `app.kubernetes.ioname` of `wordpress` while it has
an instance name, represented as `app.kubernetes.ioinstance` with a value of
`wordpress-abcxyz`. This enables the application and instance of the application
to be identifiable. Every instance of an application must have a unique name.

# # Examples

To illustrate different ways to use these labels the following examples have varying complexity.

# # # A Simple Stateless Service

Consider the case for a simple stateless service deployed using `Deployment` and `Service` objects. The following two snippets represent how the labels could be used in their simplest form.

The `Deployment` is used to oversee the pods running the application itself.
```yaml
apiVersion appsv1
kind Deployment
metadata
  labels
    app.kubernetes.ioname myservice
    app.kubernetes.ioinstance myservice-abcxyz
...
```

The `Service` is used to expose the application.
```yaml
apiVersion v1
kind Service
metadata
  labels
    app.kubernetes.ioname myservice
    app.kubernetes.ioinstance myservice-abcxyz
...
```

# # # Web Application With A Database

Consider a slightly more complicated application a web application (WordPress)
using a database (MySQL), installed using Helm. The following snippets illustrate
the start of objects used to deploy this application.

The start to the following `Deployment` is used for WordPress

```yaml
apiVersion appsv1
kind Deployment
metadata
  labels
    app.kubernetes.ioname wordpress
    app.kubernetes.ioinstance wordpress-abcxyz
    app.kubernetes.ioversion 4.9.4
    app.kubernetes.iomanaged-by Helm
    app.kubernetes.iocomponent server
    app.kubernetes.iopart-of wordpress
...
```

The `Service` is used to expose WordPress

```yaml
apiVersion v1
kind Service
metadata
  labels
    app.kubernetes.ioname wordpress
    app.kubernetes.ioinstance wordpress-abcxyz
    app.kubernetes.ioversion 4.9.4
    app.kubernetes.iomanaged-by Helm
    app.kubernetes.iocomponent server
    app.kubernetes.iopart-of wordpress
...
```

MySQL is exposed as a `StatefulSet` with metadata for both it and the larger application it belongs to

```yaml
apiVersion appsv1
kind StatefulSet
metadata
  labels
    app.kubernetes.ioname mysql
    app.kubernetes.ioinstance mysql-abcxyz
    app.kubernetes.ioversion 5.7.21
    app.kubernetes.iomanaged-by Helm
    app.kubernetes.iocomponent database
    app.kubernetes.iopart-of wordpress
...
```

The `Service` is used to expose MySQL as part of WordPress

```yaml
apiVersion v1
kind Service
metadata
  labels
    app.kubernetes.ioname mysql
    app.kubernetes.ioinstance mysql-abcxyz
    app.kubernetes.ioversion 5.7.21
    app.kubernetes.iomanaged-by Helm
    app.kubernetes.iocomponent database
    app.kubernetes.iopart-of wordpress
...
```

With the MySQL `StatefulSet` and `Service` youll notice information about both MySQL and WordPress, the broader application, are included.
