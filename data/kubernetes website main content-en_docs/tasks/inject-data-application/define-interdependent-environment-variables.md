---
title Define Dependent Environment Variables
content_type task
weight 20
---

This page shows how to define dependent environment variables for a container
in a Kubernetes Pod.

# #  heading prerequisites

# # Define an environment dependent variable for a container

When you create a Pod, you can set dependent environment variables for the containers that run in the Pod. To set dependent environment variables, you can use (VAR_NAME) in the `value` of `env` in the configuration file.

In this exercise, you create a Pod that runs one container. The configuration
file for the Pod defines a dependent environment variable with common usage defined. Here is the configuration manifest for the
Pod

 code_sample filepodsinjectdependent-envars.yaml

1. Create a Pod based on that manifest

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodsinjectdependent-envars.yaml
   ```
   ```
   poddependent-envars-demo created
   ```

2. List the running Pods

   ```shell
   kubectl get pods dependent-envars-demo
   ```
   ```
   NAME                      READY     STATUS    RESTARTS   AGE
   dependent-envars-demo     11       Running   0          9s
   ```

3. Check the logs for the container running in your Pod

   ```shell
   kubectl logs poddependent-envars-demo
   ```
   ```

   UNCHANGED_REFERENCE(PROTOCOL)172.17.0.180
   SERVICE_ADDRESShttps172.17.0.180
   ESCAPED_REFERENCE(PROTOCOL)172.17.0.180
   ```

As shown above, you have defined the correct dependency reference of `SERVICE_ADDRESS`, bad dependency reference of `UNCHANGED_REFERENCE` and skip dependent references of `ESCAPED_REFERENCE`.

When an environment variable is already defined when being referenced,
the reference can be correctly resolved, such as in the `SERVICE_ADDRESS` case.

Note that order matters in the `env` list. An environment variable is not considered
defined if it is specified further down the list. That is why `UNCHANGED_REFERENCE`
fails to resolve `(PROTOCOL)` in the example above.

When the environment variable is undefined or only includes some variables, the undefined environment variable is treated as a normal string, such as `UNCHANGED_REFERENCE`. Note that incorrectly parsed environment variables, in general, will not block the container from starting.

The `(VAR_NAME)` syntax can be escaped with a double ``, ie `(VAR_NAME)`.
Escaped references are never expanded, regardless of whether the referenced variable
is defined or not. This can be seen from the `ESCAPED_REFERENCE` case above.

# #  heading whatsnext

* Learn more about [environment variables](docstasksinject-data-applicationenvironment-variable-expose-pod-information).
* See [EnvVarSource](docsreferencegeneratedkubernetes-api#envvarsource-v1-core).
