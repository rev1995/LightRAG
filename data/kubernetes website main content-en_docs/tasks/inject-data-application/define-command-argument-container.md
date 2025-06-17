---
title Define a Command and Arguments for a Container
content_type task
weight 10
---

This page shows how to define commands and arguments when you run a container
in a .

# #  heading prerequisites

# # Define a command and arguments when you create a Pod

When you create a Pod, you can define a command and arguments for the
containers that run in the Pod. To define a command, include the `command`
field in the configuration file. To define arguments for the command, include
the `args` field in the configuration file. The command and arguments that
you define cannot be changed after the Pod is created.

The command and arguments that you define in the configuration file
override the default command and arguments provided by the container image.
If you define args, but do not define a command, the default command is used
with your new arguments.

The `command` field corresponds to `ENTRYPOINT`, and the `args` field corresponds to `CMD` in some container runtimes.

In this exercise, you create a Pod that runs one container. The configuration
file for the Pod defines a command and two arguments

 code_sample filepodscommands.yaml

1. Create a Pod based on the YAML configuration file

   ```shell
   kubectl apply -f httpsk8s.ioexamplespodscommands.yaml
   ```

1. List the running Pods

   ```shell
   kubectl get pods
   ```

   The output shows that the container that ran in the command-demo Pod has
   completed.

1. To see the output of the command that ran in the container, view the logs
from the Pod

   ```shell
   kubectl logs command-demo
   ```

   The output shows the values of the HOSTNAME and KUBERNETES_PORT environment
   variables

   ```
   command-demo
   tcp10.3.240.1443
   ```

# # Use environment variables to define arguments

In the preceding example, you defined the arguments directly by
providing strings. As an alternative to providing strings directly,
you can define arguments by using environment variables

```yaml
env
- name MESSAGE
  value hello world
command [binecho]
args [(MESSAGE)]
```

This means you can define an argument for a Pod using any of
the techniques available for defining environment variables, including
[ConfigMaps](docstasksconfigure-pod-containerconfigure-pod-configmap)
and
[Secrets](docsconceptsconfigurationsecret).

The environment variable appears in parentheses, `(VAR)`. This is
required for the variable to be expanded in the `command` or `args` field.

# # Run a command in a shell

In some cases, you need your command to run in a shell. For example, your
command might consist of several commands piped together, or it might be a shell
script. To run your command in a shell, wrap it like this

```shell
command [binsh]
args [-c, while true do echo hello sleep 10done]
```

# #  heading whatsnext

* Learn more about [configuring pods and containers](docstasks).
* Learn more about [running commands in a container](docstasksdebugdebug-applicationget-shell-running-container).
* See [Container](docsreferencegeneratedkubernetes-api#container-v1-core).
