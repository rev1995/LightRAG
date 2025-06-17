---
title Monitor Node Health
content_type task
reviewers
- Random-Liu
- dchen1107
weight 20
---

*Node Problem Detector* is a daemon for monitoring and reporting about a nodes health.
You can run Node Problem Detector as a `DaemonSet` or as a standalone daemon.
Node Problem Detector collects information about node problems from various daemons
and reports these conditions to the API server as Node [Condition](docsconceptsarchitecturenodes#condition)s
or as [Event](docsreferencekubernetes-apicluster-resourcesevent-v1)s.

To learn how to install and use Node Problem Detector, see
[Node Problem Detector project documentation](httpsgithub.comkubernetesnode-problem-detector).

# #  heading prerequisites

# # Limitations

* Node Problem Detector uses the kernel log format for reporting kernel issues.
  To learn how to extend the kernel log format, see [Add support for another log format](#support-other-log-format).

# # Enabling Node Problem Detector

Some cloud providers enable Node Problem Detector as an .
You can also enable Node Problem Detector with `kubectl` or by creating an Addon DaemonSet.

# # # Using kubectl to enable Node Problem Detector #using-kubectl

`kubectl` provides the most flexible management of Node Problem Detector.
You can overwrite the default configuration to fit it into your environment or
to detect customized node problems. For example

1. Create a Node Problem Detector configuration similar to `node-problem-detector.yaml`

    code_sample filedebugnode-problem-detector.yaml

   You should verify that the system log directory is right for your operating system distribution.

1. Start node problem detector with `kubectl`

   ```shell
   kubectl apply -f httpsk8s.ioexamplesdebugnode-problem-detector.yaml
   ```

# # # Using an Addon pod to enable Node Problem Detector #using-addon-pod

If you are using a custom cluster bootstrap solution and dont need
to overwrite the default configuration, you can leverage the Addon pod to
further automate the deployment.

Create `node-problem-detector.yaml`, and save the configuration in the Addon pods
directory `etckubernetesaddonsnode-problem-detector` on a control plane node.

# # Overwrite the configuration

The [default configuration](httpsgithub.comkubernetesnode-problem-detectortreev0.8.12config)
is embedded when building the Docker image of Node Problem Detector.

However, you can use a [`ConfigMap`](docstasksconfigure-pod-containerconfigure-pod-configmap)
to overwrite the configuration

1. Change the configuration files in `config`
1. Create the `ConfigMap` `node-problem-detector-config`

   ```shell
   kubectl create configmap node-problem-detector-config --from-fileconfig
   ```

1. Change the `node-problem-detector.yaml` to use the `ConfigMap`

    code_sample filedebugnode-problem-detector-configmap.yaml

1. Recreate the Node Problem Detector with the new configuration file

   ```shell
   # If you have a node-problem-detector running, delete before recreating
   kubectl delete -f httpsk8s.ioexamplesdebugnode-problem-detector.yaml
   kubectl apply -f httpsk8s.ioexamplesdebugnode-problem-detector-configmap.yaml
   ```

This approach only applies to a Node Problem Detector started with `kubectl`.

Overwriting a configuration is not supported if a Node Problem Detector runs as a cluster Addon.
The Addon manager does not support `ConfigMap`.

# # Problem Daemons

A problem daemon is a sub-daemon of the Node Problem Detector. It monitors specific kinds of node
problems and reports them to the Node Problem Detector.
There are several types of supported problem daemons.

- A `SystemLogMonitor` type of daemon monitors the system logs and reports problems and metrics
  according to predefined rules. You can customize the configurations for different log sources
  such as [filelog](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configkernel-monitor-filelog.json),
  [kmsg](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configkernel-monitor.json),
  [kernel](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configkernel-monitor-counter.json),
  [abrt](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configabrt-adaptor.json),
  and [systemd](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configsystemd-monitor-counter.json).

- A `SystemStatsMonitor` type of daemon collects various health-related system stats as metrics.
  You can customize its behavior by updating its
  [configuration file](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configsystem-stats-monitor.json).

- A `CustomPluginMonitor` type of daemon invokes and checks various node problems by running
  user-defined scripts. You can use different custom plugin monitors to monitor different
  problems and customize the daemon behavior by updating the
  [configuration file](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configcustom-plugin-monitor.json).

- A `HealthChecker` type of daemon checks the health of the kubelet and container runtime on a node.

# # # Adding support for other log format #support-other-log-format

The system log monitor currently supports file-based logs, journald, and kmsg.
Additional sources can be added by implementing a new
[log watcher](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12pkgsystemlogmonitorlogwatcherstypeslog_watcher.go).

# # # Adding custom plugin monitors

You can extend the Node Problem Detector to execute any monitor scripts written in any language by
developing a custom plugin. The monitor scripts must conform to the plugin protocol in exit code
and standard output. For more information, please refer to the
[plugin interface proposal](httpsdocs.google.comdocumentd1jK_5YloSYtboj-DtfjmYKxfNnUxCAvohLnsH5aGCAYQedit#).

# # Exporter

An exporter reports the node problems andor metrics to certain backends.
The following exporters are supported

- **Kubernetes exporter** this exporter reports node problems to the Kubernetes API server.
  Temporary problems are reported as Events and permanent problems are reported as Node Conditions.

- **Prometheus exporter** this exporter reports node problems and metrics locally as Prometheus
  (or OpenMetrics) metrics. You can specify the IP address and port for the exporter using command
  line arguments.

- **Stackdriver exporter** this exporter reports node problems and metrics to the Stackdriver
  Monitoring API. The exporting behavior can be customized using a
  [configuration file](httpsgithub.comkubernetesnode-problem-detectorblobv0.8.12configexporterstackdriver-exporter.json).

# # Recommendations and restrictions

It is recommended to run the Node Problem Detector in your cluster to monitor node health.
When running the Node Problem Detector, you can expect extra resource overhead on each node.
Usually this is fine, because

* The kernel log grows relatively slowly.
* A resource limit is set for the Node Problem Detector.
* Even under high load, the resource usage is acceptable. For more information, see the Node Problem Detector
  [benchmark result](httpsgithub.comkubernetesnode-problem-detectorissues2#issuecomment-220255629).
