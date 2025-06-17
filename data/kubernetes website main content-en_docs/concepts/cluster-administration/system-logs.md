---
reviewers
- dims
- 44past4
title System Logs
content_type concept
weight 80
---

System component logs record events happening in cluster, which can be very useful for debugging.
You can configure log verbosity to see more or less detail.
Logs can be as coarse-grained as showing errors within a component, or as fine-grained as showing
step-by-step traces of events (like HTTP access logs, pod state changes, controller actions, or
scheduler decisions).

In contrast to the command line flags described here, the *log
output* itself does *not* fall under the Kubernetes API stability guarantees
individual log entries and their formatting may change from one release
to the next!

# # Klog

klog is the Kubernetes logging library. [klog](httpsgithub.comkubernetesklog)
generates log messages for the Kubernetes system components.

Kubernetes is in the process of simplifying logging in its components.
The following klog command line flags
[are deprecated](httpsgithub.comkubernetesenhancementstreemasterkepssig-instrumentation2845-deprecate-klog-specific-flags-in-k8s-components)
starting with Kubernetes v1.23 and removed in Kubernetes v1.26

- `--add-dir-header`
- `--alsologtostderr`
- `--log-backtrace-at`
- `--log-dir`
- `--log-file`
- `--log-file-max-size`
- `--logtostderr`
- `--one-output`
- `--skip-headers`
- `--skip-log-headers`
- `--stderrthreshold`

Output will always be written to stderr, regardless of the output format. Output redirection is
expected to be handled by the component which invokes a Kubernetes component. This can be a POSIX
shell or a tool like systemd.

In some cases, for example a distroless container or a Windows system service, those options are
not available. Then the
[`kube-log-runner`](httpsgithub.comkuberneteskubernetesblobd2a8a81639fcff8d1221b900f66d28361a170654stagingsrck8s.iocomponent-baselogskube-log-runnerREADME.md)
binary can be used as wrapper around a Kubernetes component to redirect
output. A prebuilt binary is included in several Kubernetes base images under
its traditional name as `go-runner` and as `kube-log-runner` in server and
node release archives.

This table shows how `kube-log-runner` invocations correspond to shell redirection

 Usage                                     POSIX shell (such as bash)  `kube-log-runner  `
 ----------------------------------------------------------------------------------------------------------------------------------
 Merge stderr and stdout, write to stdout  `21`                      `kube-log-runner` (default behavior)
 Redirect both into log file               `1tmplog 21`          `kube-log-runner -log-filetmplog`
 Copy into log file and to stdout          `21  tee -a tmplog`   `kube-log-runner -log-filetmplog -also-stdout`
 Redirect only stdout into log file        `tmplog`                 `kube-log-runner -log-filetmplog -redirect-stderrfalse`

# # # Klog output

An example of the traditional klog native format

```
I1025 001515.525108       1 httplog.go79] GET apiv1namespaceskube-systempodsmetrics-server-v0.3.1-57c75779f-9p8wg (1.512ms) 200 [pod_nannyv0.0.0 (linuxamd64) kubernetesFormat 10.56.1.1951756]
```

The message string may contain line breaks

```
I1025 001515.525108       1 example.go79] This is a message
which has a line break.
```

# # # Structured Logging

Migration to structured log messages is an ongoing process. Not all log messages are structured in
this version. When parsing log files, you must also handle unstructured log messages.

Log formatting and value serialization are subject to change.

Structured logging introduces a uniform structure in log messages allowing for programmatic
extraction of information. You can store and process structured logs with less effort and cost.
The code which generates a log message determines whether it uses the traditional unstructured
klog output or structured logging.

The default formatting of structured log messages is as text, with a format that is backward
compatible with traditional klog

```
    ...
```

Example

```
I1025 001515.525108       1 controller_utils.go116] Pod status updated podkube-systemkubedns statusready
```

Strings are quoted. Other values are formatted with
[`v`](httpspkg.go.devfmt#hdr-Printing), which may cause log messages to
continue on the next line [depending on the data](httpsgithub.comkuberneteskubernetesissues106428).

```
I1025 001515.525108       1 example.go116] Example dataThis is text with a line breaknand quotation marks. someInt1 someFloat0.1 someStructStringField First line,
second line.
```

# # # Contextual Logging

Contextual logging builds on top of structured logging. It is primarily about
how developers use logging calls code based on that concept is more flexible
and supports additional use cases as described in the [Contextual Logging
KEP](httpsgithub.comkubernetesenhancementstreemasterkepssig-instrumentation3077-contextual-logging).

If developers use additional functions like `WithValues` or `WithName` in
their components, then log entries contain additional information that gets
passed into functions by their caller.

For Kubernetes , this is gated behind the `ContextualLogging`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates) and is
enabled by default. The infrastructure for this was added in 1.24 without
modifying components. The
[`component-baselogsexample`](httpsgithub.comkuberneteskubernetesblobv1.24.0-beta.0stagingsrck8s.iocomponent-baselogsexamplecmdlogger.go)
command demonstrates how to use the new logging calls and how a component
behaves that supports contextual logging.

```console
 cd GOPATHsrck8s.iokubernetesstagingsrck8s.iocomponent-baselogsexamplecmd
 go run . --help
...
      --feature-gates mapStringBool  A set of keyvalue pairs that describe feature gates for alphaexperimental features. Options are
                                     AllAlphatruefalse (ALPHA - defaultfalse)
                                     AllBetatruefalse (BETA - defaultfalse)
                                     ContextualLoggingtruefalse (BETA - defaulttrue)
 go run . --feature-gates ContextualLoggingtrue
...
I0222 151331.645988  197901 example.go54] runtime loggerexample.myname foobar duration1m0s
I0222 151331.646007  197901 example.go55] another runtime loggerexample foobar duration1h0m0s duration1m0s
```

The `logger` key and `foobar` were added by the caller of the function
which logs the `runtime` message and `duration1m0s` value, without having to
modify that function.

With contextual logging disable, `WithValues` and `WithName` do nothing and log
calls go through the global klog logger. Therefore this additional information
is not in the log output anymore

```console
 go run . --feature-gates ContextualLoggingfalse
...
I0222 151440.497333  198174 example.go54] runtime duration1m0s
I0222 151440.497346  198174 example.go55] another runtime duration1h0m0s duration1m0s
```

# # # JSON log format

JSON output does not support many standard klog flags. For list of unsupported klog flags, see the
[Command line tool reference](docsreferencecommand-line-tools-reference).

Not all logs are guaranteed to be written in JSON format (for example, during process start).
If you intend to parse logs, make sure you can handle log lines that are not JSON as well.

Field names and JSON serialization are subject to change.

The `--logging-formatjson` flag changes the format of logs from klog native format to JSON format.
Example of JSON log format (pretty printed)

```json

   ts 1580306777.04728,
   v 4,
   msg Pod status updated,
   pod
      name nginx-1,
      namespace default
   ,
   status ready

```

Keys with special meaning

* `ts` - timestamp as Unix time (required, float)
* `v` - verbosity (only for info and not for error messages, int)
* `err` - error string (optional, string)
* `msg` - message (required, string)

List of components currently supporting JSON format

*
*
*
*

# # # Log verbosity level

The `-v` flag controls log verbosity. Increasing the value increases the number of logged events.
Decreasing the value decreases the number of logged events.  Increasing verbosity settings logs
increasingly less severe events. A verbosity setting of 0 logs only critical events.

# # # Log location

There are two types of system components those that run in a container and those
that do not run in a container. For example

* The Kubernetes scheduler and kube-proxy run in a container.
* The kubelet and
  do not run in containers.

On machines with systemd, the kubelet and container runtime write to journald.
Otherwise, they write to `.log` files in the `varlog` directory.
System components inside containers always write to `.log` files in the `varlog` directory,
bypassing the default logging mechanism.
Similar to the container logs, you should rotate system component logs in the `varlog` directory.
In Kubernetes clusters created by the `kube-up.sh` script, log rotation is configured by the `logrotate` tool.
The `logrotate` tool rotates logs daily, or once the log size is greater than 100MB.

# # Log query

To help with debugging issues on nodes, Kubernetes v1.27 introduced a feature that allows viewing logs of services
running on the node. To use the feature, ensure that the `NodeLogQuery`
[feature gate](docsreferencecommand-line-tools-referencefeature-gates) is enabled for that node, and that the
kubelet configuration options `enableSystemLogHandler` and `enableSystemLogQuery` are both set to true. On Linux
the assumption is that service logs are available via journald. On Windows the assumption is that service logs are
available in the application log provider. On both operating systems, logs are also available by reading files within
`varlog`.

Provided you are authorized to interact with node objects, you can try out this feature on all your nodes or
just a subset. Here is an example to retrieve the kubelet service logs from a node

```shell
# Fetch kubelet logs from a node named node-1.example
kubectl get --raw apiv1nodesnode-1.exampleproxylogsquerykubelet
```

You can also fetch files, provided that the files are in a directory that the kubelet allows for log
fetches. For example, you can fetch a log from `varlog` on a Linux node

```shell
kubectl get --raw apiv1nodesproxylogsquery
```

The kubelet uses heuristics to retrieve logs. This helps if you are not aware whether a given system service is
writing logs to the operating systems native logger like journald or to a log file in `varlog`. The heuristics
first checks the native logger and if that is not available attempts to retrieve the first logs from
`varlog` or `varlog.log` or `varlog.log`.

The complete list of options that can be used are

Option  Description
------  -----------
`boot`  boot show messages from a specific system boot
`pattern`  pattern filters log entries by the provided PERL-compatible regular expression
`query`  query specifies services(s) or files from which to return logs (required)
`sinceTime`  an [RFC3339](httpswww.rfc-editor.orgrfcrfc3339) timestamp from which to show logs (inclusive)
`untilTime`  an [RFC3339](httpswww.rfc-editor.orgrfcrfc3339) timestamp until which to show logs (inclusive)
`tailLines`  specify how many lines from the end of the log to retrieve the default is to fetch the whole log

Example of a more complex query

```shell
# Fetch kubelet logs from a node named node-1.example that have the word error
kubectl get --raw apiv1nodesnode-1.exampleproxylogsquerykubeletpatternerror
```

# #  heading whatsnext

* Read about the [Kubernetes Logging Architecture](docsconceptscluster-administrationlogging)
* Read about [Structured Logging](httpsgithub.comkubernetesenhancementstreemasterkepssig-instrumentation1602-structured-logging)
* Read about [Contextual Logging](httpsgithub.comkubernetesenhancementstreemasterkepssig-instrumentation3077-contextual-logging)
* Read about [deprecation of klog flags](httpsgithub.comkubernetesenhancementstreemasterkepssig-instrumentation2845-deprecate-klog-specific-flags-in-k8s-components)
* Read about the [Conventions for logging severity](httpsgithub.comkubernetescommunityblobmastercontributorsdevelsig-instrumentationlogging.md)
* Read about [Log Query](httpskep.k8s.io2258)
