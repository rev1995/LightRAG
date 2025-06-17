---
title Pull an Image from a Private Registry
weight 130
---

This page shows how to create a Pod that uses a
 to pull an image
from a private container image registry or repository. There are many private
registries in use. This task uses [Docker Hub](httpswww.docker.comproductsdocker-hub)
as an example registry.

 thirdparty-content singletrue

# #  heading prerequisites

*

* To do this exercise, you need the `docker` command line tool, and a
  [Docker ID](httpsdocs.docker.comdocker-id) for which you know the password.
* If you are using a different private container registry, you need the command
  line tool for that registry and any login information for the registry.

# # Log in to Docker Hub

On your laptop, you must authenticate with a registry in order to pull a private image.

Use the `docker` tool to log in to Docker Hub. See the _log in_ section of
[Docker ID accounts](httpsdocs.docker.comdocker-id#log-in) for more information.

```shell
docker login
```

When prompted, enter your Docker ID, and then the credential you want to use (access token,
or the password for your Docker ID).

The login process creates or updates a `config.json` file that holds an authorization token.
Review [how Kubernetes interprets this file](docsconceptscontainersimages#config-json).

View the `config.json` file

```shell
cat .dockerconfig.json
```

The output contains a section similar to this

```json

    auths
        httpsindex.docker.iov1
            auth c3R...zE2

```

If you use a Docker credentials store, you wont see that `auth` entry but a `credsStore` entry with the name of the store as value.
In that case, you can create a secret directly.
See [Create a Secret by providing credentials on the command line](#create-a-secret-by-providing-credentials-on-the-command-line).

# # Create a Secret based on existing credentials #registry-secret-existing-credentials

A Kubernetes cluster uses the Secret of `kubernetes.iodockerconfigjson` type to authenticate with
a container registry to pull a private image.

If you already ran `docker login`, you can copy
that credential into Kubernetes

```shell
kubectl create secret generic regcred
    --from-file.dockerconfigjson
    --typekubernetes.iodockerconfigjson
```

If you need more control (for example, to set a namespace or a label on the new
secret) then you can customise the Secret before storing it.
Be sure to

- set the name of the data item to `.dockerconfigjson`
- base64 encode the Docker configuration file and then paste that string, unbroken
  as the value for field `data[.dockerconfigjson]`
- set `type` to `kubernetes.iodockerconfigjson`

Example

```yaml
apiVersion v1
kind Secret
metadata
  name myregistrykey
  namespace awesomeapps
data
  .dockerconfigjson UmVhbGx5IHJlYWxseSByZWVlZWVlZWVlZWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWxsbGxsbGxsbGxsbGxsbGxsbGxsbGxsbGxsbGxsbGx5eXl5eXl5eXl5eXl5eXl5eXl5eSBsbGxsbGxsbGxsbGxsbG9vb29vb29vb29vb29vb29vb29vb29vb29vb25ubm5ubm5ubm5ubm5ubm5ubm5ubm5ubmdnZ2dnZ2dnZ2dnZ2dnZ2dnZ2cgYXV0aCBrZXlzCg
type kubernetes.iodockerconfigjson
```

If you get the error message `error no objects passed to create`, it may mean the base64 encoded string is invalid.
If you get an error message like `Secret myregistrykey is invalid data[.dockerconfigjson] invalid value ...`, it means
the base64 encoded string in the data was successfully decoded, but could not be parsed as a `.dockerconfig.json` file.

# # Create a Secret by providing credentials on the command line

Create this Secret, naming it `regcred`

```shell
kubectl create secret docker-registry regcred --docker-server --docker-username --docker-password --docker-email
```

where

* `` is your Private Docker Registry FQDN.
  Use `httpsindex.docker.iov1` for DockerHub.
* `` is your Docker username.
* `` is your Docker password.
* `` is your Docker email.

You have successfully set your Docker credentials in the cluster as a Secret called `regcred`.

Typing secrets on the command line may store them in your shell history unprotected, and
those secrets might also be visible to other users on your PC during the time that
`kubectl` is running.

# # Inspecting the Secret `regcred`

To understand the contents of the `regcred` Secret you created, start by viewing the Secret in YAML format

```shell
kubectl get secret regcred --outputyaml
```

The output is similar to this

```yaml
apiVersion v1
kind Secret
metadata
  ...
  name regcred
  ...
data
  .dockerconfigjson eyJodHRwczovL2luZGV4L ... J0QUl6RTIifX0
type kubernetes.iodockerconfigjson
```

The value of the `.dockerconfigjson` field is a base64 representation of your Docker credentials.

To understand what is in the `.dockerconfigjson` field, convert the secret data to a
readable format

```shell
kubectl get secret regcred --outputjsonpath.data..dockerconfigjson  base64 --decode
```

The output is similar to this

```json
authsyour.private.registry.example.comusernamejanedoe,passwordxxxxxxxxxxx,emailjdoeexample.com,authc3R...zE2
```

To understand what is in the `auth` field, convert the base64-encoded data to a readable format

```shell
echo c3R...zE2  base64 --decode
```

The output, username and password concatenated with a ``, is similar to this

```none
janedoexxxxxxxxxxx
```

Notice that the Secret data contains the authorization token similar to your local `.dockerconfig.json` file.

You have successfully set your Docker credentials as a Secret called `regcred` in the cluster.

# # Create a Pod that uses your Secret

Here is a manifest for an example Pod that needs access to your Docker credentials in `regcred`

 code_sample filepodsprivate-reg-pod.yaml

Download the above file onto your computer

```shell
curl -L -o my-private-reg-pod.yaml httpsk8s.ioexamplespodsprivate-reg-pod.yaml
```

In file `my-private-reg-pod.yaml`, replace `` with the path to an image in a private registry such as

```none
your.private.registry.example.comjanedoejdoe-privatev1
```

To pull the image from the private registry, Kubernetes needs credentials.
The `imagePullSecrets` field in the configuration file specifies that
Kubernetes should get the credentials from a Secret named `regcred`.

Create a Pod that uses your Secret, and verify that the Pod is running

```shell
kubectl apply -f my-private-reg-pod.yaml
kubectl get pod private-reg
```

To use image pull secrets for a Pod (or a Deployment, or other object that
has a pod template that you are using), you need to make sure that the appropriate
Secret does exist in the right namespace. The namespace to use is the same
namespace where you defined the Pod.

Also, in case the Pod fails to start with the status `ImagePullBackOff`, view the Pod events

```shell
kubectl describe pod private-reg
```

If you then see an event with the reason set to `FailedToRetrieveImagePullSecret`,
Kubernetes cant find a Secret with name (`regcred`, in this example).

Make sure that the Secret you have specified exists, and that its name is spelled properly.
```shell
Events
  ...  Reason                           ...  Message
       ------                                -------
  ...  FailedToRetrieveImagePullSecret  ...  Unable to retrieve some image pull secrets () attempting to pull the image may not succeed.
```

# # Using images from multiple registries

A pod can have multiple containers, each container image can be from a different registry.
You can use multiple `imagePullSecrets` with one pod, and each can contain multiple credentials.

The image pull will be attempted using each credential that matches the registry.
If no credentials match the registry, the image pull will be attempted without authorization or using custom runtime specific configuration.

# #  heading whatsnext

* Learn more about [Secrets](docsconceptsconfigurationsecret)
  * or read the API reference for
* Learn more about [using a private registry](docsconceptscontainersimages#using-a-private-registry).
* Learn more about [adding image pull secrets to a service account](docstasksconfigure-pod-containerconfigure-service-account#add-imagepullsecrets-to-a-service-account).
* See [kubectl create secret docker-registry](docsreferencegeneratedkubectlkubectl-commands#-em-secret-docker-registry-em-).
* See the `imagePullSecrets` field within the [container definitions](docsreferencekubernetes-apiworkload-resourcespod-v1#containers) of a Pod