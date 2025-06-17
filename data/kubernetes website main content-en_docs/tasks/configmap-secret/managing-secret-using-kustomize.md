---
title Managing Secrets using Kustomize
content_type task
weight 30
description Creating Secret objects using kustomization.yaml file.
---

`kubectl` supports using the [Kustomize object management tool](docstasksmanage-kubernetes-objectskustomization) to manage Secrets
and ConfigMaps. You create a *resource generator* using Kustomize, which
generates a Secret that you can apply to the API server using `kubectl`.

# #  heading prerequisites

# # Create a Secret

You can generate a Secret by defining a `secretGenerator` in a
`kustomization.yaml` file that references other existing files, `.env` files, or
literal values. For example, the following instructions create a kustomization
file for the username `admin` and the password `1f2d1e2e67df`.

The `stringData` field for a Secret does not work well with server-side apply.

# # # Create the kustomization file

secretGenerator
- name database-creds
  literals
  - usernameadmin
  - password1f2d1e2e67df

 tab nameFiles
1.  Store the credentials in files. The filenames are the keys of the secret

    ```shell
    echo -n admin  .username.txt
    echo -n 1f2d1e2e67df  .password.txt
    ```
    The `-n` flag ensures that theres no newline character at the end of your
    files.

1.  Create the `kustomization.yaml` file

    ```yaml
    secretGenerator
    - name database-creds
      files
      - username.txt
      - password.txt
    ```
 tab
 tab name.env files
You can also define the secretGenerator in the `kustomization.yaml` file by
providing `.env` files. For example, the following `kustomization.yaml` file
pulls in data from an `.env.secret` file

```yaml
secretGenerator
- name db-user-pass
  envs
  - .env.secret
```
 tab

In all cases, you dont need to encode the values in base64. The name of the YAML
file **must** be `kustomization.yaml` or `kustomization.yml`.

# # # Apply the kustomization file

To create the Secret, apply the directory that contains the kustomization file

```shell
kubectl apply -k
```

The output is similar to

```
secretdatabase-creds-5hdh7hhgfk created
```

When a Secret is generated, the Secret name is created by hashing
the Secret data and appending the hash value to the name. This ensures that
a new Secret is generated each time the data is modified.

To verify that the Secret was created and to decode the Secret data,

```shell
kubectl get -k  -o jsonpath.data
```

The output is similar to

```
 password MWYyZDFlMmU2N2Rm, username YWRtaW4
```

```
echo MWYyZDFlMmU2N2Rm  base64 --decode
```

The output is similar to

```
1f2d1e2e67df
```

For more information, refer to
[Managing Secrets using kubectl](docstasksconfigmap-secretmanaging-secret-using-kubectl#verify-the-secret) and
[Declarative Management of Kubernetes Objects Using Kustomize](docstasksmanage-kubernetes-objectskustomization).

# # Edit a Secret #edit-secret

1.  In your `kustomization.yaml` file, modify the data, such as the `password`.
1.  Apply the directory that contains the kustomization file

    ```shell
    kubectl apply -k
    ```

    The output is similar to

    ```
    secretdb-user-pass-6f24b56cc8 created
    ```

The edited Secret is created as a new `Secret` object, instead of updating the
existing `Secret` object. You might need to update references to the Secret in
your Pods.

# # Clean up

To delete a Secret, use `kubectl`

```shell
kubectl delete secret db-user-pass
```

# #  heading whatsnext

- Read more about the [Secret concept](docsconceptsconfigurationsecret)
- Learn how to [manage Secrets using kubectl](docstasksconfigmap-secretmanaging-secret-using-kubectl)
- Learn how to [manage Secrets using config file](docstasksconfigmap-secretmanaging-secret-using-config-file)
