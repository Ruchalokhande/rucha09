## 1. Cluster Pre-checks — Air-Gapped Environment

### 1.1 Access the VKS Cluster

**List available VCF contexts**

```bash
vcf context list
```

**Switch to the target VKS cluster**

```bash
vcf context use <TARGET_VKS_CLUSTER_NAME>
```

**Verify the current Kubernetes context**

```bash
kubectl config current-context
```

### 1.2 Validate Cluster Health and Permissions

**Check Kubernetes API server connectivity**

```bash
kubectl cluster-info
```

**Check Kubernetes client and server versions**

```bash
kubectl version
```

**Verify node status and node IP addresses**

```bash
kubectl get nodes -o wide
```

**Check permission to create CustomResourceDefinitions, ClusterRoles, namespaces, ClusterRoleBindings**

```bash
kubectl auth can-i create customresourcedefinitions
kubectl auth can-i create clusterroles
kubectl auth can-i create namespaces
kubectl auth can-i create clusterrolebindings
```
### 1.3 Transfer and Register VKS Standard Packages for the Air-Gapped Environment

**Run the following command to copy the VKS Standard Packages bundle, prepared in the internet-connected environment, to the internal Harbor registry used by the air-gapped environment.**

#### 1.3.1 Copy VKS Standard Packages from the Internet-Connected Environment to Harbor

Copy the VKS Standard Packages bundle and its referenced images to the internal Harbor registry.

```bash
skopeo copy --all \
  --preserve-digests \
  docker://projects.packages.broadcom.com/vsphere/supervisor/vks-standard-packages/3.6.0-20260211/vks-standard-packages:3.6.0-20260211 \
  docker://harbor.sunfire.lab/vsphere/supervisor/vks-standard-packages/3.6.0-20260211/vks-standard-packages:3.6.0-20260211
```
#### 1.3.2 Verify Harbor Registry Connectivity

```bash
curl -I https://harbor.sunfire.lab/
```
#### 1.3.3 Register the VKS Standard Packages Repository

**After copying the bundle to Harbor, run the following command in the air-gapped environment to register the repository with VCF.**

**Register the VKS Standard Packages repository in the air-gapped environment**

```bash
vcf package repository add standard-packages --url harbor.sunfire.lab/vsphere/supervisor/vks-standard-packages/3.6.0-20260211/vks-standard-packages:3.6.0-20260211 -n vmware-system-tkg
```

#### 1.3.4 Verify the Registered VKS Standard Packages

```bash
vcf package available list -n vmware-system-tkg
```

Expected result: The command should display the packages available in the registered repository. This confirms package availability, not installation or runtime status.

| **Package name** | **Description** |
|---|---|
| ako.kubernetes.vmware.com | Avi Kubernetes Operator |
| antrea.tanzu.vmware.com | Antrea |
| cert-manager.kubernetes.vmware.com | cert-manager |
| cluster-autoscaler.kubernetes.vmware.com | Cluster Autoscaler |
| contour.kubernetes.vmware.com | Contour |
| external-dns.kubernetes.vmware.com | ExternalDNS |
| fluent-bit.kubernetes.vmware.com | Fluent Bit |
| harbor.kubernetes.vmware.com | Harbor |
| istio.kubernetes.vmware.com | Istio |
| prometheus.kubernetes.vmware.com | Prometheus |
| sriov-network-device-plugin.kubernetes.vmware.com | SR-IOV Network Device Plugin |
| telegraf.kubernetes.vmware.com | Telegraf |
| vault-injector.kubernetes.vmware.com | Vault Injector Service |
| velero.kubernetes.vmware.com | Velero |
| vsphere-pv-csi-webhook.kubernetes.vmware.com | vSphere PV CSI Webhook |
| windows-gmsa-webhook.kubernetes.vmware.com | Windows GMSA Webhook |

**Note: The output confirms that these packages are available for discovery in the namespace. It does not mean that all packages are installed or running in the cluster.In the following sections, cert-manager, Istio Ingress Gateway, and ExternalDNS will be installed and configured in the air-gapped VKS environment. The required VKS standard packages are now available through the internal Harbor registry and the registered package repository.**

## 2. Install Cert-manager, Istio and External DNS Using VCF Package

### 2.1 Install cert-manager

#### 2.1.1 Retrieve Details About the Available cert-manager Package

```bash
vcf package available get \
  cert-manager.kubernetes.vmware.com \
  -n vmware-system-tkg
```

#### 2.1.2 Set the Package Version

```bash
export CERT_MANAGER_VERSION=1.19.2+vmware.1-vks.1
export CERT_MANAGER_RELEASE_NAME=cert-manager.kubernetes.vmware.com
```

#### 2.1.3 Generate and Review the Data Values

```bash
vcf package available get \
  cert-manager.kubernetes.vmware.com/${CERT_MANAGER_VERSION} \
  -n vmware-system-tkg \
  --default-values-file-output cert-manager-data-values.yaml
cat cert-manager-data-values.yaml
```

#### 2.1.4 Perform a Dry Run

```bash
vcf package install cert-manager \
  --package cert-manager.kubernetes.vmware.com \
  --version "${CERT_MANAGER_VERSION}" \
  -n vmware-system-tkg \
  --values-file cert-manager-data-values.yaml \
  --dry-run
```

#### 2.1.5 Install cert-manager

```bash
vcf package install cert-manager \
  --package cert-manager.kubernetes.vmware.com \
  --version "${CERT_MANAGER_VERSION}" \
  -n vmware-system-tkg \
  --values-file cert-manager-data-values.yaml \
  --wait
```

#### 2.1.6 Verify the Installation

```bash
kubectl get packageinstall -n vmware-system-tkg
kubectl get namespaces
kubectl get all -n cert-manager
```

### 2.2 Install Istio

#### 2.2.1 Check Available Istio Releases

```bash
vcf package available list -A
vcf package available get \
  istio.kubernetes.vmware.com \
  -n vmware-system-tkg
```

#### 2.2.2 Set the Istio Version

```bash
export ISTIO_VERSION=1.28.2+vmware.1-vks.1
export ISTIO_RELEASE_NAME=istio.kubernetes.vmware.com
```

#### 2.2.3 Generate and Review the Data Values File

```bash
vcf package available get \
  istio.kubernetes.vmware.com/${ISTIO_VERSION} \
  -n vmware-system-tkg \
  --default-values-file-output istio-data-values.yaml
cat istio-data-values.yaml
```

#### 2.2.4 Enable the Ingress Gateway

Update istio-data-values.yaml to enable the Istio ingress gateway while retaining the other required package settings.

```yaml
istio:
  gateways:
    ingress:
      enabled: true
```

Verify the updated configuration:

```bash
cat istio-data-values.yaml
```

#### 2.2.5 Perform a Dry Run

```bash
vcf package install istio \
  --package istio.kubernetes.vmware.com \
  --version "${ISTIO_VERSION}" \
  -n vmware-system-tkg \
  --values-file istio-data-values.yaml \
  --dry-run
```

#### 2.2.6 Install Istio

```bash
vcf package install istio \
  --package istio.kubernetes.vmware.com \
  --version "${ISTIO_VERSION}" \
  -n vmware-system-tkg \
  --values-file istio-data-values.yaml \
  --wait
```

#### 2.2.7 Verify the Istio Installation

```bash
kubectl get packageinstall istio \
  -n vmware-system-tkg
kubectl get namespaces
kubectl get all -n istio-ingress
kubectl get all -n istio-system
```

#### 2.2.8 Confirm Istio and cert-manager Are Ready

```bash
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n istio-ingress
```

Expected result: All expected pods should be in the Running state, and the Istio ingress gateway should be deployed successfully.

### 2.3 Install and Configure ExternalDNS

#### 2.3.1 Create Namespace

```bash
kubectl create namespace tanzu-system-service-discovery
```

#### 2.3.2 Add Label to the Namespace

```bash
kubectl label namespace tanzu-system-service-discovery pod-security.kubernetes.io/enforce=privileged --overwrite
```

#### 2.3.3 Configure BIND DNS for ExternalDNS

##### 2.3.3.1 Create the DNS Zone

Create the lab22.sunfire.lab DNS zone on the BIND DNS server. This zone will be used by ExternalDNS to create and update DNS records for the VKS cluster.

##### 2.3.3.2 Allow Queries from the VKS Host Network

Update /etc/named.conf on the BIND DNS server to allow queries from the VKS host network (10.22.92.0/24) and the existing network (10.22.99.0/24).

```text
allow-query {
    localhost;
    10.22.99.0/24;
    10.22.92.0/24;
};
```

Note: allow-query permits DNS queries; it does not authorize DNS record updates. Configure the appropriate BIND zone with a TSIG-authenticated update-policy or allow-update rule for ExternalDNS to create or update records.

Validate the configuration and reload BIND:

```bash
named-checkconf
rndc reload
```

#### 2.3.4 Create the ExternalDNS TSIG Secret

The BIND DNS server provides the TSIG key in /etc/named/keys/externaldns.key. Use the actual secret value from the BIND DNS server.

```bash
cat /etc/named/keys/externaldns.key
```
Expected:
```
output:
key "externaldns-key" {
        algorithm hmac-sha256;
        secret <BIND_TSIG_SECRET>                                         # - Eg. "RiF7fBcpYWFR8zzQdFMfPyO7UGUJwmstUDjatZ+g4cs="  
};
```

```bash
kubectl create secret generic externaldns-tsig \
  -n tanzu-system-service-discovery \
  --from-literal=tsig-secret='<BIND_TSIG_SECRET>'
```

Replace \<BIND_TSIG_SECRET\> with the actual TSIG secret value.

Verify that the Secret exists:

```bash
kubectl get secret externaldns-tsig -n tanzu-system-service-discovery
```

#### 2.3.5 Install ExternalDNS

##### 2.3.5.1 Define the ExternalDNS Version and Package Variables

```bash
export EXTERNAL_DNS_VERSION="0.20.0+vmware.1-vks.1"
export EXTERNAL_DNS_PACKAGE="external-dns.kubernetes.vmware.com"
```

##### 2.3.5.2 Get the Available ExternalDNS Package

```bash
vcf package available get external-dns.kubernetes.vmware.com -n vmware-system-tkg
```

##### 2.3.5.3 Get Package Details in YAML Format

```bash
vcf package available get external-dns.kubernetes.vmware.com -n vmware-system-tkg -o yaml
```

##### 2.3.5.4 Export Default Values to the Values File

```bash
vcf package available get "external-dns.kubernetes.vmware.com/$EXTERNAL_DNS_VERSION" -n vmware-system-tkg --default-values-file-output external-dns-values.yaml
```

##### 2.3.5.5 Configure external-dns-values.yaml

Please use the file below as a reference.

```yaml
deployment:
  args:
    - --source=gateway-httproute
    - --provider=rfc2136                              #BIND using RFC2136 provider name
    - --registry=txt
    - --txt-owner-id=vks-external-dns-lab22           #use a different ID for independent instances
    - --domain-filter=lab22.sunfire.lab               #it only updates entry with "lab22.sunfire.lab" domain
    - --policy=upsert-only
    - --rfc2136-host=10.22.99.5                       #bind DNS IP
    - --rfc2136-port=53
    - --rfc2136-zone=lab22.sunfire.lab                #zone name
    - --rfc2136-tsig-keyname=externaldns-key          #In bind DNS server externaldns.key has "externaldns-key" key variable set.
    - --rfc2136-tsig-secret=$(RFC2136_TSIG_SECRET)    #passing env variable
    - --rfc2136-tsig-secret-alg=hmac-sha256
    - --rfc2136-tsig-axfr                             #zone transfer flag
    - --rfc2136-min-ttl=300s                          #ttl set to 300s
 
  env:
    - name: RFC2136_TSIG_SECRET                       
      valueFrom:
        secretKeyRef:
          name: externaldns-tsig                      #pass the secret
  resources:
    limits:
      cpu: ""
      memory: ""
    requests:
      cpu: 100m
      memory: 128Mi

namespace: tanzu-system-service-discovery

nsLimitRange:
  defaultLimits:
    cpu: ""
    memory: ""
  defaultRequests:
    cpu: 100m
    memory: 128Mi

priorityClassName: ""

tlsConfig:
  ca_crt: ""
  tls_enable: false
```

##### 2.3.5.6 Perform Dry Run

```bash
vcf package install external-dns \
  -p external-dns.kubernetes.vmware.com \
  -v "$EXTERNAL_DNS_VERSION" \
  -n vmware-system-tkg \
  --values-file external-dns-values.yaml \
  --dry-run
```

##### 2.3.5.7 Install the ExternalDNS Package

```bash
vcf package install external-dns \
  -p external-dns.kubernetes.vmware.com \
  -v "$EXTERNAL_DNS_VERSION" \
  -n vmware-system-tkg \
  --values-file external-dns-values.yaml \
  --wait
```

#### 2.3.6 Create the ExternalDNS ClusterRole and ClusterRoleBinding

The ClusterRole provides ExternalDNS with permissions to discover Kubernetes resources such as Services, Gateways, and HTTPRoutes.

File: `external-dns-clusterrole.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/external-dns-cluster-roles.yaml

Apply the ClusterRole:

```bash
kubectl apply -f external-dns-clusterrole.yaml
```

The ClusterRoleBinding assigns the permissions to the ExternalDNS ServiceAccount.

File: `external-dns-clusterrolebinding.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/external-dns-clusterrolebinding.yaml

Apply the ClusterRoleBinding:

```bash
kubectl apply -f external-dns-clusterrolebinding.yaml
```

#### 2.3.7 Restart ExternalDNS and Monitor Logs

```bash
kubectl rollout restart deployment external-dns \
  -n tanzu-system-service-discovery
kubectl rollout status deployment external-dns \
  -n tanzu-system-service-discovery
```

Check the pods:

```bash
kubectl get pods -n tanzu-system-service-discovery
```

Monitor ExternalDNS logs:

```bash
kubectl logs \
  -n tanzu-system-service-discovery \
  -l app=external-dns \
  -f
```

#### 2.3.8 Confirm HTTPRoute Permissions

```bash
kubectl auth can-i get httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
kubectl auth can-i list httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
kubectl auth can-i watch httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
```

Expected result: yes for all three commands.

#### 2.3.9 Validate the ExternalDNS Installation

Check the ExternalDNS pods:

```bash
kubectl get pods -n tanzu-system-service-discovery
```

Check the package installation status. Run this command 

```bash
kubectl get packageinstall -n vmware-system-tkg
```

Monitor ExternalDNS logs for errors and DNS update activity:

```bash
kubectl logs \
  -n tanzu-system-service-discovery \
  -l app=external-dns \
  -f
```

Expected result: The ExternalDNS pod is Running, and the logs show no errors.

## Tekton pipelines and Dashboard Installation in an Air-Gapped Environment

## 3. Install Tekton Pipelines

### 3.1 Create the namespace

Create the namespace for Tekton pipelines :

```bash
kubectl create namespace tekton-pipelines
```

Add the privileged Pod Security Admission label:

```bash
kubectl label namespace tekton-pipelines \
  pod-security.kubernetes.io/enforce=privileged --overwrite
```

Verify the namespace and label:

```bash
kubectl get namespace tekton-pipelines --show-labels
```

```bash
kubectl get namespace tekton-pipelines \
  -o jsonpath='{.metadata.labels.pod-security\.kubernetes\.io/enforce}{"\n"}'
```

Expected output from the last command:

```text
privileged
```
## 4. Download and prepare the Tekton pipelines release manifest

On the internet-connected VM, create a working directory and download
the release manifest:

```bash
mkdir -p ~/tekton-pipeline
cd ~/tekton-pipeline
wget https://infra.tekton.dev/tekton-releases/pipeline/v1.17.0/release.yaml
mv release.yaml tekton-release.yaml
scp tekton-release.yaml root@<target-server-ip>:/<add the path>
```

Transfer tekton-release.yaml to the air-gapped environment using your
organization's approved transfer method.

**Important:** Before proceeding with the Tekton installation, refer to the **Tekton Air-Gapped Image Download and Mirroring Guide** to download and push all required images to the internal Harbor registry. Update the image references in `tekton-release.yaml` to point to the corresponding Harbor images. 

A total of 9 image references must be updated at the line numbers: 26887, 26891, 26892, 26893, 26894, 26898, 27075, 27902, and 28137.

**Reference Document:** 
For detailed instructions on downloading, transferring, and pushing the required Tekton all  images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md 

**Update the Release Manifests**

Update the images references in  tekton-release.yaml to use the corresponding Harbor images.

| Component | GHCR Image | Harbor Image |
|---|---|---|
| Controller | `ghcr.io/tektoncd/pipeline/controller-10a3e32792f33651396d02b6855a6e36:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/controller-10a3e32792f33651396d02b6855a6e36:v1.17.0` |
| Entrypoint | `ghcr.io/tektoncd/pipeline/entrypoint-bff0a22da108bc2f16c818c97641a296:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/entrypoint-bff0a22da108bc2f16c818c97641a296:v1.17.0` |
| Nop  | `ghcr.io/tektoncd/pipeline/nop-8eac7c133edad5df719dc37b36b62482:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/nop-8eac7c133edad5df719dc37b36b62482:v1.17.0` |
| Sidecarlogresults | `ghcr.io/tektoncd/pipeline/sidecarlogresults-7501c6a20d741631510a448b48ab098f:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/sidecarlogresults-7501c6a20d741631510a448b48ab098f:v1.17.0` |
| Workingdirinit | `ghcr.io/tektoncd/pipeline/workingdirinit-0c558922ec6a1b739e550e349f2d5fc1:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/workingdirinit-0c558922ec6a1b739e550e349f2d5fc1:v1.17.0` |
| Events| `ghcr.io/tektoncd/pipeline/events-a9042f7efb0cbade2a868a1ee5ddd52c:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/events-a9042f7efb0cbade2a868a1ee5ddd52c:v1.17.0` |
| Resolvers  | `ghcr.io/tektoncd/pipeline/resolvers-ff86b24f130c42b88983d3c13993056d:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/resolvers-ff86b24f130c42b88983d3c13993056d:v1.17.0` |
| Webhook | `ghcr.io/tektoncd/pipeline/webhook-d4749e605405422fd87700164e31b2d1:v1.17.0` | `harbor.sunfire.lab/tektoncd/pipeline/webhook-d4749e605405422fd87700164e31b2d1:v1.17.0` |
| BusyBox | `cgr.dev/chainguard/busybox@sha256:19f02276bf8dbdd62f069b922f10c65262cc34b710eea26ff928129a736be791` | `harbor.sunfire.lab/tektoncd/chainguard/busybox:tekton-v1.17.0` |

## 5. Install Tekton pipelines

In the air-gapped environment, apply the updated manifest:

```bash
kubectl apply -f tekton-release.yaml
```

### 5.1 Validate Tekton pipelines

Check the namespace pods:

```bash
kubectl get pods -n tekton-pipelines
```

Check Tekton custom resource definitions and API resources:

```bash
kubectl get crd | grep tekton
```

```bash
kubectl api-resources | grep tekton
```

Check the controller logs:

```bash
kubectl logs deployment/tekton-pipelines-controller \
  -n tekton-pipelines --tail=50
```

Expected validation results:

- Namespace tekton-pipelines is Active.
- tekton-pipelines-controller and tekton-pipelines-webhook pods
  are Running and ready.
- Tekton CRDs, including Tasks, TaskRuns, Pipeline, and PipelineRuns,
  are present.
- Controller logs do not show unexplained errors, panics, failed
  connections, or image-pull failures.

Use kubectl get pods -n tekton-pipelines to confirm the actual pod
names and readiness.

## 6. Install Tekton Dashboard

Use a VM with internet access to download the Tekton Dashboard release
manifest and copy the Dashboard image to Harbor.

### 6.1. Download and prepare the Tekton Dashboard release manifest

On the internet-connected VM:

```bash
cd  ~/tekton-pipeline
wget https://infra.tekton.dev/tekton-releases/dashboard/v0.73.0/release.yaml
mv release.yaml tekton-dashboard-release.yaml
scp tekton-dashboard-release.yaml root@<target-server-ip>:/<add the path>
```

Transfer tekton-dashboard-release.yaml to the air-gapped environment
using the approved transfer method.

**Important:** Before proceeding with the Tekton Dashboard Installation, refer to the Tekton Air-Gapped Image Download and Mirroring Guide to download and push all required images to the internal Harbor registry. Update the image references in tekton-dashboard-release.yaml to point to the corresponding Harbor images, then apply the modified manifest to install Tekton in the air-gapped environment.

**Reference Document:** 
For detailed instructions on downloading, transferring, and pushing the required Tekton all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in tekton-dashboard-release.yaml to use the corresponding Harbor images.

| Component | GHCR Image | Harbor Image |
|---|---|---|
| dashboard | `ghcr.io/tektoncd/dashboard/dashboard-9623576a202fe86c8b7d1bc489905f86:v0.73.0` | `harbor.sunfire.lab/tektoncd/dashboard/dashboard-9623576a202fe86c8b7d1bc489905f86:v0.73.0` |

### 6.2. Install Tekton Dashboard

Apply the updated Dashboard manifest in the air-gapped environment:

File: tekton-dashboard-release.yaml

```bash
kubectl apply -f tekton-dashboard-release.yaml
```

### 6.3 Validate Tekton Dashboard

```bash
kubectl get pods,deployments,services -n tekton-pipelines | grep dashboard
```

Expected result:

- The Tekton Dashboard pod is Running and ready.
- The Dashboard Deployment has its desired replicas available.
- The Dashboard Service exists in namespace tekton-pipelines.

### 6.4 Create Kubernetes secrets

There are two credentials cert-manager needs:
1. BIND TSIG secret(TOKEN) — to create DNS challenge records.
2. OpenBao EAB secret — to register its ACME account.

Both will be stored in the cert-manager namespace because we are creating a ClusterIssuer/Issuer.

Note: We will refer this same secrets for argocd Issuer and travelportal ClusterIssuer

#### 6.4.1 Create the BIND TSIG secret
Extract the secret value from:
cat /etc/named/keys/certmanager-acme.key
```
key "certmanager-acme" {
        algorithm hmac-sha256;
        secret "RzBsOraf4CB/NXNAoFG+GTShVMMO8wljV4GlzXCtIug=";
};
```
It is the value inside the secret field.

Create the Kubernetes Secret:
```
kubectl create secret generic certmanager-bind-tsig \
  -n cert-manager \
  --from-literal=tsig-secret='RzBsOraf4CB/NXNAoFG+GTShVMMO8wljV4GlzXCtIug='
```
Verify:
```
kubectl get secret certmanager-bind-tsig -n cert-manager
```

Do not Base64-encode the value yourself when using --from-literal; Kubernetes handles that.

#### 6.4.2 Generate the OpenBao EAB credentials
On the OpenBao server, execute:
```
bao write -f pki/roles/lab22-services-acme/acme/new-eab
```
Output:
```
Key               Value
---               -----
acme_directory    roles/lab22-services-acme/acme/directory
created_on        2026-11-04T14:14:30+05:30
id                6ec4476a-fea8-dda1-aa07-8f307da750cc
key               vault-eab-0-7aBarmA6_T2ohKLhjlmoqBfHEOzutnIaRUUqFJRpP6c
key_type          hs
```
The output should contain an EAB key ID(id) and HMAC key(key). Keep both values secure.

The EAB must be generated for the service-acme directory, which is the directory configured in the ClusterIssuer below. 

Convert the returned HMAC key to Base64URL format:
```
EAB_KEY='vault-eab-0-7aBarmA6_T2ohKLhjlmoqBfHEOzutnIaRUUqFJRpP6c'           #vault-eab-0-7aBarmA6_T2ohKLhjlmoqBfHEOzutnIaRUUqFJRpP6c

EAB_KEY_B64URL=$(printf '%s' "$EAB_KEY" |
  tr '+/' '-_' |
  tr -d '=')
```
Important: Above commands does not actually perform Base64 encoding. It only converts an already Base64-encoded value into Base64URL-compatible form.

Create the EAB secret:
```
kubectl create secret generic services-openbao-eab \
  -n cert-manager \
  --from-literal=secret="$EAB_KEY_B64URL"
```

Verify:
```
kubectl get secret services-openbao-eab -n cert-manager
```

#### 6.4.3 If the OpenBao API SSL certificate is self-signed, use that certificate as the trust anchor.

Encode it: We will refer output of below command inside "Issuer/ClusterIssuer" yaml
```
base64 -w0 /etc/openbao/tls/bao.crt
```
Copy the resulting Base64 string. We will use it as caBundle in the ClusterIssuer/Issuer.

### 6.5 Create a Openbao ClusterIssuer

File dashboard-openbao-clusterissuer.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/dashboard-openbao-clusterissuer.yaml

Apply the ClusterIssuer:

```bash
kubectl apply -f dashboard-openbao-clusterissuer.yaml
```

Check its status:

```bash
kubectl get clusterissuer dashboard-openbao-clusterissuer
```

## 7. Istio Integration: Expose the Dashboard

All three resources are created in tekton-pipelines, the namespace of the tekton-dashboard Service, so no cross-namespace reference is needed. Apply them in this order: Certificate, Gateway, HTTPRoute. All files are in manifest/tekton-catalyst.

### 7.1 TLS certificate (cert-manager)

File dashboard-openbao-cert.yaml

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/dashboard-openbao-cert.yaml

| **Important argument** | **Value** |
|---|---|
| issuerRef | dashboard-openbao-clusterissuer |
| dnsNames | tekton.vks.lab22.sunfire.lab |
| secretName | tekton-dashboard-tls-cert |
| commonName | tekton.vks.lab22.sunfire.lab |

Apply :

```bash
kubectl apply -f dashboard-openbao-cert.yaml
```

Check its status:

```bash
kubectl get certificate -n tekton-pipelines # wait for READY=True
```

### 7.2 tekton-dashboard-gateway

File tekton-dashboard-gateway.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/tekton-dashboard-gateway.yaml

| **Important argument** | **Value** |
|---|---|
| gatewayClassName | istio |
| Listener | HTTPS, port 443, tls.mode: Terminate |
| hostname | tekton.lab22.sunfire.lab |
| certificateRefs | tekton-dashboard-tls-cert |
| allowedRoutes | Same namespace only |

```bash
kubectl get namespace tekton-pipelines --show-labels # it should be "enforce=privileged" if not "kubectl label namespace tekton-pipelines pod-security.kubernetes.io/enforce=privileged --overwrite"
kubectl apply -f tekton-dashboard-gateway.yaml
kubectl get gateway tekton-dashboard-gateway -n tekton-pipelines
```

Expected result: PROGRAMMED=True with an ADDRESS. External DNS publishes this IP.

```bash
kubectl get gateway tekton-dashboard-gateway -n tekton-pipelines \
-o jsonpath='{.status.addresses[0].value}{"\n"}'
```

### 7.3 HTTPRoute

File tekton-dashboard-httproute.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/tekton-dashboard-httproute.yaml

Apply:

```bash
kubectl apply -f tekton-dashboard-httproute.yaml
```

| **Important argument** | **Value** |
|---|---|
| parentRefs | Gateway tekton-dashboard-gateway |
| hostname | tekton.lab22.sunfire.lab (External DNS reads this value) |
| backendRefs | Service tekton-dashboard, port 9097 |

### 7.4 Validation: Istio routing

```bash
kubectl get certificate,gateway,httproute -n tekton-pipelines
kubectl describe httproute tekton-dashboard-route -n tekton-pipelines
```

Expected result: Certificate READY=True, Gateway PROGRAMMED=True, and HTTPRoute conditions Accepted=True and ResolvedRefs=True.

## 8. External DNS Reconciliation

### 8.1 Rolling restart (when required)

If the record is not created after the HTTPRoute is applied, restart External DNS so it re-reads the resources:

```bash
kubectl rollout restart deployment external-dns -n tanzu-system-service-discovery
kubectl rollout status deployment external-dns -n tanzu-system-service-discovery
kubectl get pods -n tanzu-system-service-discovery
kubectl logs -n tanzu-system-service-discovery -l app=external-dns --tail=50
```

Confirm the deployment name (kubectl get deployment -n tanzu-system-service-discovery) and whether the restart is needed in the validated sequence (TO VERIFY).

Expected result: a log line showing the record for tekton.lab22.sunfire.lab being created or updated.

### 8.2 Verify the DNS record

```bash
nslookup # Update Domain Name
```

Expected result: the IP printed equals the Gateway address from Section 7.2. Also confirm in Cloudflare that the A record and its TXT ownership record exist.

## 9. Dashboard Access

Open the Dashboard:

[https://tekton.lab22.sunfire.lab](https://tekton.lab22.sunfire.lab/)

<img width="1677" height="907" alt="image (6)" src="https://github.com/user-attachments/assets/db54052b-dc0a-4b2f-95aa-f6065ba5c8eb" />

The browser must show a valid certificate and the Tekton Dashboard.

## 10. Create the cicd Namespace with its Security Label

All application-specific Tekton resources, including Tasks, Pipelines, TaskRuns, PipelineRuns, and related CI/CD resources, will be created in the cicd namespace unless explicitly configured otherwise. Tekton's core components, such as the Pipelines controller and webhook, remain in the tekton-pipelines namespace.

```bash
# 1. Check
kubectl get namespace cicd

# 2. Create only if it does not exist
kubectl create namespace cicd

# 3. Apply the label immediately
kubectl label namespace cicd \
  pod-security.kubernetes.io/enforce=privileged \
  --overwrite
```

### 10.1 Validation

```bash
kubectl get namespace cicd --show-labels 
kubectl get namespace cicd  -o jsonpath='{.metadata.labels.pod-security\.kubernetes\.io/enforce}{"\n"}'
```

Expected result:cActive, with pod-security.kubernetes.io/enforce=privileged in the labels. / privileged
**Security note:** this label enables privileged capabilities for build workloads. In production, use the least-privilege settings your VKS, BuildKit and Buildpacks implementation and security policy support.

## 11. Smoke Test: Task and TaskRun

This minimal test proves Tekton itself works. Application Tasks (clone, build, sign, update) start in Section 13.

| **File** | **Purpose** |
|---|---|
| task.yaml | Task that prints a test message |
| taskrun.yaml | TaskRun hello-taskrun that executes the Task |

Task: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/task.yaml

TaskRun: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/taskrun.yaml

```bash
kubectl apply -f task.yaml -n cicd
kubectl apply -f taskrun.yaml -n cicd
```

### 11.1 Validation

```bash
kubectl get task,taskrun -n cicd
kubectl logs -l tekton.dev/taskRun=hello-taskrun -n cicd
```

<img width="1672" height="897" alt="image (5)" src="https://github.com/user-attachments/assets/9c4be7d9-ec19-4231-b37f-67c7adff9b19" />

In the Dashboard: Namespace → cicd. The Task and TaskRun are visible.

## 12. Installation Validation and Final Verification

Run once at the end. Every line must pass.

| **Area** | **Command** | **Expected** |
|---|---|---|
| Cluster | kubectl get nodes -o wide | All nodes Ready |
| Tekton | kubectl get pods -n tekton-pipelines | Controller, webhook, dashboard Running |
| Tekton CRDs | kubectl get crd \| grep tekton | tekton.dev CRDs present |
| Namespace | kubectl get namespace cicd --show-labels | Active, enforce=privileged |
| Dashboard | kubectl get pods,svc -n tekton-pipelines \| grep dashboard | 1/1 Running, port 9097 |
| External DNS | kubectl get pods -n tanzu-system-service-discovery | Running, no log errors |
| Istio | kubectl get certificate,gateway,httproute -n tekton-pipelines | Ready, Programmed, Accepted |
| DNS | dig +short tekton.k8slabs.com | Gateway IP |
| User access | Open https://tekton.lab22.sunfire.lab | Dashboard loads |

## 13. Tekton Pipeline Tasks

Create each Task below by applying its manifest from manifest/tekton-catalyst (Figure 1). Each Task lists its important arguments. A single validation at the end of this section checks all of them.

### 13.1 git-clone

git-clone is a Tekton Task used to clone source code from a Git repository into a shared workspace. It cleans the workspace, clones the required branch/revision, and validates the Git repository. The cloned source code is then used by BuildKit or Buildpacks to build the container image.

**Note**: Since the Gitea server uses a custom TLS certificate, its CA certificate must be extracted from the secret and stored in a ConfigMap in the cicd namespace. This ConfigMap is mounted into the git-clone Task to establish CA trust and enable secure Git repository cloning in the air-gapped environment.

**Extract Gitea CA Certificate**

Extract the CA certificate from the gitea-tls-secret Kubernetes Secret.

```
kubectl get secret gitea-tls-secret -n gitea \
  -o jsonpath='{.data.ca\.crt}' \
  | base64 -d > /root/lab22/gitea_ssl/gitea-ca.crt
```
**Create ConfigMap**

Create a ConfigMap in the cicd namespace using the extracted CA certificate.

```
kubectl create configmap lab22-gitea-ca \
  --from-file=ca.crt=/root/lab22/gitea_ssl/gitea-ca.crt \
  -n cicd
```
**Configure Gitea CA in git-clone Task**

Refer to the following YAML file for Gitea CA integration in the git-clone Task. The YAML includes the CA trust configuration, ConfigMap volume mount, and volume definition required for secure Git repository access.
 
File : `git-clone-airgap.yaml`

Below are the changes to add gitea-ca cert inside git-clone task  
```
CA_CERT="/workspace/ca/ca.crt"

echo "Configuring Gitea CA trust..."
test -s "${CA_CERT}"
git config --global \
  http."https://gitea.lab22.sunfire.lab/".sslCAInfo \
  "${CA_CERT}"
echo "CA certificate configured: ${CA_CERT}"
echo ""

# Volume Mount to be added under the step:
volumeMounts:
  - name: gitea-ca
    mountPath: /workspace/ca
    readOnly: true

# Volume to be added under Task spec:
volumes:
  - name: gitea-ca
    configMap:
      name: lab22-gitea-ca
```

#### 13.1.1 Git Clone task create

**Important:** Before creating the git-clone Task, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required container images to the internal Harbor registry. Update the image references in git-clone-airgap.yaml to point to the corresponding Harbor images, and then apply the updated git-clone-airgap.yaml manifest.

Reference Document: For detailed instructions on downloading, transferring, and pushing the required Tekton all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in  git-clone-airgap.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| alpine | `docker.io/alpine/git:v2.54.0` | `harbor22.lab22.sunfire.lab/library/alpine/git` |

File: `git-clone-airgap.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/git-clone-airgap.yaml 

Apply:

```bash
kubectl apply -f  git-clone-airgap.yaml
```

### 13.2 detect-build-type

detect-build-type is a Tekton Task that checks whether the source repository contains a Dockerfile. It selects BuildKit when a Dockerfile exists, otherwise it selects Cloud Native Buildpacks. The selected build type is stored as a Tekton result for the next pipeline task to use.

**Important:** Before creating the Detect Task, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required container images to the internal Harbor registry. Update the image references in detect-airgap.yaml to point to the corresponding Harbor images, and then apply the updated detect-airgap.yaml manifest.

Reference Document: For detailed instructions on downloading, transferring, and pushing the required Tekton all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in  detect-airgap.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| alpine | `docker.io/library/alpine:3.20` | `harbor22.lab22.sunfire.lab/library/alpine` |

File: `detect-airgap.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/detect-airgap.yaml

Apply:

```bash
kubectl apply -f detect-airgap.yaml -n cicd
```

### 13.3 Buildpacks task

Buildpacks automatically detect an application's language and dependencies, build it without requiring a Dockerfile, and package it into a container image that Tekton can push to Harbor.

Prerequisites for Buildpacks Tasks:

1. PersistentVolumeClaim (PVC) for source and buildpack working files
2. ServiceAccount
3. Harbor registry secret
4. Attach the secret to the ServiceAccount
5. kubectl access to the VKS cluster
6. The cicd namespace exists
7. Tekton Pipelines installed
8. vSphere CSI storage available
9. Harbor credentials with push access to the target project

```text
VKS
 |
 v
Buildpacks
 |
 v
lab25-harbor.lab25.sunfire.lab/devops/myapp
```

#### 13.3.1 Create the PVC for Buildpacks

The official Buildpacks/Tekton guide uses a PVC to store the cloned source and buildpack working files. Its example uses 500Mi; for a real application, start larger.

Check the StorageClass

Your VKS environment already has vSphere CSI storage. Confirm which StorageClass is available (and which is the default):

```bash
kubectl get storageclass
```

Look for the StorageClass using the provisioner:

```text
csi.vsphere.vmware.com
```

#### 13.3.2 Apply the PVC manifest

File:  buildpacks-source-pvc.yaml

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/buildpacks-source-pvc.yaml

Apply:

```bash
kubectl apply -f buildpacks-source-pvc.yaml
```

#### 13.3.3 Validate the PVC

```bash
kubectl get pvc -n cicd -w
```

Wait until the status shows:

```text
NAME                    STATUS
buildpacks-source-pvc   Bound
```

#### 13.3.4 Create the ServiceAccount

```bash
kubectl create serviceaccount buildpacks-service-account -n cicd
```

Verify:

```bash
kubectl get sa buildpacks-service-account -n cicd
```

Expected:

```text
NAME                         SECRETS   AGE
buildpacks-service-account   0         ...
```

#### 13.3.5 Create the Harbor Registry Secret

Since images are pushed to Harbor, create a Docker registry secret in the cicd namespace.

Harbor URL: lab25-harbor.lab25.sunfire.lab

```bash
kubectl create secret docker-registry harbor-registry-secret \
  --docker-server=lab25-harbor.lab25.sunfire.lab \
  --docker-username='<HARBOR_USERNAME>' \
  --docker-password='<HARBOR_PASSWORD>' \
  -n cicd
```

Replace \<HARBOR_USERNAME\> and \<HARBOR_PASSWORD\> with your Harbor credentials (a robot account is recommended).

Verify:

```bash
kubectl get secret harbor-registry-secret -n cicd
```

#### 13.3.6 Attach the Harbor Secret to the ServiceAccount

```bash
kubectl patch serviceaccount buildpacks-service-account \
  -n cicd \
  -p '{"imagePullSecrets":[{"name":"harbor-registry-secret"}]}'
```

Verify:

```bash
kubectl get sa buildpacks-service-account -n cicd -o yaml
```

You should see:

```yaml
imagePullSecrets:
  - name: harbor-registry-secret
```

**Tip:** imagePullSecrets covers *pulling* images. Tekton uses credentials from the ServiceAccount's secrets list to *push* to a registry. If the build fails to push to Harbor, also add the secret there:

```bash
kubectl patch serviceaccount buildpacks-service-account \
  -n cicd \
  -p '{"secrets":[{"name":"harbor-registry-secret"}]}'
```

#### 13.3.7 Checklist

| **Resource** | **Name** | **Namespace** |
|---|---|---|
| PVC | buildpacks-source-pvc | cicd |
| ServiceAccount | buildpacks-service-account | cicd |
| Secret | harbor-registry-secret | cicd |

### 13.4 buildpacks-phases

A ready-made Tekton Task that builds an image from source code using Cloud Native Buildpacks. It builds the image when there is no Dockerfile.

#### 13.4.1 Download and Mirror Go Dependencies 

**Note** In an air-gapped environment, Buildpacks cannot download application dependencies directly from external sources such as Go, GitHub, or other public repositories. Therefore, the required builder images and application dependencies must be mirrored to an internally accessible repository before executing the Buildpacks pipeline.

**Verify Buildpacks Builder Image and Go Version**

The builder image paketobuildpacks/builder-jammy-base:0.4.674 includes the Paketo Go Buildpack, which provides Go 1.27.3 as the default Go version for the configured build.

To identify the installed buildpack versions and their supported dependencies, refer to the following release documentation:

https://github.com/paketo-buildpacks/go/releases

**Download Go 1.27.3**

Source URL:
https://go.dev/dl/go1.27.3.linux-amd64.tar.gz

**Upload Go Archive to Gitea**

Upload the downloaded archive to the following Gitea repository location:

https://gitea.lab12.k8slabs.com/admin/TravelPortal-lab11/src/branch/main/dl/go1.27.3.linux-amd64.tar.gz


**Configure Go Dependency Mirror in Pipeline**

Update the below varible in travelportal-pipeline-airgap.yaml

```
- "BP_DEPENDENCY_MIRROR=https://gitea.lab12.k8slabs.com/admin/TravelPortal-lab11/raw/branch/main"
```

#### 13.4.1.1 Configure Java Buildpack Dependencies Mirror

**Note:** In the case of Java Buildpacks, verify which version of the Java Buildpack is included in `paketobuildpacks/builder-jammy-base:0.4.674` before configuring the dependency mirror.

**Verify the Java Buildpack Version**

Run the following command to inspect the builder image available in Harbor:

```bash
docker image inspect \
  lab25-harbor.lab25.sunfire.lab/library/paketobuildpacks/builder-jammy-base@sha256:7f09f46d782798c5a11375d3ed6c4ef41d61c88766cbe008d336c282606860ad
```

Based on the Java Buildpack version identified in the builder image, refer to the following release documentation to verify compatibility with the required Java version and dependencies:

Reference: [Releases – Paketo Java Buildpack](https://github.com/paketo-buildpacks/java/releases)

**Download the Required Java Dependencies**

Identify the Java runtime and other dependencies required by the application and download the corresponding versions before executing the Buildpacks pipeline in the air-gapped environment.

Example source URLs for downloading the required dependencies:

- **Java JDK 21:**
  https://github.com/bell-sw/Liberica/releases/download/21.0.13.0.2+5/bellsoft-jdk21.0.13.0.2+5-linux-amd64.tar.gz

- **Java JRE 21:**
  https://github.com/bell-sw/Liberica/releases/download/21.0.13.0.2+5/bellsoft-jre21.0.13.0.2+5-linux-amd64.tar.gz

- **Syft 1.54.1:**
  https://github.com/anchore/syft/releases/download/v1.54.1/syft_1.54.1_linux_amd64.tar.gz

**Note:** These are example dependency versions. Verify that the selected Java runtime and other dependencies are compatible with the application and the Java Buildpack version included in the builder image. Download and mirror only the dependencies required by the application and build configuration.

**Upload Dependencies to the Internal Mirror Repository**

Upload the downloaded dependency archives to the configured GitHub repository or another repository accessible from the build environment.

Example repository location:

https://github.com/myorg/buildpack-mirror

Organize the downloaded files so that their paths match the URLs expected by the dependency mirror configuration.

**Configure the Java Dependency Mirror in the Pipeline**

Update the following variable in `travelportal-pipeline-airgap.yaml` to point to the root URL of the repository containing the mirrored dependencies:

```yaml
- "BP_DEPENDENCY_MIRROR=https://github.com/myorg/buildpack-mirror/raw/main"
```

Ensure that the configured mirror URL and file paths match the actual locations of the uploaded dependency archives.

**Important:** In an air-gapped environment, the GitHub repository must be accessible from the build environment for this configuration to work. If external GitHub access is unavailable, host the mirrored dependencies in an internally accessible repository, such as Gitea or Harbor, and configure `BP_DEPENDENCY_MIRROR` to use that internal URL.

#### 13.4.2 Configure Harbor CA Certificate Trust

In an air-gapped VKS environment, the Harbor registry may use a private CA certificate. The Buildpacks Task must trust this CA to communicate securely with Harbor without TLS certificate errors such as x509: certificate signed by unknown authority.

**Create the Harbor CA Secret**

Ensure that the Harbor CA certificate is available on the machine where you execute the command.

For example, the certificate may be located at:
```
/etc/docker/certs.d/lab25-harbor.lab25.sunfire.lab/ca.crt
```

Create or update the Kubernetes Secret in the cicd namespace:

```
kubectl create secret generic harbor-ca \
  -n cicd \
  --from-file=ca.crt=/etc/docker/certs.d/lab25-harbor.lab25.sunfire.lab/ca.crt \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Verify the Secret**

```
kubectl get secret harbor-ca -n cicd
```

Verify that the Secret contains the ca.crt key:

```
kubectl get secret harbor-ca -n cicd \
  -o jsonpath='{.data.ca\.crt}' | wc -c
```

The output should be greater than zero.

**Mount the Harbor CA Secret in the Buildpacks Task**

Update buildpacks-phase-task-airgap.yaml to mount the Secret into the get-labels-and-env step.
Add the following volumeMounts entry under the existing get-labels-and-env container. Preserve any existing volume mounts.

```
steps:
  - name: get-labels-and-env
    # Preserve the existing image, script, and other configuration.
    volumeMounts:
      - name: harbor-ca-secret
        mountPath: /etc/containers/certs.d/lab25-harbor.lab25.sunfire.lab
        readOnly: true
```

Add the following volume under the Task's existing spec:

```
volumes:
  - name: harbor-ca-secret
    secret:
      secretName: harbor-ca
```

**Important:** Merge these entries into the existing Task YAML. Do not replace the entire steps or volumes sections if they already contain other configuration.

The certificate will be available inside the step at:

```
/etc/containers/certs.d/lab25-harbor.lab25.sunfire.lab/ca.crt
```

This path is suitable for tools that use the containers/image certificate directory convention, such as Skopeo. The certificate directory hostname must match the registry hostname used by the tool.

#### 13.4.3 Buildpack-task create 

**Important:** Before proceeding with the Buildpack task creation, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required images to the internal Harbor registry. Update the image references in buildpacks-phase-task-airgap.yaml to point to the corresponding images in Harbor, and then apply the buildpacks-phase-task-airgap.yaml file.

Reference Document: For detailed instructions on downloading, transferring, and pushing the required buildpack all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in buildpacks-phase-task-airgap.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| skopeo-jq | `quay.io/halkyonio/skopeo-jq:0.1.3@sha256:1b3d21ad541227dc9d3e793d18cef9eb00a969c0c01eb09cab88997bc63680c6` | `harbor22.lab22.sunfire.lab/library/skopeo-jq` |
| ubi-minimal | `registry.access.redhat.com/ubi8/ubi-minimal@sha256:b2a1bec3dfbc7a14a1d84d98934dfe8fdde6eb822a211286601cf109cbccb075` | `harbor22.lab22.sunfire.lab/library/ubi8/ubi-minimal` |
| python-311 | `registry.access.redhat.com/ubi8/python-311@sha256:43605cb2491ef2297a7acf4b4bf0b7f54f0c91b96daf12ae41c49cc7f192b153` | `harbor22.lab22.sunfire.lab/library/ubi8/python-311` |
| run-jammy-base | `docker.io/paketobuildpacks/run-jammy-base:0.1.294` | `harbor22.lab22.sunfire.lab/library/paketobuildpacks/run-jammy-base` |

**Note:** Following image needs to be update in tekton pipelines `travelportal-pipeline-airgap.yaml`

| Component | Images | Harbor Image |
|---|---|---|
| builder-jammy-base | `docker.io/paketobuildpacks/builder-jammy-base:0.4.675` | `harbor22.lab22.sunfire.lab/library/paketobuildpacks/builder-jammy-base` |

File: `buildpacks-phase-task-airgap.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/buildpacks-phase-task-airgap.yaml

| **Important argument** | **Meaning** |
|---|---|
| Builder image | Buildpacks builder image, passed by the PipelineRun (Section 16) |
| Workspaces source, cache | Source code and per-run build cache PVCs |
| Result | APP_IMAGE_DIGEST |
| Digest file | $(workspaces.source.path)/image-digest |

```bash
kubectl apply -f buildpacks-phase-task-airgap.yaml -n cicd
```

### 13.5 buildkit-build

buildkit-build is a Tekton Task that runs when detect-build-type detects a Dockerfile, using BuildKit to build the application container image. It takes the source code from the shared workspace, builds the image from the Dockerfile, and pushes it to Harbor with the configured credentials.

#### 13.5.1 Configure Harbor CA Certificate Trust for BuildKit

In an air-gapped VKS environment, BuildKit must trust the Harbor CA certificate to securely pull base images and push the built application image to the internal Harbor registry.

**Note:** The harbor-ca Secret was already created in Section 13.4.2. Reuse the same Secret in the buildkit-build Task. Do not create it again.

**Update the BuildKit Task YAML**

File: buildkit-task-airgap.yaml

Add the following volumeMounts entry under the existing build step:

```
steps:
  - name: build
    # Preserve the existing image, script, environment variables,
    # and other configuration.
    volumeMounts:
      - name: harbor-ca-secret
        mountPath: /etc/buildkit/certs
        readOnly: true
```

Add the following volume under the Task's existing spec:

```
volumes:
  - name: harbor-ca-secret
    secret:
      secretName: harbor-ca
```

#### 13.5.2 Configure Go Dependencies for Air-Gapped BuildKit Builds

Go dependencies are being downloaded from `proxy.golang.org`, which is unreachable from the air-gapped environment. To resolve this issue, download the required Go dependencies on an internet-connected VM, vendor them into the application repository, and update the Dockerfile to build using the vendored dependencies.

**Go 1.27.3 Alpine**

From internet-connected VM Pull the standard Go 1.27.3 Alpine image is required in the air-gapped environment and push it to the internal Harbor registry.

Pull and Push the Go base image to internal harbor:

```
docker pull golang:1.27.3-alpine

docker tag golang:1.27.3-alpine \
  harbor22.lab22.sunfire.lab/test/golang:1.27.3-alpine

docker push harbor22.lab22.sunfire.lab/test/golang:1.27.3-alpine
```

**Clone the TravelPortal Repository**

On an internet-connected VM, clone the application repository:

```bash
git clone https://gitea.lab22.sunfire.lab/admin/TravelPortal-lab22.git
cd TravelPortal-lab22
```

Verify the repository contents:

```bash
ls
```

Example output:

```text
config  Dockerfile  go.sum  infrastructure  istio-preview-template.yaml
deploy  go.mod      helm-charts  internal    LICENSE
main.go Makefile   middleware   models      pipeline.yaml
routes  sh          tests        utils       web
```

** Download and Vendor Go Dependencies**

Run the following command to download the Go dependencies and vendor them into the application repository. Execute this command on an internet-connected VM that can access the required Go module sources.

```bash
docker run --rm \
  -v /root/lab22/TravelPortal-lab22:/app \
  -w /app \
  harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine \
  sh -c 'go mod download && go mod vendor'
```

**Note:** Ensure that the host directory mounted into `/app` matches the actual location of your cloned repository. The container image must be available locally or accessible from the registry.

**Verify the Vendor Directory**

After the command completes successfully, verify that the `vendor` directory has been created:

```bash
ls -ld vendor
ls vendor | head
```

Example output:

```text
drwxr-xr-x. 4 root root 59 Oct 27 22:01 vendor
```

Example directory contents:

```text
github.com
gopkg.in
modules.txt
```

The `vendor` directory contains the downloaded application dependencies. Commit or otherwise transfer the updated repository, including the `vendor` directory, to the air-gapped environment.

**Update the Dockerfile**

Update the Go build command in the application's Dockerfile to use the vendored dependencies instead of downloading modules during the build.

Replace the existing build command:

```dockerfile
ENV CGO_ENABLED=1
RUN go build -ldflags "-s -w" -o /app/main .
```

With the following:

```dockerfile
ENV CGO_ENABLED=1
RUN go build -mod=vendor -ldflags "-s -w" -o /app/main .
```

The `-mod=vendor` flag instructs Go to use dependencies from the local `vendor` directory rather than downloading them from external module repositories.

**Create a Custom Go Builder Image with SQLite Dependencies**

If the application requires CGO and SQLite, create a custom Go builder image containing the required compilation tools and SQLite development libraries.

update the `Dockerfile.builder` file:

```
FROM harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine

RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    sqlite-dev
```

Build the custom image:

```bash
docker build -f Dockerfile.builder \
  -t harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine-cgo-sqlite .
```

**Note:** Ensure that the base image `harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine` is available and that the build host can access the required Alpine packages. If the VM is also air-gapped, mirror the required packages or prepare the image on an internet-connected VM before transferring it.

**Step 5: Push the Custom Builder Image to Harbor**

Push the image to the registry that will be used by the build environment:

```bash
docker push harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine-cgo-sqlite
```

Ensure that the target Harbor project and repository exist and that the required push permissions are configured.

**Step 6: Update the Application Dockerfile**

Update the builder stage in the application's Dockerfile to use the custom Go builder image.

Replace the existing base image reference:

```dockerfile
FROM <existing-base-image>
```

With:

```dockerfile
FROM harbor22.lab22.sunfire.lab/build/golang:1.27.3-alpine-cgo-sqlite
```

**Reference:** Use the Dockerfile as the reference when updating the builder stage, and preserve all application-specific build and runtime instructions.

Dockerfile: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/Dockerfile

#### 13.5.3 Buildkit Task Create 

**Important:** Before proceeding with the Buildkit task creation, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required images to the internal Harbor registry. Update the image references in buildkit-task-airgap.yaml to point to the corresponding images in Harbor, and then apply the buildkit-task-airgap.yaml file.

**Reference Document:** For detailed instructions on downloading, transferring, and pushing the required buildkit all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in buildkit-task-airgap.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| buildkit | `docker.io/moby/buildkit:v0.34.1` | `harbor22.lab22.sunfire.lab/library/moby/buildkit` |
        
File: `buildkit-task-airgap.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/buildkit-task-airgap.yaml

| **Important argument** | **Meaning** |
|---|---|
| Harbor image | Target image in Harbor, passed by the PipelineRun |
| Workspace dockerconfig | Harbor credentials (Secret harbor-registry-secret) |
| Harbor CA | ConfigMap harbor-ca-cert mounted through taskRunSpecs in the PipelineRun |
| Registry setting | References the Harbor CA. Does not use registry.insecure=true |
| Result | IMAGE_DIGEST |
| Digest file | $(workspaces.source.path)/image-digest |

```bash
kubectl apply -f buildkit-task-airgap.yaml 
```

### 13.6 sign-image (SBOM + Cosign)

#### 13.6.1 Install and Configure Cosign

**Check the Jumpbox Operating System**

Before installing Cosign, verify the operating system and CPU architecture of your jumpbox.

```bash
cat /etc/os-release
uname -m
```

For an x86_64/amd64 system, you can install Cosign using the official Sigstore release binary.

Official releases: [Cosign Releases](https://github.com/sigstore/cosign/releases)

**Download cosign from internet connected VM and SCP to AirGap and Install Cosign**

Download the latest Cosign binary and install it in /usr/local/bin.

```bash
cd /tmp
```

```bash
curl -fL \
https://github.com/sigstore/cosign/releases/download/v3.1.3/cosign-linux-amd64 \
  -o cosign
scp cosign root@<Airgap-jumphost-ip>:/root/
chmod +x cosign
sudo mv cosign /usr/local/bin/cosign
```

**Note:** We use Cosign v3.1.3 for compatibility with our current Harbor version. You can use a newer version of Cosign if it is compatible with your Harbor deployment. This command is for x86_64/amd64 systems. For ARM64 systems, use the corresponding ARM64 release binary. Ensure /usr/local/bin is in your PATH.

Verify the installation:

```bash
cosign version
```

**Generate the Cosign Key Pair**

Generate a public and private key pair for signing container images.

```bash
cosign generate-key-pair
```

When prompted, did not created any password for the cosign key, Keep it empty
The command generates the following files in the current directory:

- cosign.key — Private key used to sign container images.
- cosign.pub — Public key used to verify image signatures.

Verify that both files exist:

```bash
ls -l cosign.key cosign.pub
```

**Protect the Private Key**

Restrict access to the private key so that only its owner can read or modify it.

```bash
chmod 600 cosign.key
```

**Important:** Never commit cosign.key to Git or share it publicly. Store the key and its password securely.

**Cosign password Secret (cosign-password)**

Password of the Cosign private key.

File cosign-pass.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/cosign-pass.yaml

Note: We have not kept any password while generating key-pair, so we have kept password as blank
Apply:

```bash
kubectl apply -f cosign-pass.yaml
```

**Cosign key pair Secret (cosign-key)**

Cosign private and public key. Create it from the generated key pair (no manifest file):

```bash
kubectl create secret generic cosign-key \
  -n cicd \
  --from-file=cosign.key=cosign.key \
  --from-file=cosign.pub=cosign.pub
```

**Validate the Kubernetes Secret**

Check that the Secret was created successfully:

```bash
kubectl get secret cosign-key -n cicd
```

Expected output:

```text
NAME        TYPE     DATA   AGE
cosign-key  Opaque   2      ...
```

Verify the available keys without displaying their contents:

```bash
kubectl describe secret cosign-key -n cicd
```

The Secret should contain both cosign.key and cosign.pub.

**Integration with Tekton**

Tekton tasks can mount the cosign-key Secret through a workspace or another supported Secret-mounting configuration. The signing task must also receive the private-key password securely.

The private key is used to sign the container image, while the public key can be used to verify its signature.

This sign-image Task is the security/signing stage after BuildKit or Buildpacks. Its job is to generate an SBOM, sign the image with Cosign, attach the SBOM as an attestation, and return the image digest.

#### 13.6.2 Sign-image task create 

**Important:** Before proceeding with the Sign-image task creation, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required images to the internal Harbor registry. Update the image references in sign-image-task-airgap.yaml to point to the corresponding images in Harbor, and then apply the sign-image-task-airgap.yaml file.

**Reference Document:** For detailed instructions on downloading, transferring, and pushing the required buildkit all images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

**Update the Release Manifests**

Update the images references in sign-image-task-airgap.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| syft | `docker.io/anchore/syft:1.55.0` | `harbor22.lab22.sunfire.lab/library/anchore/syft` |
| cosign | `ghcr.io/sigstore/cosign/cosign:v3.1.3` | `harbor22.lab22.sunfire.lab/library/sigstore/cosign/cosign` |
| alpine | `docker.io/library/alpine:3.20` | `harbor22.lab22.sunfire.lab/library/alpine` |

File: `sign-image-task-airgap.yaml`

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/sign-image-task-airgap.yaml

| **Important argument** | **Meaning** |
|---|---|
| COSIGN_TLOG_UPLOAD | Defaults to false for the air-gapped/private-registry lab flow. Set to true only when the cluster can reach the approved Sigstore transparency-log service |
| Image digest | Digest from the build Task. The image is signed by digest, not by tag |
| Workspace cosign-key | Cosign key pair (Secret cosign-key) |
| Cosign password | Secret cosign-password |
| SBOM | Generated with Syft, stored in the source workspace, attached as an spdxjson attestation |
| Output | Image digest, passed to update-values |

```bash
kubectl apply -f sign-image-task-airgap.yaml
```

Note: Cosign signing and SBOM attestation are working. The issue is with Harbor's SBOM display/API behavior, not an established error in your Tekton Task.

### 13.7 Update-values

update-values is the GitOps update stage of the Tekton pipeline. It updates the Helm chart's values.yaml with the newly built image and its digest, commits the change, and pushes it back to the repository.

**Important:** Before proceeding with the update-values Task creation, refer to the Air-Gapped Image Download and Mirroring Guide to download and push all required images to the internal Harbor registry. Update the image references in update-values-airgap-gitea-ca.yaml to point to the corresponding images in Harbor, and then apply. 

**Reference Document:** For detailed instructions on downloading, transferring, and pushing the required images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md 

**Update the Release Manifests**

Update the images references in update-values-airgap-gitea-ca.yaml to use the corresponding Harbor images.

| Component | Images | Harbor Image |
|---|---|---|
| alpine/git | `docker.io/alpine/git:v2.54.0` | `harbor22.lab22.sunfire.lab/library/alpine/git` |

File: update-values-airgap-gitea-ca.yaml

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/update-values-airgap-gitea-ca.yaml

| **Important argument** | **Meaning** |
|---|---|
| Image digest | Received from the sign-image Task |
| Target file | helm-charts/values.yaml |
| Commit message prefix | ci: update TravelPortal image digest (the EventListener CEL filter ignores commits with this prefix) |
| Credentials | username and token keys of Secret repo-git-credentials |

```bash
kubectl apply -f update-values-airgap-gitea-ca.yaml
```

### 13.8 Validation – Tasks

```bash
kubectl get tasks -n cicd
```

Expected result (all six Tasks listed):

```text
git-clone-update
detect-build-type
buildpacks-phases
buildkit-build
sign-image
update-values
```

## 14. Pipeline Dependencies (Storage, ConfigMap, Secrets, Service Account)

Before triggering the Pipeline and PipelineRun, create the required storage, configuration, and credentials. A single validation at the end of this section checks everything.

### 14.1 Source and Build Cache Storage

The PipelineRun creates dedicated source and cache PVCs for each run using the lab-gold-storage-policy StorageClass. This prevents concurrent PipelineRuns from overwriting the same shared source workspace.

If the StorageClass has a different name in your VKS environment, replace lab-gold-storage-policy in the PipelineRun and TriggerTemplate workspace definitions.

### 14.2 Harbor CA ConfigMap

Create a ConfigMap named harbor-ca-cert in the cicd namespace from your Harbor / vCenter CA certificate (the key must be ca.crt):

```bash
kubectl create configmap harbor-ca-cert \
  -n cicd \
  --from-file=ca.crt=harbor-ca.crt
```

### 14.3 Secrets

The pipeline needs four secrets in the cicd namespace. Create them one by one. Replace every placeholder value with your real credentials before applying, and never commit real credentials to Git.

#### 14.3.1 Git credentials Secret (repo-git-credentials)

Follow the steps to generate the token:

##### 14.3.1.1 Steps to create a Gitea PAT

1. Log in to Gitea Open your Gitea URL in a browser and sign in with your account.
2. Open Application settings Click your profile icon → Settings → Applications.
3. Generate a new token Under Manage Access Tokens, enter a token name, such as tekton-cicd-token, and click Generate Token.
4. Copy and save the token Copy the generated PAT immediately and store it securely. You may not be able to view it again.

Create a secret using Git repository username and token, used by the clone and update-values Tasks.

```bash
kubectl create secret generic repo-git-credentials \
  -n cicd \
  --from-literal=username=admin \
  --from-literal=token='<REPO_TOKEN>'
```

The 'username' and 'token' keys from the 'repo-git-credentials' Secret are used by the 'update-values' task as environment variables, as shown below.

```yaml
steps:
    - name: update-and-push
      image: alpine/git:latest
      env:
        - name: GIT_USERNAME
          valueFrom:
            secretKeyRef:
              name: repo-git-credentials
              key: username
        - name: GIT_TOKEN
          valueFrom:
            secretKeyRef:
              name: repo-git-credentials
              key: token
```

#### 14.3.2 Harbor registry Secret (harbor-credentials)

Harbor credential used to sign images.

File harbor-credentials.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/harbor-credentials.yaml

Apply:

```bash
kubectl apply -f harbor-credentials.yaml
```

### 14.4 ServiceAccount, role, and rolebinding for tekton-pipeline in cicd namespace

The TriggerTemplate and PipelineRun use the dedicated tekton-pipeline-sa ServiceAccount.

File tekton-pipeline-sa.yaml:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/tekton-pipeline-sa.yaml

| **Important argument** | **Value** |
|---|---|
| ServiceAccount | tekton-pipeline-sa in cicd |
| Used by | PipelineRun and TriggerTemplate |

Apply:

```bash
kubectl apply -f tekton-pipeline-sa.yaml
```

### 14.5 Validation – Dependencies

```bash
kubectl get storageclass lab-gold-storage-policy
kubectl get configmap harbor-ca-cert -n cicd
kubectl get secret repo-git-credentials harbor-registry-secret harbor-credentials cosign-password cosign-key -n cicd
kubectl get sa tekton-pipeline-sa -n cicd
```

Expected result: every object is found, with no NotFound errors.

## 15. Pipeline

Tasks are added to the Pipeline using taskRef, which references previously created Tekton Tasks. params pass required values to each Task, workspaces share files/credentials, and runAfter/when control the execution order and conditions. The Pipeline first clones the repository, then detects whether a Dockerfile exists and runs BuildKit or Buildpacks accordingly. After the image is built, the sign Task generates the SBOM and signs/attests the image. Finally, the update-values Task receives the image digest from the sign Task, updates values.yaml, commits the change, and pushes it back to the same Git repository—completing the GitOps update.

**Important:** Before proceeding with the Pipeline creation, ensure that all required Tekton Tasks have been created and their corresponding container images have been downloaded and pushed to the internal Harbor registry. Update the image references in the Pipeline configuration to point to the appropriate internal Harbor images, and then apply the travelportal-pipeline-airgap.yaml file.

**Reference Document:** For detailed instructions on downloading, transferring, and pushing the required images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

File: travelportal-pipeline-airgap.yaml

**Configure Go Dependency Mirror in Pipeline**

Update the below varible in travelportal-pipeline-airgap.yaml

```
- "BP_DEPENDENCY_MIRROR=https://gitea.lab12.k8slabs.com/admin/TravelPortal-lab11/raw/branch/main"
```

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/travelportal-pipeline-airgap.yaml

| **Important argument** | **Meaning** |
|---|---|
| Pipeline name | travelportal-pipeline-values-update |
| taskRef | The six Tasks from Section 13 |
| Order | clone → detect → (BuildKit or Buildpacks) → sign → update-values |
| when | Selects BuildKit or Buildpacks from the detect-build-type result |
| runAfter | Sets the order of the Tasks |
| Workspaces | source, cache, dockerconfig, git-credentials, cosign-key |

Apply:

```bash
kubectl apply -f travelportal-pipeline-airgap.yaml
```

The Pipeline is validated together with the PipelineRun in Section 16.1.

## 16. PipelineRun

The PipelineRun is used to start and execute the travelportal-pipeline-values-update Pipeline. It provides the pipeline parameters, connects the required workspaces and secrets, and applies additional Pod configuration needed during the pipeline execution. The pipelineRef selects the travelportal-pipeline-values-update Pipeline, while params provide the Git repository, branch, Harbor image, and Buildpacks builder image that the Pipeline will use. The workspaces provide the required storage and credentials: the source and cache PVCs are created per PipelineRun, dockerconfig provides Harbor authentication, git-credentials provides repository push credentials, and cosign-key provides the image-signing key. The SBOM is stored in the source workspace, so no separate SBOM workspace is required. The taskRunSpecs customize the BuildKit TaskRun by adding the harbor-ca ConfigMap as a volume. This allows the BuildKit Pod to access the Harbor CA certificate. The BuildKit registry configuration references that CA and does not use registry.insecure=true for the Harbor HTTPS endpoint.

**Important:** Before proceeding with the PipelineRun creation, ensure that the Pipeline and all required Tekton Tasks have been created successfully. Verify that the required PVCs, Secrets, and ConfigMaps are available in the cicd namespace. Update the PipelineRun configuration with the appropriate repository details, Harbor image references, workspace configurations, and required credentials, and then apply the travelportal-pipelinerun-airgap.yaml file.

**Reference Document:** For detailed instructions on downloading, transferring, and pushing the required images to Harbor, refer to the following document:

Document Name: tekton-airgap-images.md

Document Link: https://github.com/kondurupurandhar/keda-vks/blob/main/tekton-airgap-images.md

File: travelportal-pipelinerun-airgap.yaml

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/travelportal-pipelinerun-airgap.yaml

| **Important argument** | **Meaning** |
|---|---|
| pipelineRef | travelportal-pipeline-values-update |
| serviceAccountName | tekton-pipeline-sa |
| params | Git repository, branch, Harbor image, Buildpacks builder image |
| Workspace source, cache | PVCs created per run from StorageClass lab-gold-storage-policy |
| Workspace dockerconfig | Secret harbor-registry-secret |
| Workspace git-credentials | Secret repo-git-credentials |
| Workspace cosign-key | Secret cosign-key |
| taskRunSpecs | Adds ConfigMap harbor-ca-cert as a volume to the BuildKit TaskRun |

```bash
kubectl create -f travelportal-pipelinerun-airgap.yaml
```

### 16.1 Validation – Pipeline, PipelineRun, and output

Pipeline and run status:

```bash
kubectl get pipeline -n cicd
kubectl get pipelinerun -n cicd -w
kubectl get pvc -n cicd
```

Expected result: the Pipeline exists, the PipelineRun finishes with SUCCEEDED=True, and the source and cache PVCs are Bound. You can also follow the run in the Tekton Dashboard.

Image in Harbor (from a machine that can reach Harbor):

```bash
docker login lab25-harbor.lab25.sunfire.lab
docker pull lab25-harbor.lab25.sunfire.lab/cicd/travelportal:latest
```

Cosign signature and SBOM attestation:

```bash
cosign verify \
  --offline \
  --key cosign.pub \
  "lab25-harbor.lab25.sunfire.lab/cicd/travelportal@sha256:<IMAGE_DIGEST>"
```

```bash
cosign verify-attestation \
  --offline \
  --key cosign.pub \
  --type spdxjson \
  "lab25-harbor.lab25.sunfire.lab/cicd/travelportal@sha256:<IMAGE_DIGEST>"
```

Expected result: the image pulls, the signature verifies, and the SBOM attestation is associated with the image digest.

## 17. Automatic Pipeline Trigger with Tekton Triggers

Until now the PipelineRun was started by hand with `kubectl apply`. In this section a push to the repository starts it automatically. The repository server sits on the internal lab network, so it calls the Tekton EventListener NodePort directly.

### 17.1 Lab values used in this section

| Item | Value |
|---|---|
| Namespace | `cicd` |
| Pipeline | `travelportal-pipeline-values-update` |
| EventListener | `travelportal-gitea-listener` |
| EventListener Service | `el-travelportal-gitea-listener` (NodePort) |
| Webhook port | `8080 -> 31816` (observed lab mapping) |
| Node used for webhook | `10.10.92.2` |
| Webhook URL | `http://10.10.92.2:31816` (observed lab URL) |
| Repository server | `http://10.12.90.62` |
| Repository | `http://10.12.90.62/admin/TravelPortal-lab10.git` |
| Branch | `main` |

### 17.2 The Tekton Triggers objects created in this section

| Object | Name |
|---|---|
| ServiceAccount | `gitea-trigger-sa` |
| Role | `gitea-trigger-role` |
| RoleBinding | `gitea-trigger-rolebinding` |
| ClusterRole | `gitea-trigger-cluster-role` |
| ClusterRoleBinding | `gitea-trigger-cluster-rolebinding` |
| EventListener | `travelportal-gitea-listener` |
| CEL interceptor | `ClusterInterceptor` |
| TriggerBinding | `travelportal-gitea-binding` |
| TriggerTemplate | `travelportal-gitea-template` |

The EventListener/Trigger object names retain repository terminology. The names do not change EventListener functionality; the repository endpoint used by the binding is the internal repository server.

### 17.3 Install Tekton Triggers

**Install Tekton Triggers in an Air-Gapped Environment**

In an air-gapped VKS environment, the cluster cannot access external Tekton release URLs or pull images directly from GHCR. To install Tekton Triggers, download the release manifests, copy the required images to the internal Harbor registry, update the image references, and apply the manifests to the cluster.

#### 17.3.1 Download the Tekton Triggers Release Manifest

On an internet-connected machine, create a directory for the release files:

```bash
mkdir -p /root/lab22/airgap-images/tekton-triggers
cd /root/lab22/airgap-images/tekton-triggers
```

Download the Tekton Triggers release manifest for version `v0.34.0`:

```bash
wget https://infra.tekton.dev/tekton-releases/triggers/previous/v0.34.0/release.yaml \
  -O tekton-trigger-release.yaml
```

This file contains the Kubernetes resources required to install Tekton Triggers.

#### 17.3.2 Download the Tekton Interceptors Manifest

Download the interceptor manifest for the same version:

```bash
wget https://infra.tekton.dev/tekton-releases/triggers/previous/v0.34.0/interceptors.yaml \
  -O tekton-interceptors-release.yaml
```

This file contains the resources required to configure Tekton interceptors.

> **Note:** Use the same version (`v0.34.0`) for both manifests.

#### 17.3.3 Prepare the Container Images for the Air-Gapped Installation

Copy the required Tekton Triggers images from GHCR to the internal Harbor registry using the internet-connected jumpbox.

##### Environment Details

| Item               | Value                            |
| ------------------ | -------------------------------- |
| Jumpbox            | `lab11-jumpbox`                  |
| Source registry    | `ghcr.io/tektoncd/triggers`      |
| Destination Harbor | `lab25-harbor.lab25.sunfire.lab` |
| Harbor project     | `tektoncd`                       |
| Version            | `v0.34.0`                        |
| Copy utility       | Skopeo                           |

##### 17.3.3.1 Step 1: Verify Skopeo

```bash
skopeo --version
```

Verified version in this environment: Skopeo `1.4.2-dev`.

##### 17.3.3.2 Step 2: Log In to Harbor

```bash
skopeo login harbor.sunfire.lab
```

Enter your Harbor credentials when prompted. Ensure that the `tektoncd` project exists and your account has permission to push images.

##### 17.3.3.3 Step 3: Copy the Images to Harbor

```bash
cd /root/lab22/airgap-images/tekton-triggers
```

**3.1 Interceptors**

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/triggers/interceptors-3176d6a3f314c3655b30bfd36e421dd5:v0.34.0 \
  docker://harbor.sunfire.lab/tektoncd/triggers/interceptors-3176d6a3f314c3655b30bfd36e421dd5:v0.34.0
```

**3.2 Controller**

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/triggers/controller-f656ca31de179ab913fa76abc255c315:v0.34.0 \
  docker://harbor.sunfire.lab/tektoncd/triggers/controller-f656ca31de179ab913fa76abc255c315:v0.34.0
```

**3.3 Webhook**

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/triggers/webhook-dd1edc925ee1772a9f76e2c1bc291ef6:v0.34.0 \
  docker://harbor.sunfire.lab/tektoncd/triggers/webhook-dd1edc925ee1772a9f76e2c1bc291ef6:v0.34.0
```

**3.4 EventListener Sink**

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/triggers/eventlistenersink-7ad1faa98cddbcb0c24990303b220bb8:v0.34.0 \
  docker://harbor.sunfire.lab/tektoncd/triggers/eventlistenersink-7ad1faa98cddbcb0c24990303b220bb8:v0.34.0
```

The `--all` option copies all platform variants when the source image is a multi-platform image.

> **Note:** Images were copied using the `v0.34.0` tags instead of the original manifest digests. Verify digest equivalence if image integrity must be guaranteed.

##### 17.3.3.5 Step 4: Verify the Image Tags

Run the following commands to verify that all four images are available in Harbor.

```bash
skopeo list-tags docker://harbor.sunfire.lab/tektoncd/triggers/interceptors-3176d6a3f314c3655b30bfd36e421dd5

skopeo list-tags docker://harbor.sunfire.lab/tektoncd/triggers/controller-f656ca31de179ab913fa76abc255c315

skopeo list-tags docker://harbor.sunfire.lab/tektoncd/triggers/webhook-dd1edc925ee1772a9f76e2c1bc291ef6

skopeo list-tags docker://harbor.sunfire.lab/tektoncd/triggers/eventlistenersink-7ad1faa98cddbcb0c24990303b220bb8
```

Expected output for each repository:

```json
{
  "Tags": [
    "v0.34.0"
  ]
}
```

Proceed only after confirming that the expected tag exists in all four repositories.

##### 17.3.3.6 Step 5: Update the Release Manifests

Update the image references in `tekton-trigger-release.yaml` and `tekton-interceptors-release.yaml` to use the corresponding Harbor images.

| Component | GHCR Image | Harbor Image |
|---|---|---|
| Controller | `ghcr.io/tektoncd/triggers/controller-f656ca31de179ab913fa76abc255c315:v0.34.0@sha256:472ed3311309a9ac066afd8be2ae435cb6cc5bc73240a5f82a9b04ee5c6f77eb` | `lab25-harbor.sunfire.lab/tektoncd/triggers/controller-f656ca31de179ab913fa76abc255c315:v0.34.0@sha256:472ed3311309a9ac066afd8be2ae435cb6cc5bc73240a5f82a9b04ee5c6f77eb` |
| Webhook | `ghcr.io/tektoncd/triggers/webhook-dd1edc925ee1772a9f76e2c1bc291ef6:v0.34.0@sha256:71ef9c830240870c76496f1858abc350d08be93365fc8bea17853aa94f64adcd` | `lab25-harbor.sunfire.lab/tektoncd/triggers/webhook-dd1edc925ee1772a9f76e2c1bc291ef6:v0.34.0@sha256:71ef9c830240870c76496f1858abc350d08be93365fc8bea17853aa94f64adcd` |
| EventListener Sink | `ghcr.io/tektoncd/triggers/eventlistenersink-7ad1faa98cddbcb0c24990303b220bb8:v0.34.0@sha256:6af463bd78b8ce864cc89f961a4c604db297c1a5c375dd038519483b3f85541d` | `lab25-harbor.sunfire.lab/tektoncd/triggers/eventlistenersink-7ad1faa98cddbcb0c24990303b220bb8:v0.34.0@sha256:6af463bd78b8ce864cc89f961a4c604db297c1a5c375dd038519483b3f85541d` |
| Interceptors | `ghcr.io/tektoncd/triggers/interceptors-3176d6a3f314c3655b30bfd36e421dd5:v0.34.0@sha256:b7858bddaa29c376c1f79f87c96ccdf9588f320a547950638f51e0c84303fd69` | `lab25-harbor.sunfire.lab/tektoncd/triggers/interceptors-3176d6a3f314c3655b30bfd36e421dd5:v0.34.0@sha256:b7858bddaa29c376c1f79f87c96ccdf9588f320a547950638f51e0c84303fd69` |

Replace the original image references with the corresponding Harbor references in both manifests. Preserve the version and digest for each image.

> **Note:** Ensure the Harbor images are available with the specified digests before applying the manifests.

##### 17.3.3.7 Step 6: Transfer the Updated Manifests

Transfer both updated YAML files to the machine used to administer the air-gapped VKS cluster.

Ensure that the cluster can access Harbor and has the required image-pull credentials, if authentication is enabled.

#### 17.3.4 Apply the Updated Manifests and Validate the Installation

#####  17.3.4.1 Step 1: Verify the Cluster Context

Confirm that `kubectl` is connected to the intended air-gapped VKS cluster.

```bash
kubectl config current-context
kubectl cluster-info
```

##### 17.3.4.2 Step 2: Apply the Tekton Triggers Manifest

```bash
kubectl apply -f tekton-trigger-release.yaml
```

##### 17.3.4.3 Step 3: Apply the Interceptors Manifest

```bash
kubectl apply -f tekton-interceptors-release.yaml
```

#### 17.3.5 Validation – Tekton Trigger

```bash
kubectl get all -n tekton-pipelines       #below deploymwnt should up and running
kubectl get crd | grep triggers.tekton.dev
```

Expected result:

```text
CRDs: clusterinterceptors, clustertriggerbindings, eventlisteners, interceptors,
      triggerbindings, triggers, triggertemplates (all .triggers.tekton.dev)

Pods (all 1/1 Running):
      tekton-triggers-controller-...
      tekton-triggers-core-interceptors-...
      tekton-triggers-webhook-...
```

### 17.4 Create the GitHub webhook secret

```
openssl rand -hex 32
```

"6addb375b90c7b68bbe9b1b5365ad6fbec8b1aedb2e87ca973849a5a589abb8c"      # Generate token from terminal using this command and use it while creating the gitea-webhook-secret

For example, suppose it returns:

YOUR_RANDOM_SECRET

File 'gitea-webhook-secret.yaml'

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/tekton-catalyst/gitea-webhook-secret.yaml

```
kubectl apply -f gitea-webhook-secret.yaml
```

### 17.5 EventListener ServiceAccount and RBAC

The EventListener runs with `serviceAccountName: repository-trigger-sa`. It needs permission to read the Triggers resources in `cicd`, read the cluster-scoped Triggers resources (`clusterinterceptors`, `clustertriggerbindings`), and create PipelineRuns.

#### 17.5.1 Create ServiceAccount rolebinding and RBAC

The EventListener needs permission to create PipelineRuns.

Service Account: 

File `gitea-trigger-sa.yaml`:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/tekton-catalyst/gitea-trigger-sa.yaml

| Important argument | Value |
|---|---|
| ServiceAccount | `gitea-trigger-sa` in `cicd` |

```bash
kubectl apply -f gitea-trigger-sa.yaml
```

#### 17.5.2 Role, RoleBinding, ClusterRole and ClusterRoleBinding

Create : eventlistener-gitea-rbac.yaml

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/tekton-catalyst/eventlistener-gitea-rbac.yaml

File `gitea-trigger-role.yaml` contains the Role (`repository-trigger-rol`), RoleBinding (`gitea-trigger-rolebinding.yaml`), ClusterRole (`gitea-trigger-cluster-role.yaml`) and ClusterRoleBinding (`gitea-trigger-cluster-rolebinding.yaml`).

| Important argument | Meaning |
|---|---|
| Role (namespace `cicd`) | Read `triggerbindings`, `triggertemplates`, `eventlisteners`; create `pipelineruns` |
| ClusterRole | Read `clusterinterceptors`, `clustertriggerbindings` |
| Subject | ServiceAccount `gitea-trigger-sa` |

```bash
kubectl apply -f eventlistener-gitea-rbac.yaml
```

#### 17.5.3 Validation – RBAC

```bash
kubectl get sa gitea-trigger-sa -n cicd
kubectl get role,rolebinding -n cicd | grep gitea-trigger
kubectl get clusterrole,clusterrolebinding | grep gitea-trigger

kubectl auth can-i list triggertemplates.triggers.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa \
  -n cicd

kubectl auth can-i list triggerbindings.triggers.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa \
  -n cicd

kubectl auth can-i list eventlisteners.triggers.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa \
  -n cicd

kubectl auth can-i list clusterinterceptors.triggers.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa -n cicd

kubectl auth can-i list clustertriggerbindings.triggers.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa -n cicd

kubectl auth can-i create pipelineruns.tekton.dev \
  --as=system:serviceaccount:cicd:gitea-trigger-sa \
  -n cicd
```

Expected result: all objects exist and every `can-i` answers `yes`.

### 17.6 TriggerBinding, TriggerTemplate and EventListener

Create the three objects below, then validate them once at the end.

#### 17.6.1 TriggerBinding

File `travelportal-gitea-binding.yaml`:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/tekton-catalyst/travelportal-gitea-binding.yaml

| Important argument | Value |
|---|---|
| Name | `travelportal-gitea-binding` |
| `REPO_URL` | `http://10.12.90.62/admin/travelPortal-test-buildpack.git` |
| `REVISION` | `main` |
| Payload fields read | `body.ref`, `body.after`, `body.head_commit.message` |

```bash
kubectl apply -f travelportal-gitea-binding.yaml
```

#### 17.6.2 TriggerTemplate

File `travelportal-gitea-template.yaml`:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/airgap-manifest/travelportal-gitea-template-airgap.yaml

| Important argument | Value |
|---|---|
| Name | `travelportal-gitea-template` |
| `pipelineRef` | `travelportal-pipeline-values-update` |
| `serviceAccountName` | `tekton-pipeline-sa` |
| Repository URL | `http://10.12.90.62/admin/travelPortal-test-buildpack.git` |
| PipelineRun name | `travelportal-build-xxxxx` (generated) |
| Workspaces | Per-run `source` and `cache` PVCs from StorageClass `lab-gold-storage-policy` |

Note: Update the BUILDER_IMAGE value to internal facing registry as shown below (We have already placed this image in buildpack phase)

```yaml
  - name: BUILDER_IMAGE
    value: lab25-harbor.lab25.sunfire.lab/library/paketobuildpacks/builder-jammy-base@sha256:7f09f46d782798c5a11375d3ed6c4ef41d61c88766cbe008d336c282606860ad
```

```bash
kubectl apply -f travelportal-gitea-template.yaml
```

#### 17.6.3 EventListener

File `travelportal-gitea-listener.yaml`:

YAML: https://github.com/kondurupurandhar/keda-vks/blob/main/manifest/tekton-catalyst/travelportal-gitea-listener.yaml

| Important argument | Value |
|---|---|
| Name | `travelportal-gitea-listener` |
| `serviceAccountName` | `gitea-trigger-sa` |
| Interceptor | Cluster-scoped `cel` interceptor (main branch only, ignores CI commits) |
| Binding / Template | `travelportal-gitea-binding` / `travelportal-gitea-template` |
| `serviceType` | `NodePort` |
| Application port | `8080` (observed NodePort in this lab: `31877`) |

```bash
kubectl apply -f travelportal-gitea-listener.yaml
```

Because the manifest specifies only `serviceType: NodePort`, Kubernetes may assign a different NodePort in another environment, so always confirm the actual value in the validation below.

#### 17.6.4 Validation – TriggerBinding, TriggerTemplate and EventListener

Check the objects, the CEL interceptor, and the generated Service:

```bash
kubectl get triggerbinding -n cicd
kubectl get triggertemplate -n cicd
kubectl get eventlistener -n cicd
kubectl get clusterinterceptor cel
kubectl get svc,endpoints -n cicd
kubectl get pods -n cicd | grep -i listener
```

Expected result:

```text
EventListener   → AVAILABLE=True, READY=True
CEL interceptor → present
Service         → 8080:31877/TCP
Endpoint        → <pod-ip>:8080
```

### 17.7 Expose the EventListener externally and configure the Gitea webhook.

First check:

```
kubectl get svc -n cicd | grep travelportal-gitea-listener

kubectl get eventlistener travelportal-gitea-listener -n cicd

kubectl get pods -n cicd | grep travelportal-gitea-listener
```

If the EventListener is already externally reachable, configure the GitHub repository webhook with:

- Payload URL: your external EventListener URL
- Content type: application/json
- Secret: the value stored in github-webhook-secret
- Events: Just the push event
- Active: enabled

Then make a small commit/push to main.

1. Change the Service to NodePort

```


kubectl patch eventlistener travelportal-gitea-listener -n cicd \
  --type=merge \
  -p '{"spec":{"resources":{"kubernetesResource":{"serviceType":"NodePort"}}}}'
```

Then check:

```
kubectl get svc el-travelportal-gitea-listener -n cicd
```

Expected:

```
NAME                              TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)
el-travelportal-gitea-listener   NodePort   198.54.22.125   <none>        8080:31234/TCP,9000:3xxxx/TCP
```

Url of eventlistner 

```
 http://<NODE-IP>:<NODEPORT>
```

### 17.8 Add the webhook in the repository

Repository:

```text
http://10.12.90.62/admin/TravelPortal-lab10.git

```

Open:

```text
Repository
  -> Settings
  -> Webhooks
  -> Add Webhook
```

Use:

| Setting | Value |
|---|---|
| Target URL | `http://<NODE-IP>:<NODEPORT>` |
| Method | `POST` |
| Content type | `application/json` |
| Event | Push events |
| Branch | `main` |
| Secrets | `gitea-webhook-secret` |

Save the webhook.

Note: Make sure your git repository is reachable to target URL

Click the created webhook, navigate to **Recent Deliveries**, and click **Test Push Event** to verify that the webhook is working correctly and triggering the Tekton pipeline.

### 17.9 Pipeline changes for automatic runs

Push the changes to the source repository. This will automatically trigger the pipeline.

```
git add .

git commit -m "Your commit message"

git push origin <branch-name>
``` 

#### 17.9.1 Repository credentials for update-values

The clone and `update-values` Tasks read the repository credentials from the `username` and `token` keys of the `repo-git-credentials` Secret (created in Section 14.3.1). The token is a Personal Access Token created in the repository server.

## 18. Final Validation – End to End

Then watch the automation:

```bash
kubectl logs -f deployment/el-travelportal-gitea-listener -n cicd
kubectl get pipelineruns -n cicd -w
```

Expected result:

```text
Webhook delivery   → HTTP 202 Accepted
New PipelineRun    → travelportal-build-xxxxx
Build              → BuildKit if the repo has a Dockerfile, otherwise Buildpacks
Pipeline           → clone → detect → build → sign → update-values all succeed
Self-trigger       → no second PipelineRun after the CI commit
Harbor             → image digest signed, SBOM attestation present
```

## 19. Conclusion

This implementation provides a Kubernetes-native CI workflow on VKS using Tekton, BuildKit, Buildpacks, Syft, Cosign, and Harbor. The pipeline automatically selects BuildKit or Buildpacks based on the application source, generates an SBOM, signs the immutable image digest, and stores the resulting image and security artifacts in Harbor.

The architecture also separates the application source and internal CI/GitOps responsibilities across two repositories, providing a clear and controlled workflow for source management and deployment configuration.

Overall, the solution provides an automated, traceable, and security-focused CI process that can be operated in both connected and disconnected VKS environments, provided the required images, dependencies, credentials, and registry artifacts are available locally.
