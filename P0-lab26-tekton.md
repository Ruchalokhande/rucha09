# Tekton Installation and Integration with Buildpacks, BuildKit, and Cosign on VKS

## 1. Introduction

- **VKS (vSphere Kubernetes Service):** A Kubernetes service provided by VMware Cloud Foundation for creating and managing Kubernetes clusters on vSphere infrastructure.
- **Tekton:** A Kubernetes-native CI/CD framework used to create and run automated pipelines as Kubernetes resources.
- **Cloud Native Buildpacks (CNB):** A technology that converts application source code into container images without requiring the developer to write a Dockerfile.
- **BuildKit:** A modern container image build engine used to build images from Dockerfiles efficiently and supports advanced build features such as caching and parallel builds.
- **Cosign:** A tool used to digitally sign and verify container images and other OCI artifacts. It helps confirm that an image came from a trusted source and has not been tampered with.
- **SBOM (Software Bill of Materials):** A detailed list of the software components, libraries, and dependencies contained in a container image. It helps with vulnerability tracking and software supply-chain security.
- **Harbor:** A private container registry used to store and manage the container images produced by the Tekton pipeline.
- **Tekton Triggers:** A Tekton component that listens for events (such as a repository webhook) and automatically creates PipelineRuns. It is made of an EventListener, a TriggerBinding and a TriggerTemplate.
- **Webhook:** An HTTP call the repository sends to the Tekton EventListener every time someone pushes code.
- **External DNS:** A Kubernetes component that automatically creates DNS records (here in Cloudflare) for exposed services, so the Tekton Dashboard is reached by a DNS name.
- **Istio + Kubernetes Gateway API:** Provides the HTTPS entry point (a Gateway and an HTTPRoute) used to reach the Tekton Dashboard.
- **cert-manager:** Issues the TLS certificate for the Dashboard hostname from the letsencrypt-prod ClusterIssuer.

### 1.1 Operational ownership

| Role| Ownership of services |
|---|---|
| VKS administrator| Supervisor cluster, Supervisor namespace, ArgoCD service |
| Platform/DevOps Engineer| Guest cluster & namespaces, Tekton Dashboard, Pipeline, Tasks, Auto-Trigger, Manifest repository, cert-manager, istio, externalDNS |
| Application Developer | Source code repository |

## 2. Architecture

### 2.1 Architecture Diagram and Flow

All of this — Tekton, BuildKit, Buildpacks, and Cosign — runs inside the VKS cluster itself. Nothing extra needs to be installed outside it. Everything after the developer's push happens automatically, with no manual steps.

```text
  Developer                              User (browser)
      │ git push origin main                  │ https://tekton.k8slabs.com    ***review comment*** { https://tekton.labX.k8slabs.com } 
      ▼                                       ▼
  Git SourceCode Repository              Cloudflare DNS
  (TravelPortal-labx)
      │ webhook (POST)                        │ resolves to the Dashboard
      ▼                                       ▼
┌─────────────────────────────── VKS cluster ────────────────────────────────┐
│                                                                            │
│  ┌─ namespace: tekton-pipelines ────────────────────────────────────────┐  │
│  │ Tekton Pipelines     Tekton Triggers     Tekton Dashboard  ◄── User  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─ namespace: tanzu-system-service-discovery ──────────────────────────┐  │
│  │ External DNS ─► creates the Dashboard DNS record in Cloudflare       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─ namespace: cicd ────────────────────────────────────────────────────┐  │
│  │ EventListener ─► CEL filter ─► TriggerBinding / TriggerTemplate      │  │
│  │                       │ creates                                      │  │
│  │                       ▼                                              │  │
│  │ PipelineRun  (travelportal-pipeline-values-update)                   │  │
│  │  1. clone         git-clone-update                                   │  │
│  │  2. detect        detect-build-type (Dockerfile present?)            │  │
│  │         ┌──── yes ────┴──── no ────┐                                 │  │
│  │         ▼                          ▼                                 │  │
│  │     BuildKit                  Buildpacks                             │  │
│  │         └────────────┬─────────────┘                                 │  │
│  │                      ▼                                               │  │
│  │  3. sign          Syft SBOM ─► Cosign sign + attest (image digest)   │  │
│  │  4. update-values commit image digest to helm-charts/values.yaml     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ArgoCD ─► watches the Git repository ─► deploys the signed image to VKS   │
└────────────────────────────────────────────────────────────────────────────┘
      │ image + signature + SBOM            │ values.yaml commit (GitOps)
      ▼                                     ▼
   Harbor                             Git Manifest Repository(k8s-manifest-labx)
```

| Step | What it means in plain words |
|---|---|
| Developer pushes code | The only thing a person actually has to do |
| Webhook | The repository's call to Tekton on every push |
| EventListener | Receives the webhook, filters it (main branch only, ignores the pipeline's own commits) and creates the PipelineRun |
| Self-trigger filter | Stops the values.yaml commit from the last step from starting the pipeline again |
| Tekton | The automation engine — runs the whole pipeline |
| Dockerfile check | The one decision point: which tool builds the image |
| BuildKit / Buildpacks | The two ways an image actually gets built (Dockerfile present → BuildKit, absent → Buildpacks) |
| SBOM (ingredients list) | A record of everything that went into the image, for security and audit |
| Cosign signing | Proves the image really came from our pipeline and hasn't been tampered with |
| Harbor | Where images and OCI security artifacts are stored and scanned; signature enforcement depends on the configured security policy |
| update-values (GitOps) | Writes the new image digest into the Helm values file in Git |
| ArgoCD | Takes the newly signed image and rolls it out to the running application |
| External DNS + Cloudflare | Publishes the DNS name used to open the Tekton Dashboard |

### 2.2 Namespaces

| Namespace | Contents |
|---|---|
| `tekton-pipelines` | Tekton Pipelines, Tekton Triggers, Tekton Dashboard, Gateway, Certificate, HTTPRoute |
| `cicd` | Application Tasks, Pipelines, PipelineRuns and the Triggers objects (Sections 16 to 20) |
| `tanzu-system-service-discovery` | External DNS |
| `istio-system`, `istio-ingress` | Istio |
| `cert-manager` | cert-manager |

## 3. Why use BuildKit and Buildpacks with Tekton on VKS?

In plain terms, here's why this combination makes sense:

**One automatic process for everyone.** Whether or not a team wrote a Dockerfile, their code still gets built the same reliable way — Tekton decides which tool to use, so nobody has to remember or configure it manually.

**Nothing gets built by hand.** A person pushing code is the only manual step. Everything after that — building, checking for security issues, signing, publishing — happens on its own.

**Trust is built in, not bolted on.** Every image gets signed and gets an ingredients list (SBOM) the moment it's built — not added later as an afterthought. Harbor can be configured to enforce signature or artifact-security policies before images are promoted or deployed.

**It all runs on the same platform.** VKS already gives us the cluster, the storage, the networking, Harbor (image store), and ArgoCD (deployment) — Tekton just plugs into what's already there instead of needing its own separate servers.

**Works with no internet too.** Everything described here — BuildKit, Buildpacks, Tekton, and Cosign — can run fully offline, which matters for secure/air-gapped environments.

**One dashboard to watch it all.** Tekton's Dashboard shows every build, whether it succeeded or failed, and why — in a web page, not just log files.

**A push starts the build by itself.** The repository sends a webhook to Tekton Triggers, which creates the PipelineRun. Nobody has to start it manually, and commits made by the pipeline itself never start a second build.

## 4. Benefits

| Benefit | In simple words |
|---|---|
| No guesswork on how to build | Tekton always knows: Dockerfile present → BuildKit, absent → Buildpacks. |
| Less work for app teams | Teams don't need to write or maintain a Dockerfile if they don't want to — Buildpacks handles it for them. |
| Fully automatic | A code push is the only human action; the build, sign, and publish steps run by themselves. |
| Every image is signed | Cosign signs the image and its ingredients list (SBOM) right after it's built, so nothing unverified reaches production. |
| Safer builds | BuildKit and Buildpacks both run without needing root/admin access inside the cluster. |
| One place for images | Every built image lands in Harbor, where it's scanned, verified, and stored. |
| Same steps everywhere | The same pipeline can be used offline after all required artifacts and dependencies have been mirrored into the environment. |
| You can always prove what's running | Images are tracked by a fixed ID (a digest), not a name that can change, and each one has a matching signed ingredients list. |
| One dashboard for visibility | Anyone can open the Tekton Dashboard by its DNS name and see the status of every build, without needing cluster access. |
| Builds start on their own | A push to the main branch sends a webhook to Tekton Triggers, which creates the PipelineRun — no manual PipelineRun needed. |
| No endless build loops | The commit the pipeline makes to update values.yaml is filtered out, so it does not trigger another build. |

## 5. Prerequisites

### 5.1 Platform and components ***review comment: { a new version column is added } 

<table>
  <thead>
    <tr>
      <th rowspan="2">Component</th>
      <th colspan="2">Version</th>
      <th rowspan="2">Requirement</th>
    </tr>
    <tr>
      <th>VCF 9.0</th>
      <th>VCF 9.1.x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VMware Cloud Foundation (VCF)</td>
      <td>9.0.1.0</td>
      <td>9.1.1.0</td>
      <td>VCF 9.x environment available</td>
    </tr>
    <tr>
      <td>VMware Kubernetes Service (VKS)</td>
      <td>3.6.0+v1.35</td>
      <td>3.8.0+v1.37</td>
      <td>Target VKS workload cluster available</td>
    </tr>
    <tr>
      <td>Kubernetes (SUP)</td>
      <td>TBD</td>
      <td>v1.34.9</td>
      <td>Supervisor Kubernetes runtime version</td>
    </tr>
    <tr>
      <td>Kubernetes (GUEST)</td>
      <td>v1.35.5</td>
      <td>v1.37.1</td>
      <td>Kubernetes runtime for the VKS workload cluster</td>
    </tr>
    <tr>
      <td>kubectl</td>
      <td>v1.30.10</td>
      <td>v1.32.9</td>
      <td>Kubernetes CLI installed and configured to access the target cluster</td>
    </tr>
    <tr>
      <td>Harbor</td>
      <td>v2.15.2</td>
      <td>v2.15.2</td>
      <td>Harbor registry available as a Supervisor service</td>
    </tr>
    <tr>
      <td>vks-standard-packages</td>
      <td>3.6.0-20260211</td>
      <td>TBD</td>
      <td>VKS standard packages available for add-on installation</td>
    </tr>
    <tr>
      <td>cert-manager</td>
      <td>1.19.2+vmware.1-vks.1</td>
      <td>1.21.2+vmware.1-vks.1</td>
      <td>Installed through VKS add-on package management</td>
    </tr>
    <tr>
      <td>Istio</td>
      <td>1.28.2+vmware.1-vks.1</td>
      <td>1.31.0+vmware.1-vks.1</td>
      <td>Installed through VKS add-on package management</td>
    </tr>
    <tr>
      <td>ClusterIssuer</td>
      <td>Let&#x27;s Encrypt (ACME)</td>
      <td>Let&#x27;s Encrypt (ACME)</td>
      <td>cert-manager configured with a Let&#x27;s Encrypt ClusterIssuer</td>
    </tr>
    <tr>
      <td>ExternalDNS</td>
      <td>0.20.0+vmware.1-vks.1</td>
      <td>0.22.0+vmware.1-vks.1</td>
      <td>Installed through VKS add-on package management</td>
    </tr>
    <tr>
      <td>DNS</td>
      <td>Cloudflare</td>
      <td>Cloudflare</td>
      <td>Cloudflare DNS zone and API credentials configured</td>
    </tr>
    <tr>
      <td>Tekton Pipelines</td>
      <td>v1.17.0</td>
      <td>v1.17.0</td>
      <td>Installed and configured in the target Kubernetes cluster</td>
    </tr>
    <tr>
      <td>Tekton Dashboard</td>
      <td>v0.73.0</td>
      <td>v0.73.0</td>
      <td>Tekton Dashboard installed and accessible</td>
    </tr>
    <tr>
      <td>Tekton Triggers</td>
      <td>v0.34.0</td>
      <td>v0.34.0</td>
      <td>Tekton Triggers installed with compatible release manifests</td>
    </tr>
    <tr>
      <td>Cosign</td>
      <td>v3.1.3</td>
      <td>v3.1.3</td>
      <td>Installed for container image signing and verification</td>
    </tr>
    <tr>
      <td>Syft</td>
      <td>v1.55.0</td>
      <td>v1.57.1</td>
      <td>Installed for Software Bill of Materials (SBOM) generation</td>
    </tr>
    <tr>
      <td>Argo CD (Supervisor Service)</td>
      <td>1.1.0-25100889 (v3.0.19)</td>
      <td>TBD</td>
      <td>Argo CD Supervisor service available</td>
    </tr>
    <tr>
      <td>Argo CD CLI</td>
      <td>v3.0.19</td>
      <td>TBD</td>
      <td>Installed and configured to access the Argo CD instance</td>
    </tr>
  </tbody>
</table>


Dashboard DNS name: `tekton.k8slabs.com` is the actual hostname used in this document. If several VKS labs share the `k8slabs.com` zone, confirm the name does not collide with another lab (TO VERIFY).
***review comment*** {remove (TO VERIFY) from above line. and decide whether the DNS name in entire document should be tekton.k8slabs.com or tekton.labX.k8slabs.com}

### 5.2 Required CLI tools

```bash
vcf version
kubectl version --client
git --version
curl --version
```

## 6. Manifest Repository

The YAML manifests are not pasted inline. They are stored in the repository below. Each section names the exact file, lists its important arguments, and links to the YAML.

### 6.1 Repository Locations

| Item | Location |
|---|---|
| Tekton pipeline, External DNS and Triggers manifests | https://github.com/kondurupurandhar/ci-cd-stack-on-vks/tree/main/manifest/tekton-catalyst |
| Smoke test Task / TaskRun manifests | https://github.com/kondurupurandhar/ci-cd-stack-on-vks/tree/main/manifest/test-task |

**Note:** The ci-cd-stack-on-vks repository is private. You need access before the links open or the clone works.

## 7. Cluster Pre-checks

### 7.1 Access the VKS cluster

```bash
vcf context list
vcf context use <TARGET_VKS_CLUSTER_NAME>
```

### 7.2 Validate cluster and permissions

```bash
kubectl config current-context        # confirm the right cluster
kubectl cluster-info
kubectl version
kubectl get nodes -o wide             # all nodes Ready

kubectl auth can-i create customresourcedefinitions   # expected: yes
kubectl auth can-i create clusterroles                # expected: yes
kubectl auth can-i create namespaces                  # expected: yes
```

### 7.3 Internet connectivity

A successful ping is not required and does not prove HTTPS access. Test the actual endpoints.

From your workstation:

```bash
curl -I https://infra.tekton.dev
curl -I https://registry-1.docker.io/v2/
curl -I https://ghcr.io/v2/
```

### 7.4 Install cert-manager

Install cert-manager Using VKS Package Management

#### 7.4.1 Retrieve details about the available cert-manager package:

```
vcf package available get \
  cert-manager.kubernetes.vmware.com \
  -n vmware-system-tkg
```

#### 7.4.2 Set the Package Version

The following version is from the source runbook. Confirm its availability in the target repository.
```bash
export CERT_MANAGER_VERSION=1.19.2+vmware.1-vks.1
***review comment***{ export CERT_MANAGER_VERSION="1.21.2+vmware.1-vks.1" }
export CERT_MANAGER_RELEASE_NAME="cert-manager.kubernetes.vmware.com"
```

#### 7.4.3 Generate and Review the Data Values

```bash
vcf package available get \
  cert-manager.kubernetes.vmware.com/${CERT_MANAGER_VERSION} \
  -n vmware-system-tkg \
  --default-values-file-output cert-manager-data-values.yaml
```
verify: cert-manager-data-values.yaml
```
cat cert-manager-data-values.yaml
```

#### 7.4.4 Perform a Dry Run

```bash
vcf package install cert-manager \
  --package cert-manager.kubernetes.vmware.com \
  --version "${CERT_MANAGER_VERSION}" \
  -n vmware-system-tkg \
  --values-file cert-manager-data-values.yaml \
  --dry-run
```

#### 7.4.5 Install cert-manager

```bash
vcf package install cert-manager \
  --package cert-manager.kubernetes.vmware.com \
  --version "${CERT_MANAGER_VERSION}" \
  -n vmware-system-tkg \
  --values-file cert-manager-data-values.yaml \
  --wait
```

#### 7.4.6 Verify Installation

```bash
kubectl get packageinstall -n vmware-system-tkg
kubectl get namespaces
kubectl get all -n cert-manager
```

### 7.5 Install Istio Using VKS Package Management

#### 7.5.1 Check available releases

```bash
vcf package available list -A

vcf package available get \
  istio.kubernetes.vmware.com \
  -n vmware-system-tkg
```

#### 7.5.2 Set the Istio version

```
export ISTIO_VERSION=1.28.2+vmware.1-vks.1
***review comment***{ export ISTIO_VERSION="1.31.0+vmware.1-vks.1" }
export ISTIO_RELEASE_NAME="istio.kubernetes.vmware.com"
```

#### 7.5.4 Generate the data values file

```bash
vcf package available get \
  istio.kubernetes.vmware.com/${ISTIO_VERSION} \
  -n vmware-system-tkg \
  --default-values-file-output istio-data-values.yaml
cat istio-data-values.yaml
```

#### 7.5.5 Enable the ingress gateway

Edit `istio-data-values.yaml` to include the following configuration while retaining other required package settings:

```yaml
istio:
  gateways:
    ingress:
      enabled: true
```

Review the updated file:

```bash
cat istio-data-values.yaml
```

#### 7.5.6 Perform a dry run

```bash
vcf package install istio \
  --package istio.kubernetes.vmware.com \
  --version "${ISTIO_VERSION}" \
  -n vmware-system-tkg \
  --values-file istio-data-values.yaml \
  --dry-run
```

#### 7.5.7 Install Istio

```bash
vcf package install istio \
  --package istio.kubernetes.vmware.com \
  --version "${ISTIO_VERSION}" \
  -n vmware-system-tkg \
  --values-file istio-data-values.yaml \
  --wait
```

#### 7.5.8 Verify installation

```bash
kubectl get packageinstall istio \
  -n vmware-system-tkg

kubectl get namespaces
kubectl get all -n istio-ingress
kubectl get all -n istio-system
```

### 7.6 Confirm Istio and cert-manager are ready

```bash
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n istio-ingress
```
Expected result: all Pods `Running.


### 7.7 External DNS

#### Section 7.7.1 Check available releases

```bash
vcf package available list -A

vcf package available get \
  external-dns.kubernetes.vmware.com \
  -n vmware-system-tkg
````

#### 7.7.2 Create Namespace
Create the tanzu-system-service-discovery namespace to deploy, manage External DNS and cloudflare-secret.
```
kubectl create namespace tanzu-system-service-discovery
```

#### 7.7.3 Add label to tanzu-system-service-discovery

```
kubectl label namespace tanzu-system-service-discovery  pod-security.kubernetes.io/enforce=privileged --overwrite
```

#### 7.7.4 Cloudflare API token

File `cloudflare-secret.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/cloudflare-secret.yaml

#### 7.7.5 Create Cloudflare Secret

```bash 
kubectl apply -f cloudflare-secret.yaml

```

#### 7.7.6 Install External DNS
 
##### 7.7.6.1 Define the ExternalDNS Version Variable
 
```bash
EXTERNAL_DNS_VERSION="0.20.0+vmware.1-vks.1"
***review comment***{ EXTERNAL_DNS_VERSION="0.22.0+vmware.1-vks.1" }
EXTERNAL_DNS_PACKAGE="external-dns.kubernetes.vmware.com"
```
 
##### 7.7.6.2 Get Package Details in YAML Format
 
```bash
vcf package available get \
  external-dns.kubernetes.vmware.com \
  -n vmware-system-tkg \
  -o yaml
```
 
##### 7.7.6.3 Export Default Values to the Values File
 
```bash
vcf package available get \
  "external-dns.kubernetes.vmware.com/$EXTERNAL_DNS_VERSION" \
  -n vmware-system-tkg \
  --default-values-file-output external-dns-values.yaml
```
##### 7.7.6.4 Configure external-dns-values.yaml

File `external-dns-values.yaml`:
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/external-dns-values.yaml

##### 7.7.6.5 Perform dry-run
```
vcf package install external-dns \
  -p external-dns.kubernetes.vmware.com \
  -v "$EXTERNAL_DNS_VERSION" \
  -n vmware-system-tkg \
  --values-file external-dns-values.yaml \
  --dry-run 
```

##### 7.7.6.6 Install the ExternalDNS Package
 
```bash
vcf package install external-dns \
  -p external-dns.kubernetes.vmware.com \
  -v "$EXTERNAL_DNS_VERSION" \
  -n vmware-system-tkg \
  --values-file external-dns-values.yaml \
  --wait
```

#### 7.7.7 Create the External DNS ClusterRole and ClusterRoleBinding

**Create the External DNS ClusterRole**

The ClusterRole provides External DNS with read-only permissions to discover Kubernetes resources such as Services, Ingresses, Gateways, and HTTPRoutes.

FILE: `external-dns-cluster-roles.yaml`

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/external-dns-cluster-roles.yaml

**Create the External DNS ClusterRoleBinding**

The ClusterRoleBinding assigns these permissions to the External DNS ServiceAccount so it can monitor these resources and manage the corresponding DNS records.

FILE: `external-dns-clusterrolebinding.yaml`

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/external-dns-clusterrolebinding.yaml

```bash
kubectl apply -f  external-dns-cluster-roles.yaml
kubectl apply -f external-dns-clusterrolebinding.yaml
```

#### 7.7.8 Restart external-dns and Monitor Logs

```bash
kubectl rollout restart deployment external-dns -n tanzu-system-service-discovery
kubectl rollout status deployment external-dns -n tanzu-system-service-discovery
kubectl get pods -n tanzu-system-service-discovery
kubectl logs -n tanzu-system-service-discovery  -l app=external-dns -f
***review comment*** { Press Ctrl+C to exit the logs view }
```

#### 7.7.9 Confirm HTTPRoute permissions

```bash
kubectl auth can-i get httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
kubectl auth can-i list httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
kubectl auth can-i watch httproutes.gateway.networking.k8s.io \
  --as=system:serviceaccount:tanzu-system-service-discovery:external-dns
```

Expected result: `yes` for all three.

#### 7.7.10 Validation: External DNS

```bash
kubectl get pods -n tanzu-system-service-discovery
kubectl get packageinstall -n vmware-system-tkg      # run in the Supervisor context
kubectl logs -n tanzu-system-service-discovery  -l app=external-dns -f
***review comment*** { Press Ctrl+C to exit the logs view }
```

Expected result: the Pod is `Running` and the logs show no errors. If the label returns nothing, check the real labels with `kubectl get pods -n tanzu-system-service-discovery --show-labels`. The old document used `app.kubernetes.io/name=external-dns` and the Istio runbook used `app=external-dns` (TO VERIFY).

***review comment*** { remove (TO VERIFY) from above section}

## 8. Install Tekton Pipelines

Create namespace for tekton pipline

```bash
kubectl create namespace tekton-pipelines
```

Add the label to namespace

```bash
kubectl label namespace tekton-pipelines pod-security.kubernetes.io/enforce=privileged --overwrite
```

verify tekton-pipelines namespace

```bash
kubectl get namespace tekton-pipelines --show-labels
kubectl get namespace tekton-pipelines -o jsonpath='{.metadata.labels.pod-security\.kubernetes\.io/enforce}{"\n"}'
```

Expected output for the last command

```text
privileged
```

### 8.1 Install tekton pipeline

File: tekton-pipeline-release.yaml

```bash
kubectl apply --filename https://infra.tekton.dev/tekton-releases/pipeline/latest/release.yaml
or
kubectl apply --filename https://infra.tekton.dev/tekton-releases/pipeline/previous/v1.17.0/release.yaml
```
Note: We have used tekton pipeline "v1.17.0" version
***review comment*** {
Note: During testing of this deployment we have used tekton pipeline "v1.17.0" version. For other versions results may vary.
}
Many resources are created (CRDs, the `tekton-pipelines` namespace, service accounts, cluster roles, controller and webhook deployments). Wait until the Pods are ready.

### 8.2 Validation: Tekton Pipelines

```bash
kubectl get pods -n tekton-pipelines
kubectl get crd | grep tekton
kubectl api-resources | grep tekton
kubectl logs deployment/tekton-pipelines-controller -n tekton-pipelines --tail=50
kubectl logs -n tanzu-system-service-discovery -l app=external-dns --tail=50

```

```text
Namespace tekton-pipelines    → Active
tekton-pipelines-controller   → 1/1 Running
tekton-pipelines-webhook      → 1/1 Running
CRDs                          → tasks, taskruns, pipelines, pipelineruns (tekton.dev) present
Controller logs               → no ERROR / failed / panic / connection refused
```

## 9. Install Tekton Dashboard

File: tekton-dashboard-release.yaml

```bash
kubectl apply -f https://infra.tekton.dev/tekton-releases/dashboard/latest/release.yaml
or                                       
kubectl apply --filename https://infra.tekton.dev/tekton-releases/dashboard/previous/v0.73.0/release.yaml
```

Note: We have used tekton dashboard "v0.73.0" version

The Dashboard is installed into `tekton-pipelines`. It stays reachable only inside the cluster until Section 7.7 and Sections 10 to 12 are complete.

### 9.1 Validation: Tekton Dashboard

```bash
kubectl get pods,deployment,svc -n tekton-pipelines | grep dashboard
```

Expected result: `tekton-dashboard` Pod and Deployment `1/1 Running`; Service `tekton-dashboard` on port `9097`.

### 9.2 Cloudflare API token for cert-manager

cert-manager uses a Cloudflare API token to complete DNS-01 challenges and verify domain ownership when requesting TLS certificates from Let's Encrypt.

 **Update the DNS Name Eg. tekton.labX.k8slabs.com  and remove this comment from this file**
 **Make sure no white spaces in front of the placeholder values replaced in the file**
 
File: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/cloudflare-secret-cert-manager.yaml
 
Apply: 
 ```
kubectl apply -f cloudflare-secret-cert-manager.yaml

 ```
#### 9.2.1 Validate that the Secret


```
kubectl get secret external-dns-cloudflare -n cert-manager
```

### 9.3 Create a Let's Encrypt ClusterIssuer

File `cluster-issuer.yaml`:

| Important argument | Value |
|---|---|
| ClusterIssuer name | `letsencrypt-prod` |
| email | `<ACME_CONTACT_EMAIL>` (replace with the environment's contact email) |
| server | `https://acme-v02.api.letsencrypt.org/directory` |
| privateKeySecretRef | `letsencrypt-prod-account-key` |
| Solver | `dns01`, Cloudflare |
| apiTokenSecretRef | Secret `external-dns-cloudflare`, key `api-token` (must be available to cert-manager) |

File: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/cluster-issuer.yaml

Apply the ClusterIssuer:

```bash
kubectl apply -f cluster-issuer.yaml
```

Check its status:

```bash
kubectl get clusterissuer letsencrypt-prod
```

 
## 10. Istio Integration: Expose the Dashboard

All three resources are created in `tekton-pipelines`, the namespace of the `tekton-dashboard` Service, so no cross-namespace reference is needed. Apply them in this order: Certificate, Gateway, HTTPRoute. All files are in `manifest/tekton-catalyst`.

### 10.1 TLS certificate (cert-manager)

File `tekton-dashboard-certificate.yaml`

***review comment*** { # Update the DNS Name Eg. tekton.labX.k8slabs.com  and remove this comment from this file }
***review comment*** { Make sure no white spaces in front of the placeholder values replaced in the file }
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/tekton-dashboard-certificate.yaml

| Important argument | Value |
|---|---|
| issuerRef | ClusterIssuer `letsencrypt-prod` |
| dnsNames | `tekton.k8slabs.com` |              ***review comment*** {tekton.labX.k8slabs.com}
| secretName | `tekton-dashboard-tls-cert` (created by cert-manager, used by the Gateway) |

```bash
kubectl apply -f tekton-dashboard-certificate.yaml
kubectl get certificate -n tekton-pipelines     # wait for READY=True  ***review comment*** { if the wait time is more than 2 mins, then recheck the above steps execution }
```

### 10.2 tekton-dashboard-gateway

File `tekton-dashboard-gateway.yaml`:

***review comment*** { # Update the DNS Name Eg. tekton.labX.k8slabs.com  and remove this comment from this file }
***review comment*** { Make sure no white spaces in front of the placeholder values replaced in the file }
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/tekton-dashboard-gateway.yaml

| Important argument | Value |
|---|---|
| gatewayClassName | `istio` |
| Listener | HTTPS, port 443, `tls.mode: Terminate` |
| hostname | `tekton.k8slabs.com` |            ***review comment*** {tekton.labX.k8slabs.com}
| certificateRefs | Secret `tekton-dashboard-tls-cert` (from Section 10.1) |
| allowedRoutes | Same namespace only |


```bash
kubectl get namespace tekton-pipelines --show-labels   # it should be "enforce=privileged" if not "kubectl label namespace tekton-pipelines pod-security.kubernetes.io/enforce=privileged --overwrite"
kubectl apply -f tekton-dashboard-gateway.yaml
kubectl get gateway tekton-dashboard-gateway -n tekton-pipelines
```

Expected result: `PROGRAMMED=True` with an `ADDRESS`. External DNS publishes this IP.

```bash
kubectl get gateway tekton-dashboard-gateway -n tekton-pipelines \
  -o jsonpath='{.status.addresses[0].value}{"\n"}'
```

### 10.3 HTTPRoute

File `tekton-dashboard-httproute.yaml`:

***review comment*** { # Update the DNS Name Eg. tekton.labX.k8slabs.com  and remove this comment from this file }
***review comment*** { Make sure no white spaces in front of the placeholder values replaced in the file }
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/tekton-dashboard-httproute.yaml

| Important argument | Value |
|---|---|
| parentRefs | Gateway `tekton-dashboard-gateway` |
| hostnames | `tekton.k8slabs.com` (External DNS reads this value) |      ***review comment*** {tekton.labX.k8slabs.com}
| backendRefs | Service `tekton-dashboard`, port `9097` |

```bash
kubectl apply -f tekton-dashboard-httproute.yaml
```

### 10.4 Validation: Istio routing

```bash
kubectl get certificate,gateway,httproute -n tekton-pipelines
kubectl describe httproute tekton-dashboard-route -n tekton-pipelines
```

Expected result: Certificate `READY=True`, Gateway `PROGRAMMED=True`, and HTTPRoute conditions `Accepted=True` and `ResolvedRefs=True`.

## 11. External DNS Reconciliation

### 11.1 Rolling restart (when required)

If the record is not created after the HTTPRoute is applied, restart External DNS so it re-reads the resources:

```bash
kubectl rollout restart deployment external-dns -n tanzu-system-service-discovery
kubectl rollout status deployment external-dns -n tanzu-system-service-discovery

kubectl get pods -n tanzu-system-service-discovery
kubectl logs -n tanzu-system-service-discovery -l app=external-dns --tail=50
```
Output: Below line confirms that the entries are added successfully

time="2026-11-13T09:30:53Z" level=info msg="All records are already up to date"

Confirm the deployment name (`kubectl get deployment -n tanzu-system-service-discovery`) and whether the restart is needed in the validated sequence (TO VERIFY).

Note: The DNS Entries are created auotmatically

Expected result: a log line showing the record for `tekton.labX.k8slabs.com` being created or updated.  

### 11.2 Verify the DNS record

```bash
nslookup <FQDN>       # Update Domain Name - Eg. tekton.labX.k8slabs.com
```

Expected result: the IP printed equals the Gateway address from Section 10.2. Also confirm in Cloudflare that the A record and its TXT ownership record exist.

## 12. Dashboard Access

Open the Dashboard:

***Review comment***

{

```text
https://tekton.labX.k8slabs.com      >>> labX added
```
}

The browser must show a valid certificate and the Tekton Dashboard.

<img width="1637" height="660" alt="image (3)" src="https://github.com/user-attachments/assets/bbea6d4e-1b6e-4349-aebe-3f35c1e5740b" />

**Production:** the Dashboard can act on the cluster. Add authentication, RBAC and network restrictions before wider use.


## 13. Create the cicd Namespace with its Security Label

All application-specific Tekton resources, including Tasks, Pipelines, TaskRuns, PipelineRuns, and related CI/CD resources, will be created in the cicd namespace unless explicitly configured otherwise. Tekton's core components, such as the Pipelines controller and webhook, remain in the tekton-pipelines namespace.

```bash
# 1. Check
kubectl get namespace cicd

# 2. Create only if it does not exist
kubectl create namespace cicd

# 3. Apply the label
kubectl label namespace cicd \
  pod-security.kubernetes.io/enforce=privileged \
  --overwrite
```
### 13.1 Validation

```bash
kubectl get namespace cicd --show-labels 
kubectl get namespace cicd  -o jsonpath='{.metadata.labels.pod-security\.kubernetes\.io/enforce}{"\n"}'
```

**Expected result:** Namespace status should be `Active`, and the label `pod-security.kubernetes.io/enforce=privileged` should be present.

**Security note:** this label enables privileged capabilities for build workloads. In production, use the least-privilege settings your VKS, BuildKit and Buildpacks implementation and security policy support.


## 14. Smoke Test: Task and TaskRun

This minimal test proves Tekton itself works. Application Tasks (clone, build, sign, update) start in Section 16.

| File | Purpose |
|---|---|
| `test-task.yaml` | Task that prints a test message |
| `taskrun.yaml` | TaskRun `hello-taskrun` that executes the Task |

Task: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/test-task/test-task.yaml

TaskRun: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/test-task/taskrun.yaml

```bash
kubectl apply -f test-task.yaml
kubectl apply -f taskrun.yaml
```

### 14.1 Validation


```bash
kubectl get task,taskrun -n cicd
kubectl logs -l tekton.dev/taskRun=hello-taskrun -n cicd
```

Screenshot from 2026-10-08 14-57-10

In the Dashboard: Select the Namespace → `cicd` from right top corner . The Task and TaskRun are visible.  

<img width="1873" height="636" alt="Screenshot from 2026-10-08 14-57-10" src="https://github.com/user-attachments/assets/39199f3e-b49f-4af3-8c25-05d1ec498786" />

## 15. Installation Validation and Final Verification

Run once at the end. Every line must pass.

| Area | Command | Expected |
|---|---|---|
| Cluster | `kubectl get nodes -o wide` | All nodes Ready |
| Tekton | `kubectl get pods -n tekton-pipelines` | Controller, webhook, dashboard Running |
| Tekton CRDs | `kubectl get crd \| grep tekton` | tekton.dev CRDs present |
| Namespace | `kubectl get namespace cicd --show-labels` | Active, enforce=privileged |
| Dashboard | `kubectl get pods,svc -n tekton-pipelines \| grep dashboard` | 1/1 Running, port 9097 |
| External DNS | `kubectl get pods -n tanzu-system-service-discovery` | Running, no log errors |
| Istio | `kubectl get certificate,gateway,httproute -n tekton-pipelines` | Ready, Programmed, Accepted |
| HTTPRoute | `kubectl describe httproute tekton-dashboard-route -n tekton-pipelines` | Accepted=True | 
| DNS | `dig +short tekton.labX.k8slabs.com` | Gateway IP |
| User access | Open `https://tekton.labX.k8slabs.com` | Dashboard loads |

## 16. Tekton Pipeline Tasks

Create each Task below by applying its manifest from `manifest/tekton-catalyst` (Figure 1). Each Task lists its important arguments. A single validation at the end of this section checks all of them.

### 16.1 git-clone

`git-clone ` is a Tekton Task used to clone source code from a Git repository into a shared workspace. It cleans the workspace, clones the required branch/revision, and validates the Git repository. The cloned source code is then used by BuildKit or Buildpacks to build the container image.

File `git-clone-update-task.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/git-clone-update-task.yaml

| Important argument | Meaning |
|---|---|
| Repository URL | Git repository to clone (`REPO_URL` from the TriggerBinding in Section 20.6.1) |
| Revision | Branch or revision to clone (`main`) |
| Workspace `source` | Shared workspace that receives the cloned code |
| Credentials | `username` and `token` keys of Secret `repo-git-credentials` (Section 17) |

```bash
kubectl apply -f git-clone-update-task.yaml
```

### 16.2 detect-build-type

`detect-build-type` is a Tekton Task that checks whether the source repository contains a Dockerfile. It selects BuildKit when a Dockerfile exists, otherwise it selects Cloud Native Buildpacks. The selected build type is stored as a Tekton result for the next pipeline task to use.

File `detect-build-type.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/detect-build-type.yaml

| Important argument | Meaning |
|---|---|
| Workspace `source` | Cloned source code that is checked |
| Check | Dockerfile present → BuildKit, absent → Buildpacks |
| Result | Selected build type, read by the Pipeline `when` conditions (Section 18) |

```bash
kubectl apply -f detect-build-type.yaml
```

### 16.3 Buildpacks task

Buildpacks automatically detect an application's language and dependencies, build it without requiring a Dockerfile, and package it into a container image that Tekton can push to Harbor.

Prerequisites for Buildpacks Tasks: 
1. PersistentVolumeClaim (PVC) for source and buildpack working files
2. ServiceAccount
3. Harbor registry secret
4. Attach the secret to the ServiceAccount
5 `kubectl` access to the VKS cluster
6 The `cicd` namespace exists
7 Tekton Pipelines installed
8 vSphere CSI storage available
9 Harbor credentials with push access to the target project

```
VKS
 |
 v
Buildpacks
 |
 v
lab25-harbor.lab25.sunfire.lab/devops/myapp
```

#### 16.3.1 Create the PVC for Buildpacks

The official Buildpacks/Tekton guide uses a PVC to store the cloned source and buildpack working files. Its example uses `500Mi`; for a real application, start larger.

Check the StorageClass

Your VKS environment already has vSphere CSI storage. Confirm which StorageClass is available (and which is the default):

```bash
kubectl get storageclass
```

Look for the StorageClass using the provisioner:

```
csi.vsphere.vmware.com
```
{ 
**For VCF 9.1.1 with VKS 3.8**

Check the available StorageClasses:

```
kubectl get storageclass
```
Example output:

```
NAME                                  PROVISIONER              RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
lab-gold-storage-policy               csi.vsphere.vmware.com   Delete          Immediate              true                   25h
lab-gold-storage-policy-latebinding   csi.vsphere.vmware.com   Delete          WaitForFirstConsumer   true                   25h
```
StorageClass selection: Use the StorageClass that does not have the latebinding suffix in its name. Its VOLUMEBINDINGMODE must be Immediate.

For this environment, use:
```
lab-gold-storage-policy
```
}

#### 16.3.2 Apply the PVC manifest

File: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/buildpacks-source-pvc.yaml

Apply:
```
kubectl apply -f buildpacks-source-pvc.yaml
```

#### 16.3.3 Validate the PVC

```bash
kubectl get pvc -n cicd -w
```

Wait until the status shows:

```
NAME                    STATUS
buildpacks-source-pvc   Bound
```

---

#### 16.3.4 Create the ServiceAccount

```bash
kubectl create serviceaccount buildpacks-service-account -n cicd
```

Verify:

```bash
kubectl get sa buildpacks-service-account -n cicd
```

Expected:

```
NAME                         SECRETS   AGE
buildpacks-service-account   0         ...
```

{

**Observed output (VCF 9.1.1 with VKS 3.8):**

```
NAME                         AGE
buildpacks-service-account   12s
```

Note: In VKS 3.8, the kubectl get sa output does not display the SECRETS column. This is an output-format difference; the ServiceAccount was created successfully.


}

#### 16.3.5 Create the Harbor Registry Secret

Since images are pushed to Harbor, create a "docker-registry" type secret in the `cicd` namespace.

Harbor URL: `lab25-harbor.lab25.sunfire.lab`

```bash
kubectl create secret docker-registry harbor-registry-secret \
  --docker-server=lab25-harbor.lab25.sunfire.lab \
  --docker-username='<HARBOR_USERNAME>' \
  --docker-password='<HARBOR_PASSWORD>' \
  -n cicd
```
{ **Change in VCF 9.1.1: Update Harbor URL format to harborX.labX.sunfire.lab (e.g., harbor11.lab11.sunfire.lab).** }


Replace `<HARBOR_USERNAME>` and `<HARBOR_PASSWORD>` with your Harbor credentials (a robot account is recommended).

Verify:

```bash
kubectl get secret harbor-registry-secret -n cicd
```

---

#### 16.3.6 Attach the Harbor Secret to the ServiceAccount

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

> **Tip:** `imagePullSecrets` covers *pulling* images. Tekton uses credentials from the ServiceAccount's `secrets` list to *push* to a registry. If the build fails to push to Harbor, also add the secret there:
>
> ```bash
> kubectl patch serviceaccount buildpacks-service-account \
>   -n cicd \
>   -p '{"secrets":[{"name":"harbor-registry-secret"}]}'
> ```

{
 **Change in VCF 9.1.1: Recheck the conditional statement 'If the build fails to push to Harbor'. Make adding the secret to the ServiceAccount's secrets list a required step rather than a conditional tip, as Tekton requires it to push images to Harbor.** 

Note: imagePullSecrets only covers pulling images. Tekton uses credentials from the ServiceAccount's secrets list to push images to a registry. Ensure the Harbor secret is added to the ServiceAccount's secrets list so builds can push to Harbor.

}


#### 16.3.7 Checklist

| Resource | Name | Namespace |
|---|---|---|
| PVC | `buildpacks-source-pvc` | `cicd` |
| ServiceAccount | `buildpacks-service-account` | `cicd` |
| Secret | `harbor-registry-secret` | `cicd` |


### 16.4 buildpacks-phases

A ready-made Tekton Task that builds an image from source code using Cloud Native Buildpacks. It builds the image when there is no Dockerfile.

File `buildpacks-phase-task.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/buildpacks-phase-task.yaml

| Important argument | Meaning |
|---|---|
| Builder image | Buildpacks builder image, passed by the PipelineRun (Section 19) |
| Workspaces `source`, `cache` | Source code and per-run build cache PVCs |
| Result | `APP_IMAGE_DIGEST` |
| Digest file | `$(workspaces.source.path)/image-digest` |

```bash
kubectl apply -f buildpacks-phase-task.yaml -n cicd
```

### 16.5 buildkit-build

`buildkit-build` is a Tekton Task that runs when `detect-build-type` detects a Dockerfile, using BuildKit to build the application container image. It takes the source code from the shared workspace, builds the image from the Dockerfile, and pushes it to Harbor with the configured credentials.

File `buildkit-build-task-update.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/buildkit-build-task-update.yaml

| Important argument | Meaning |
|---|---|
| Harbor image | Target image in Harbor, passed by the PipelineRun |
| Workspace `dockerconfig` | Harbor credentials (Secret `harbor-registry-secret`) |
| Harbor CA | ConfigMap `harbor-ca-cert` mounted through `taskRunSpecs` in the PipelineRun |
| Registry setting | References the Harbor CA. Does not use `registry.insecure=true` |
| Result | `IMAGE_DIGEST` |
| Digest file | `$(workspaces.source.path)/image-digest` |

```bash
kubectl apply -f buildkit-build-task-update.yaml
```

### 16.6 sign-image (Syft + Cosign)

#### 16.6.1 Install and Configure Cosign

##### 16.6.1.1 Check the Jumpbox Operating System

Before installing Cosign, verify the operating system and CPU architecture of your jumpbox.

```bash
cat /etc/os-release
uname -m
```

For an `x86_64`/`amd64` system, you can install Cosign using the official Sigstore release binary.

Official releases: [Cosign Releases](https://github.com/sigstore/cosign/releases)

##### 16.6.1.2 Download and Install Cosign

Download the latest Cosign binary and install it in `/usr/local/bin`.

```bash
cd /tmp

curl -fL \
https://github.com/sigstore/cosign/releases/download/v3.1.3/cosign-linux-amd64 \
  -o cosign

chmod +x cosign

sudo mv cosign /usr/local/bin/cosign
```

**Note:** We use Cosign v3.1.3 for compatibility with our current Harbor version. You can use a newer version of Cosign if it is compatible with your Harbor deployment. This command is for `x86_64`/`amd64` systems. For ARM64 systems, use the corresponding ARM64 release binary. Ensure `/usr/local/bin` is in your `PATH`.

Verify the installation:

```bash
cosign version
```

##### 16.6.1.3 Generate the Cosign Key Pair

Generate a public and private key pair for signing container images.

```bash
cosign generate-key-pair
```

Note: Please set password as per your requirement, We have not kept any password.

The command generates the following files in the current directory:

- `cosign.key` — Private key used to sign container images.
- `cosign.pub` — Public key used to verify image signatures.

Verify that both files exist:

```bash
ls -l cosign.key cosign.pub
```

##### 16.6.1.4 Protect the Private Key

Restrict access to the private key so that only its owner can read or modify it.

```bash
chmod 600 cosign.key
```

**Important:** Never commit `cosign.key` to Git or share it publicly. Store the key and its password securely.

#### 16.6.1.5 Cosign password Secret (cosign-password)

Password of the Cosign private key.

File `cosign-pass.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/cosign-pass.yaml

| Important argument | Value |
|---|---|
| Secret name | `cosign-password` |
| Content | Password of the Cosign private key (placeholder in Git) |

Note: We have not kept any password while generating key-pair, so we have kept password as blank

```bash
kubectl apply -f cosign-pass.yaml
```

#### 16.6.1.6 Cosign key pair Secret (cosign-key)

Cosign private and public key. Create it from the generated key pair (no manifest file):

```bash
kubectl create secret generic cosign-key \
  -n cicd \
  --from-file=cosign.key=cosign.key \
  --from-file=cosign.pub=cosign.pub
```

##### 16.6.1.7 Validate the Kubernetes Secret

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

The Secret should contain both `cosign.key` and `cosign.pub`.

##### 16.6.1.8 Integration with Tekton

Tekton tasks can mount the `cosign-key` Secret through a workspace or another supported Secret-mounting configuration. The signing task must also receive the private-key password securely.

The private key is used to sign the container image, while the public key can be used to verify its signature.

Syft generates the SBOM, and Cosign signs the container image and can attest to the SBOM. Together, they help you track what software is inside an image and verify its integrity and provenance.

This `sign-image` Task is the security/signing stage after BuildKit or Buildpacks. Its job is to generate an SBOM, sign the image with Cosign, attach the SBOM as an attestation, and return the image digest.

File `sign-image-task-update.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/sign-image-task-update.yaml

| Important argument | Meaning |
|---|---|
| `COSIGN_TLOG_UPLOAD` | Defaults to `false` for the air-gapped/private-registry lab flow. Set to `true` only when the cluster can reach the approved Sigstore transparency-log service |
| Image digest | Digest from the build Task. The image is signed by digest, not by tag |
| Workspace `cosign-key` | Cosign key pair (Secret `cosign-key`) |
| Cosign password | Secret `cosign-password` |
| SBOM | Generated with Syft, stored in the `source` workspace, attached as an `spdxjson` attestation |
| Output | Image digest, passed to `update-values` |

```bash
kubectl apply -f sign-image-task-update.yaml
```
Note: Cosign signing and SBOM attestation are working. The issue is with Harbor's SBOM display/API behavior, not an established error in your Tekton Task.


{ **Change in VCF 9.1.1: In sign-image-task-update.yaml, update the step images for sbom, sign, attest-sbom, and image-digest to pull from harbor.sunfire.lab. In buildkit-build-task-update.yaml, replace all occurrences of lab25-harbor.lab25.sunfire.lab with harbor11.lab11.sunfire.lab (or harborX.labX.sunfire.lab).** }


### 16.7 update-values

`update-values` is the GitOps update stage of the Tekton pipeline. It updates the Helm chart's `values.yaml` with the newly built image and its digest, commits the change, and pushes it back to the repository.

File `update-values.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/update-values.yaml

| Important argument | Meaning |
|---|---|
| Image digest | Received from the `sign-image` Task |
| Target file | `helm-charts/values.yaml` |
| Commit message prefix | `ci: update TravelPortal image digest` (the EventListener CEL filter ignores commits with this prefix) |
| Credentials | `username` and `token` keys of Secret `repo-git-credentials` |

```bash
kubectl apply -f update-values.yaml
```
{ **Change in VCF 9.1.1: Add default: <REPO_URL> under Important arguments** 

Important argument:

```
YAML
default: <REPO_URL>         # Update the REPO_URL of Manifest Repo - e.g. https://gitea.lab12.k8slabs.com/admin/k8s-manifests.git
```

}

### 16.8 Validation – Tasks

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
{
**Verify the Task (VCF 9.1.1 with VKS 3.8):**

```
kubectl get tasks -n cicd
```
Observed output:

```
NAME                AGE
buildkit-build      2d20h
buildpacks-phases   2d20h
detect-build-type   2d21h
git-clone-update    2d21h
hello-task          2d21h
sign-image          2d20h
update-values       25s
```
}


## 17. Pipeline Dependencies (Storage, ConfigMap, Secrets, Service Account)

Before triggering the Pipeline and PipelineRun, create the required storage, configuration, and credentials. A single validation at the end of this section checks everything.

Note: before files where the Harbor URL contains 'cicd' to ensure the project 'cicd' is created in Harbor first.

Prerequisite: Ensure that a project named cicd is created in Harbor before applying manifests that reference harborX.labX.sunfire.lab/cicd/.

### 17.1 Source and Build Cache Storage

The PipelineRun creates dedicated `source` and `cache` PVCs for each run using the `lab-gold-storage-policy` StorageClass. This prevents concurrent PipelineRuns from overwriting the same shared source workspace.

If the StorageClass has a different name in your VKS environment, replace `lab-gold-storage-policy` in the PipelineRun and TriggerTemplate workspace definitions.

### 17.2 Harbor CA ConfigMap

Create a ConfigMap named `harbor-ca-cert` in the `cicd` namespace from your Harbor / vCenter CA certificate (the key must be `ca.crt`):

{ **Change in VCF 9.1.1: Explain that the Harbor certificate must be downloaded and saved as harbor-ca.crt in the current working directory (/root/) before running kubectl create configmap, otherwise it fails with 'error: error reading harbor-ca.crt: no such file or directory'.** }

Download the Harbor CA certificate (ca.crt), copy it to your working directory (for example, /root/), and rename it to harbor-ca.crt before creating the ConfigMap:


```bash
kubectl create configmap harbor-ca-cert \
  -n cicd \
  --from-file=harbor-ca.crt
```

### 17.3 Secrets

The pipeline needs four secrets in the `cicd` namespace. Create them one by one. Replace every placeholder value with your real credentials before applying, and never commit real credentials to Git.

#### 17.3.1 Git credentials Secret (repo-git-credentials)

Follow the steps to generate the token:

##### 17.3.1.1 Steps to create a Gitea PAT

1. Log in to Gitea
Open your Gitea URL in a browser and sign in with your account.

2. Open Application settings
Click your profile icon → Settings → Applications.

3. Generate a new token Under Manage Access Tokens, enter a token name, such as tekton-cicd-token, and click Generate Token.
It gives message as > `You must select at least one permission to create a token`
Selected Read & Write for all the 9 pointers.

4. Copy and save the token
Copy the generated PAT immediately and store it securely. You may not be able to view it again.



File `github-push-secret.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/github-push-secret.yaml

```bash
kubectl apply -f github-push-secret.yaml
```
The 'username' and 'token' keys from the 'repo-git-credentials' Secret are used by the 'update-values' task as environment variables, as shown below.

```bash
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

#### 17.3.2 Harbor registry Secret (harbor-credentials)

Harbor credential "Opaque" type secret used to sign images.

File `harbor-credentials.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/harbor-credentials.yaml

```bash
kubectl apply -f harbor-credentials.yaml
```



### 17.4 ServiceAccount, role, and rolebinding for tekton-pipeline in cicd namespace

The TriggerTemplate and PipelineRun use the dedicated `tekton-pipeline-sa` ServiceAccount.

File `tekton-pipeline-sa.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/tekton-pipeline-sa.yaml

| Important argument | Value |
|---|---|
| ServiceAccount | `tekton-pipeline-sa` in `cicd` |
| Used by | PipelineRun and TriggerTemplate |

```bash
kubectl apply -f tekton-pipeline-sa.yaml
```

### 17.5 Validation – Dependencies

```bash
kubectl get storageclass lab-gold-storage-policy
kubectl get configmap harbor-ca-cert -n cicd
kubectl get secret repo-git-credentials harbor-registry-secret harbor-credentials cosign-password cosign-key -n cicd
kubectl get sa tekton-pipeline-sa -n cicd
```

Expected result: every object is found, with no `NotFound` errors.

## 18. Pipeline

Tasks are added to the Pipeline using `taskRef`, which references previously created Tekton Tasks. `params` pass required values to each Task, `workspaces` share files/credentials, and `runAfter`/`when` control the execution order and conditions. The Pipeline first clones the repository, then detects whether a Dockerfile exists and runs BuildKit or Buildpacks accordingly. After the image is built, the sign Task generates the SBOM and signs/attests the image. Finally, the `update-values` Task receives the image digest from the sign Task, updates `values.yaml`, commits the change, and pushes it back to the same Git repository—completing the GitOps update.

File `travelportal-pipeline-values-update.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/travelportal-pipeline-values-update.yaml

| Important argument | Meaning |
|---|---|
| Pipeline name | `travelportal-pipeline-values-update` |
| `taskRef` | The six Tasks from Section 16 |
| Order | clone → detect → (BuildKit or Buildpacks) → sign → update-values |
| `when` | Selects BuildKit or Buildpacks from the `detect-build-type` result |
| `runAfter` | Sets the order of the Tasks |
| Workspaces | `source`, `cache`, `dockerconfig`, `git-credentials`, `cosign-key` |

```bash
kubectl apply -f travelportal-pipeline-values-update.yaml
```

The Pipeline is validated together with the PipelineRun in Section 19.


## 19. PipelineRun

The PipelineRun is used to start and execute the `travelportal-pipeline-values-update` Pipeline. It provides the pipeline parameters, connects the required workspaces and secrets, and applies additional Pod configuration needed during the pipeline execution. The `pipelineRef` selects the `travelportal-pipeline-values-update` Pipeline, while `params` provide the Git repository, branch, Harbor image, and Buildpacks builder image that the Pipeline will use. The workspaces provide the required storage and credentials: the `source` and `cache` PVCs are created per PipelineRun, `dockerconfig` provides Harbor authentication, `git-credentials` provides repository push credentials, and `cosign-key` provides the image-signing key. The SBOM is stored in the `source` workspace, so no separate SBOM workspace is required. The `taskRunSpecs` customize the BuildKit TaskRun by adding the `harbor-ca` ConfigMap as a volume. This allows the BuildKit Pod to access the Harbor CA certificate. The BuildKit registry configuration references that CA and does not use `registry.insecure=true` for the Harbor HTTPS endpoint.

File `travelportal-pipelinerun-pack-values-update.yaml`:

YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/travelportal-pipelinerun-pack-values-update.yaml

| Important argument | Meaning |
|---|---|
| `pipelineRef` | `travelportal-pipeline-values-update` |
| `serviceAccountName` | `tekton-pipeline-sa` |
| `params` | Git repository, branch, Harbor image, Buildpacks builder image |
| Workspace `source`, `cache` | PVCs created per run from StorageClass `lab-gold-storage-policy` |
| Workspace `dockerconfig` | Secret `harbor-registry-secret` |
| Workspace `git-credentials` | Secret `repo-git-credentials` |
| Workspace `cosign-key` | Secret `cosign-key` |
| `taskRunSpecs` | Adds ConfigMap `harbor-ca-cert` as a volume to the BuildKit TaskRun |

```bash
kubectl create -f travelportal-pipelinerun-pack-values-update.yaml
```

### 19.1 Validation – Pipeline, PipelineRun, and output

Pipeline and run status:

```bash
kubectl get pipeline -n cicd
kubectl get pipelinerun -n cicd -w
kubectl get pvc -n cicd
```

Expected result: the Pipeline exists, the PipelineRun finishes with `SUCCEEDED=True`, and the `source` and `cache` PVCs are `Bound`. You can also follow the run in the Tekton Dashboard.

{ for vcf 9.1.1 

Needs re-verification:

 1) Fix 'docker login' x509 certificate signed by unknown authority error by copying harbor-ca.crt into /etc/docker/certs.d/harbor11.lab11.sunfire.lab/ca.crt.

 2) Update 'cosign verify-attestation' because --offline is deprecated in Cosign v3.1.3 and keyless verification fails without --certificate-identity or --key. 


Trust the Harbor CA certificate on the jumpbox and log in to Harbor:

```Bash

mkdir -p /etc/docker/certs.d/harbor11.lab11.sunfire.lab
cp /root/harbor-ca.crt /etc/docker/certs.d/harbor11.lab11.sunfire.lab/ca.crt
docker login harbor11.lab11.sunfire.lab

```
cosign verify-attestation \
  --key cosign.pub \
  --insecure-ignore-tlog=true \
  --type spdxjson \
  "harbor11.lab11.sunfire.lab/cicd/travelportal@sha256:<IMAGE_DIGEST>"

}

Image in Harbor (from a machine that can reach Harbor):

```bash
docker login lab25-harbor.lab25.sunfire.lab
docker pull lab25-harbor.lab25.sunfire.lab/cicd/travelportal:latest
```

Cosign signature and SBOM attestation:

```bash
cosign verify \
  --insecure-ignore-tlog=true \
  --key cosign.pub \
  "lab25-harbor.lab25.sunfire.lab/cicd/travelportal@sha256:<IMAGE_DIGEST>"

cosign verify-attestation \
  --offline \
  --insecure-ignore-tlog=true \
  --type spdxjson \
  "lab25-harbor.lab25.sunfire.lab/cicd/travelportal@sha256:<IMAGE_DIGEST>"
```

Expected result: the image pulls, the signature verifies, and the SBOM attestation is associated with the image digest.

## 20. Automatic Pipeline Trigger with Tekton Triggers
 
Until now the PipelineRun was started by hand with `kubectl apply`. In this section a push to the repository starts it automatically. The repository server sits on the internal lab network, so it calls the Tekton EventListener NodePort directly.
 
### 20.1 Lab values used in this section
 
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
 
### 20.2 The Tekton Triggers objects created in this section
 
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
 
### 20.3 Install Tekton Triggers
 
For an Internet-connected VKS cluster, install the Tekton Triggers release before creating the EventListener, TriggerBinding, and TriggerTemplate objects.
 
File: tekton-trigger-release.yaml
 
```bash

kubectl apply --filename https://infra.tekton.dev/tekton-releases/triggers/latest/release.yaml
 
Specific version - v0.34.0

kubectl apply --filename https://infra.tekton.dev/tekton-releases/triggers/previous/v0.34.0/release.yaml

```
 
For an Internet-connected VKS cluster, install the Tekton Interceptor before creating the EventListener, TriggerBinding, and TriggerTemplate objects.
 
File: tekton-interceptors-release.yaml

```

kubectl apply -f https://storage.googleapis.com/tekton-releases/triggers/latest/interceptors.yaml
 
Specific version - v0.34.0

kubectl apply --filename https://infra.tekton.dev/tekton-releases/triggers/previous/v0.34.0/interceptors.yaml

```

Note: Tekton Trigger and Tekton Interceptor needs to be of same version
 
For production, pin the Triggers version that is tested with your selected Tekton Pipelines release instead of using `latest`.
 
#### 20.3.1 Validation – Tekton Trigger
 
***Review comment***
 
{

```bash

kubectl get all -n tekton-pipelines       #below deployment should up and running    >>> corrected the deployment spelling

kubectl get crd | grep triggers.tekton.dev

```

}

Expected result:
 
```text

CRDs: clusterinterceptors, clustertriggerbindings, eventlisteners, interceptors,

      triggerbindings, triggers, triggertemplates (all .triggers.tekton.dev)
 
Pods (all 1/1 Running):

      tekton-triggers-controller-...

      tekton-triggers-core-interceptors-...

      tekton-triggers-webhook-...

```
 
### 20.4 Create the GitHub webhook secret
 
```

openssl rand -hex 32

```

"6addb375b90c7b68bbe9b1b5365ad6fbec8b1aedb2e87ca973849a5a589abb8c"      # Generate token from terminal using this command and use it while creating the gitea-webhook-secret

For example, suppose it returns:

YOUR_RANDOM_SECRET
 
File 'gitea-webhook-secret.yaml'
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/gitea-webhook-secret.yaml
 
```

kubectl apply -f gitea-webhook-secret.yaml

```

### 20.5 EventListener ServiceAccount and RBAC
 
The EventListener runs with `serviceAccountName: repository-trigger-sa`. It needs permission to read the Triggers resources in `cicd`, read the cluster-scoped Triggers resources (`clusterinterceptors`, `clustertriggerbindings`), and create PipelineRuns.
 
#### 20.5.1 Create ServiceAccount rolebinding and RBAC

The EventListener needs permission to create PipelineRuns.
 
Service Account: 

File `gitea-trigger-sa.yaml`:
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/gitea-trigger-sa.yaml
 
| Important argument | Value |

|---|---|

| ServiceAccount | `gitea-trigger-sa` in `cicd` |
 
```bash

kubectl apply -f gitea-trigger-sa.yaml

```
 
#### 20.5.2 Role, RoleBinding, ClusterRole and ClusterRoleBinding
 
Create : eventlistener-gitea-rbac.yaml
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/eventlistener-gitea-rbac.yaml
 
File `gitea-trigger-role.yaml` contains the Role (`repository-trigger-rol`), RoleBinding (`gitea-trigger-rolebinding.yaml`), ClusterRole (`gitea-trigger-cluster-role.yaml`) and ClusterRoleBinding (`gitea-trigger-cluster-rolebinding.yaml`).
 
| Important argument | Meaning |

|---|---|

| Role (namespace `cicd`) | Read `triggerbindings`, `triggertemplates`, `eventlisteners`; create `pipelineruns` |

| ClusterRole | Read `clusterinterceptors`, `clustertriggerbindings` |

| Subject | ServiceAccount `gitea-trigger-sa` |
 
```bash

kubectl apply -f eventlistener-gitea-rbac.yaml

```
 
#### 20.5.3 Validation – RBAC
 
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
 
### 20.6 TriggerBinding, TriggerTemplate and EventListener
 
Create the three objects below, then validate them once at the end.
 
#### 20.6.1 TriggerBinding
 
File `travelportal-gitea-binding.yaml`:
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/travelportal-gitea-binding.yaml
 
| Important argument | Value |

|---|---|

| Name | `travelportal-gitea-binding` |

| `REPO_URL` | `http://10.12.90.62/admin/travelPortal-test-buildpack.git` |

| `REVISION` | `main` |

| Payload fields read | `body.ref`, `body.after`, `body.head_commit.message` |
 
```bash

kubectl apply -f travelportal-gitea-binding.yaml

```
 
#### 20.6.2 TriggerTemplate
 
File `travelportal-gitea-template.yaml`:
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/travelportal-gitea-template.yaml
 
| Important argument | Value |

|---|---|

| Name | `travelportal-gitea-template` |

| `pipelineRef` | `travelportal-pipeline-values-update` |

| `serviceAccountName` | `tekton-pipeline-sa` |

| Repository URL | `http://10.12.90.62/admin/travelPortal-test-buildpack.git` |

| PipelineRun name | `travelportal-build-xxxxx` (generated) |

| Workspaces | Per-run `source` and `cache` PVCs from StorageClass `lab-gold-storage-policy` |
 
```bash

kubectl apply -f travelportal-gitea-template.yaml

```
 
#### 20.6.3 EventListener
 
File `travelportal-gitea-listener.yaml`:
 
YAML: https://github.com/kondurupurandhar/ci-cd-stack-on-vks/blob/main/manifest/tekton-catalyst/travelportal-gitea-listener.yaml
 
| Important argument | Value |

|---|---|

| Name | `travelportal-gitea-listener` |

| `serviceAccountName` | `gitea-trigger-sa` |

| Interceptor | Cluster-scoped `cel` interceptor (main branch only, ignores CI commits) |

| Binding / Template | `travelportal-gitea-binding` / `travelportal-gitea-template` |

| `serviceType` | `NodePort` |

| Application port | `8080` (observed NodePort in this lab: `31877`) |
 
***Review comment***
 
{
 
Note: The manifest specifies only serviceType: NodePort, Kubernetes may assign a different NodePort in another environment, so always confirm the actual value in the validation table above.

}
 
```bash

kubectl apply -f travelportal-gitea-listener.yaml

```
 
Because the manifest specifies only `serviceType: NodePort`, Kubernetes may assign a different NodePort in another environment, so always confirm the actual value in the validation below.
 
#### 20.6.4 Validation – TriggerBinding, TriggerTemplate and EventListener
 
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

### 20.7 Expose the EventListener externally and configure the Gitea webhook.

First check:

kubectl get svc -n cicd | grep travelportal-gitea-listener

kubectl get eventlistener travelportal-gitea-listener -n cicd

kubectl get pods -n cicd | grep travelportal-gitea-listener

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
 
***Review comment
 
{
 
```

kubectl get svc el-travelportal-gitea-listener -n cicd  >>> removed -w 

```

}
 
Expected:
 
```

NAME                              TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)

el-travelportal-gitea-listener   NodePort   198.54.22.125   <none>        8080:31234/TCP,9000:3xxxx/TCP

```

***Review comment***
 
{

2. Retrieve the Kubernetes Node IP

##2.1 Identify the Node Running the EventListener Pod
 
```bash

kubectl get pods -n cicd -o wide

```
 
Expected Output:
 
NAME                                              READY   STATUS      RESTARTS   AGE     IP             NODE                                  NOMINATED NODE   READINESS GATES

el-travelportal-gitea-listener-685c6467d9-2jjpl   1/1     Running     0          9m35s   192.168.2.21   vkc11-node-pool-1-tmqkh-6rxnm-d2jlb   <none>           <none>

hello-taskrun-pod                                 0/1     Completed   0          3d2h    192.168.3.27   vkc11-node-pool-1-tmqkh-6rxnm-7wjsp   <none>           <none>

Url of eventlistner
 
From the output, identify the node running the EventListener Pod
 
##2.2 Retrieve the Node IP Address
 
```bash
 
kubectl get node vkc11-node-pool-1-tmqkh-6rxnm-d2jlb -o wide

```

Output:
 
NAME                                  STATUS   ROLES    AGE    VERSION            INTERNAL-IP   EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION             CONTAINER-RUNTIME

vkc11-node-pool-1-tmqkh-6rxnm-d2jlb   Ready    <none>   4d4h   v1.37.1+vmware.4   10.11.92.2    <none>        Ubuntu 24.04.5 LTS   7.0.0-38-generic (amd64)   containerd://2.4.0+vmware.2-fips
 
Use This IP and Nodeport to access
 
}
 
```
http://<NODE-IP>:<NODEPORT>
http://<NODE-IP>:31234

```
 
***Review comment
 
{

A working EventListener returns `HTTP/1.1 202 Accepted` with a Tekton event ID. This proves the path jumpbox → NodePort → EventListener works, and a new `travelportal-build-xxxxx` PipelineRun appears.
 
}     >>> Need to delete the below section because getting `HTTP/1.1 202 Accepted` this output but everything works fine.
 
Verify the Eventlistener URL is reachable
 
``` 

curl -v  http://<NODE-IP>:<NODEPORT>

```
 
### 20.8 Add the webhook in the repository
 
***Review comment***
 
{
 
Repository: gitea repo (source code)                 >>> Added the gitea repo to be used
 
```text
http://10.12.90.62/admin/TravelPortal-lab10.git

```

}
 
***Review comment***
 
{

This needs to be done on Gitea GUI
 
In source code repo
 
Use This Secret :  "Secrets	gitea-webhook-secret"
 
Note:  Need to add the Secret value generated in Section 20.4
 
}

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
 
### 20.9 Pipeline changes for automatic runs
 
Push the changes to the source repository. This will automatically trigger the pipeline.
 
```

git add .

git commit -m "Your commit message"

git push origin <branch-name>

```
 
#### 20.9.1 Repository credentials for update-values
 
The clone and `update-values` Tasks read the repository credentials from the `username` and `token` keys of the `repo-git-credentials` Secret (created in Section 17.3.1). The token is a Personal Access Token created in the repository server.
 
## 21. Final Validation – End to End
 
Then watch the automation:
 
```bash

kubectl logs -f deployment/el-travelportal-gitea-listener -n cicd

kubectl get pipelineruns -n cicd -w

```
 
***Review comment***         >>> Need to be deleted as unexpected output from above Two commands
 
{

Expected result:
 
```text

Webhook delivery   → HTTP 202 Accepted

New PipelineRun    → travelportal-build-xxxxx

Build              → BuildKit if the repo has a Dockerfile, otherwise Buildpacks

Pipeline           → clone → detect → build → sign → update-values all succeed

Self-trigger       → no second PipelineRun after the CI commit

Harbor             → image digest signed, SBOM attestation present

```

}
 
## 22. Conclusion
 
This implementation provides a Kubernetes-native CI workflow on VKS using Tekton, BuildKit, Buildpacks, Syft, Cosign, and Harbor. The pipeline automatically selects BuildKit or Buildpacks based on the application source, generates an SBOM, signs the immutable image digest, and stores the resulting image and security artifacts in Harbor.
 
The architecture also separates the application source and internal CI/GitOps responsibilities across two repositories, providing a clear and controlled workflow for source management and deployment configuration.
 
Overall, the solution provides an automated, traceable, and security-focused CI process that can be operated in both connected and disconnected VKS environments, provided the required images, dependencies, credentials, and registry artifacts are available locally.
 
