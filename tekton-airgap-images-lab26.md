# Prerequisites

Download and push the images to the local Harbor registry for Tekton air-gapped installation and Tekton pipeline execution.

## 1. Tekton Installation Images

### 1.1 Controller v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/controller-10a3e32792f33651396d02b6855a6e36:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/controller-10a3e32792f33651396d02b6855a6e36:v1.17.0
```

### 1.2 Entrypoint v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/entrypoint-bff0a22da108bc2f16c818c97641a296:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/entrypoint-bff0a22da108bc2f16c818c97641a296:v1.17.0
```

### 1.3 Nop v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/nop-8eac7c133edad5df719dc37b36b62482:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/nop-8eac7c133edad5df719dc37b36b62482:v1.17.0
```

### 1.4 Sidecarlogresults v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/sidecarlogresults-7501c6a20d741631510a448b48ab098f:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/sidecarlogresults-7501c6a20d741631510a448b48ab098f:v1.17.0
```

### 1.5 Workingdirinit v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/workingdirinit-0c558922ec6a1b739e550e349f2d5fc1:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/workingdirinit-0c558922ec6a1b739e550e349f2d5fc1:v1.17.0
```

### 1.6 Events v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/events-a9042f7efb0cbade2a868a1ee5ddd52c:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/events-a9042f7efb0cbade2a868a1ee5ddd52c:v1.17.0
```

### 1.7 Resolvers v1.17.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/resolvers-ff86b24f130c42b88983d3c13993056d:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/resolvers-ff86b24f130c42b88983d3c13993056d:v1.17.0
```

### 1.8 Webhook v1.17.0
 
```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/pipeline/webhook-d4749e605405422fd87700164e31b2d1:v1.17.0 \
  docker://harbor.sunfire.lab/tektoncd/pipeline/webhook-d4749e605405422fd87700164e31b2d1:v1.17.0
```

### 1.9 BusyBox v1.17.0

```bash
skopeo copy --all \
  docker://cgr.dev/chainguard/busybox@sha256:19f02276bf8dbdd62f069b922f10c65262cc34b710eea26ff928129a736be791 \
  docker://harbor.sunfire.lab/tektoncd/chainguard/busybox:tekton-v1.17.0
```

## 2. Tekton Dashboard Image v0.73.0

```bash
skopeo copy --all \
  docker://ghcr.io/tektoncd/dashboard/dashboard-9623576a202fe86c8b7d1bc489905f86:v0.73.0 \
  docker://harbor.sunfire.lab/tektoncd/dashboard/dashboard-9623576a202fe86c8b7d1bc489905f86:v0.73.0
```

## 3. Tekton AutoTrigger

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

## 4. Tekton Pipeline Images for Tekton CI Tasks

### 4.1 Git-Clone v2.54.0

```bash
imgpkg copy -i docker.io/alpine/git:v2.54.0 \
  --to-repo harbor22.lab22.sunfire.lab/library/alpine/git
```

### 4.2 Detect v3.20

```bash
imgpkg copy -i docker.io/library/alpine:3.20 \
  --to-repo harbor22.lab22.sunfire.lab/library/alpine
```

### 4.3 BuildKit v0.34.1 

```bash
imgpkg copy -i docker.io/moby/buildkit:v0.34.1 \
  --to-repo harbor22.lab22.sunfire.lab/library/moby/buildkit
```

### 4.4 Buildpack Phase Images

#### 4.4.1 Skopeo-JQ v0.1.3 

```bash
imgpkg copy -i quay.io/halkyonio/skopeo-jq:0.1.3@sha256:1b3d21ad541227dc9d3e793d18cef9eb00a969c0c01eb09cab88997bc63680c6 \
  --to-repo harbor22.lab22.sunfire.lab/library/skopeo-jq
```

#### 4.4.2 UBI 8 Minimal v8.10-1793307920

```bash
imgpkg copy -i registry.access.redhat.com/ubi8/ubi-minimal@sha256:b2a1bec3dfbc7a14a1d84d98934dfe8fdde6eb822a211286601cf109cbccb075 \
  --to-repo harbor22.lab22.sunfire.lab/library/ubi8/ubi-minimal
```

#### 4.4.3 UBI 8 Python v3.11

```bash
imgpkg copy -i registry.access.redhat.com/ubi8/python-311@sha256:43605cb2491ef2297a7acf4b4bf0b7f54f0c91b96daf12ae41c49cc7f192b153 \
  --to-repo harbor22.lab22.sunfire.lab/library/ubi8/python-311
```

#### 4.4.4 Run Jammy Base v0.1.294 

```bash
imgpkg copy -i docker.io/paketobuildpacks/run-jammy-base:0.1.294 \
  --to-repo harbor22.lab22.sunfire.lab/library/paketobuildpacks/run-jammy-base
```
#### 4.4.5 Builder Jammy Base v0.4.675 

```bash
imgpkg copy -i docker.io/paketobuildpacks/builder-jammy-base:0.4.675 \
  --to-repo harbor22.lab22.sunfire.lab/library/paketobuildpacks/builder-jammy-base
```

### 4.5 Cosign and SBOM Images

##### 4.5.1 Anchore Syft v1.55.0

```bash
imgpkg copy -i docker.io/anchore/syft:1.55.0 \
  --to-repo harbor22.lab22.sunfire.lab/library/anchore/syft
```

#### 4.5.2 Cosign v3.1.3

```bash
imgpkg copy -i ghcr.io/sigstore/cosign/cosign:v3.1.3 \
  --to-repo harbor22.lab22.sunfire.lab/library/sigstore/cosign/cosign

```
#### 4.5.3 alpine v3.20

```bash
imgpkg copy -i docker.io/library/alpine:3.20 \
  --to-repo harbor22.lab22.sunfire.lab/library/alpine
```

### 4.6 Update-values

```bash
imgpkg copy -i docker.io/alpine/git:v2.54.0 \
  --to-repo harbor22.lab22.sunfire.lab/library/alpine/git
```


