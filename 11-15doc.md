## 11. External DNS Reconciliation

### 11.1 Rolling restart (when required)

If the record is not created after the HTTPRoute is applied, restart External DNS so it re-reads the resources:

```bash
kubectl rollout restart deployment external-dns -n tanzu-system-service-discovery
kubectl rollout status deployment external-dns -n tanzu-system-service-discovery

kubectl get pods -n tanzu-system-service-discovery
kubectl logs -n tanzu-system-service-discovery -l app=external-dns --tail=50
```
***Review comment***
{

Below line confirms that the entries are added successfully

time="2026-11-13T09:30:53Z" level=info msg="All records are already up to date"
}

Confirm the deployment name (`kubectl get deployment -n tanzu-system-service-discovery`) and whether the restart is needed in the validated sequence (TO VERIFY).

***Review comment***

{
Note:
The DNS Entries are created auotmatically


Expected result: a log line showing the record for `tekton.labX.k8slabs.com` being created or updated.    >>> LabX added

}

### 11.2 Verify the DNS record

```bash
nslookup <FQDN>       # Update Domain Name
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

Expected result:c`Active`, with `pod-security.kubernetes.io/enforce=privileged` in the labels. / privileged 

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

***Review comment***
{

In the Dashboard: Select the Namespace → `cicd` from right top corner . The Task and TaskRun are visible.  >>> comment is modified
}

<img width="1873" height="636" alt="Screenshot from 2026-10-08 14-57-10" src="https://github.com/user-attachments/assets/39199f3e-b49f-4af3-8c25-05d1ec498786" />


## 15. Installation Validation and Final Verification

Run once at the end. Every line must pass.

***Review comment***

{

| Area | Command | Expected |
|---|---|---|
| Cluster | `kubectl get nodes -o wide` | All nodes Ready |
| Tekton | `kubectl get pods -n tekton-pipelines` | Controller, webhook, dashboard Running |
| Tekton CRDs | `kubectl get crd \| grep tekton` | tekton.dev CRDs present |
| Namespace | `kubectl get namespace cicd --show-labels` | Active, enforce=privileged |
| Dashboard | `kubectl get pods,svc -n tekton-pipelines \| grep dashboard` | 1/1 Running, port 9097 |
| External DNS | `kubectl get pods -n tanzu-system-service-discovery` | Running, no log errors |
| Istio | `kubectl get certificate,gateway,httproute -n tekton-pipelines` | Ready, Programmed, Accepted |
| HTTPRoute | `kubectl describe httproute tekton-dashboard-route -n tekton-pipelines` | Accepted=True, ResolvedRefs=True | 
| DNS | `dig +short tekton.labX.k8slabs.com` | Gateway IP |
| User access | Open `https://tekton.labX.k8slabs.com` | Dashboard loads |

}
