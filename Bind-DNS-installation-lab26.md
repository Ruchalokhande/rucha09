# BIND DNS Server Installation and External-DNS Integration on Lab22

## 1. Document Purpose

This document describes the installation and configuration of a BIND DNS
server on the `lab22-jumpbox` in an air-gapped environment. It covers:

-   Downloading BIND RPM packages and dependencies on an
    internet-connected staging VM.
-   Transferring the RPM archive to the air-gapped `lab22-jumpbox`.
-   Installing and configuring BIND.
-   Creating the authoritative DNS zone and manual DNS records.
-   Validating DNS service operation and resolution.
-   Optionally forwarding external DNS queries to an approved upstream
    DNS server.
-   Configuring TSIG authentication for dynamic DNS updates.
-   Testing dynamic DNS updates using `nsupdate`.
-   Integrating BIND with External-DNS on the VKS cluster through
    RFC2136.

## 2. Environment Details

  Item                                        Value
  ------------------------------------------- ----------------------------------
  Target host                                 `lab22-jumpbox`
  BIND server IP                              `10.22.99.5`
  Target operating system                     CentOS 8
  Target architecture                         `x86_64`
  DNS zone configured in BIND                 `vks.lab22.sunfire.lab`
  External-DNS VKS host-network range         `10.22.92.0/24`
  Jumpbox subnet                              `10.22.99.0/24`
  Upstream DNS server (optional forwarding)   `10.0.254.5`
  External-DNS namespace                      `tanzu-system-service-discovery`
  TSIG key name                               `externaldns-key`
  TSIG algorithm                              `hmac-sha256`

> **Important:** The commands and configuration values below are
> retained from the supplied implementation notes. Review the validation
> notes at the end before using this document in another lab. In
> particular, the source notes contain different DNS suffixes in some
> test commands, and the TSIG secret shown in the notes should be
> treated as exposed and rotated before production use.

## 3. Implementation Flow

1.  Download EL8-compatible BIND RPMs and dependencies on a connected
    VM.
2.  Transfer the RPM archive to `lab22-jumpbox`.
3.  Install BIND without accessing online repositories.
4.  Configure the authoritative DNS zone and its initial records.
5.  Validate and start the `named` service.
6.  Test DNS resolution.
7.  Configure optional forwarding, if permitted by the environment.
8.  Generate and configure the TSIG key for dynamic updates.
9.  Test dynamic DNS updates.
10. Allow the VKS External-DNS host-network subnet to query BIND and
    configure External-DNS to use RFC2136.

------------------------------------------------------------------------

## 4. Download BIND RPM Packages on the Connected VM

### 4.1 Check the Operating System and Architecture

Run these commands on the internet-connected VM before downloading
packages. The target system is CentOS 8 `x86_64`, so use compatible EL8
`x86_64` RPM packages.

``` bash
cat /etc/os-release
uname -m
```

### 4.2 Install the DNF Download Utility

Install the DNF plugins package on the connected VM:

``` bash
sudo dnf install -y dnf-plugins-core
```

**Repository note:** If the connected VM is CentOS 8 and its normal
repositories no longer work, configure a trusted CentOS Vault or
internal EL8 mirror first. Do not change the repositories on the
air-gapped VM just to download packages.

### 4.3 Download BIND and Its Dependencies

Run the following commands on the connected VM:

``` bash
mkdir -p /tmp/lab22-airgap/bind
cd /tmp/lab22-airgap/bind

dnf download \
  --resolve \
  --alldeps \
  --arch=x86_64,noarch \
  --destdir=/tmp/lab22-airgap/bind \
  bind
```

The `--resolve` and `--alldeps` options download the requested package
and its dependencies, including dependencies already installed on the
staging VM.

### 4.4 Verify the Downloaded RPM Files

Check that the BIND server package is present:

``` bash
ls -lh /tmp/lab22-airgap/bind/
ls /tmp/lab22-airgap/bind/bind-*.rpm
```

You should see a package similar to:

``` text
bind-9.11.26-6.el8.x86_64.rpm
```

The exact version may differ depending on the repository. Prefer a
consistent EL8 package set and verify package signatures before
installation.

> If the connected VM is running a newer OS, do not blindly download its
> default packages. Use compatible EL8 repositories or a matching CentOS
> 8 archive.

------------------------------------------------------------------------

## 5. Transfer the RPM Archive to the Air-Gapped VM

### 5.1 Create the RPM Archive

If the connected staging machine can reach `10.22.99.5` over SSH, run
this on the connected VM:

``` bash
tar -czf /tmp/lab22-bind-rpms.tar.gz \
  -C /tmp/lab22-airgap bind
```

### 5.2 Copy the Archive to the Jumpbox

``` bash
scp /tmp/lab22-bind-rpms.tar.gz \
  root@10.22.99.5:/tmp
```

If there is no network route between the machines, copy the archive
using the approved removable-media or offline-transfer process instead.

------------------------------------------------------------------------

## 6. Install BIND on `lab22-jumpbox`

Run the following commands on the air-gapped VM.

### 6.1 Extract and Inspect the RPM Files

``` bash
cd /tmp

tar -xzf lab22-bind-rpms.tar.gz

find /tmp/bind -maxdepth 1 -name '*.rpm' -type f
```

### 6.2 Install BIND from the Local RPM Files

The following command disables configured repositories and installs from
the RPM files copied to the VM:

``` bash
dnf install --disablerepo='*' \
  /tmp/bind/*.rpm
```

### 6.3 Verify the Installation

``` bash
rpm -q bind bind-utils
command -v named
named -V
```

------------------------------------------------------------------------

## 7. Configure the BIND DNS Server

### 7.1 Back Up the Existing Configuration

Create a backup before modifying the BIND configuration:

``` bash
cp -a /etc/named.conf /etc/named.conf.backup
```

Open the configuration file:

``` bash
vi /etc/named.conf
```

### 7.2 Configure the BIND Options

Inside the existing `options { ... };` block, ensure the following
settings are present. Modify existing settings rather than adding
duplicates.

"10.22.99.5" is my VM IP

``` conf
listen-on port 53 { 127.0.0.1; 10.22.99.5; };
allow-query { localhost; 10.22.99.0/24; };
recursion no;
```

### 7.3 Add the Authoritative DNS Zone

At the end of `/etc/named.conf`, add:

``` conf
zone "vks.lab22.sunfire.lab" IN {
    type master;
    file "vks.lab22.sunfire.lab.zone";
    allow-update { none; };
};
```

This configuration serves authoritative DNS records for the zone without
providing recursive DNS service.

------------------------------------------------------------------------

## 8. Create the DNS Zone File

The zone file contains the DNS records for the `lab22.sunfire.lab` zone,
including the name server and the manually configured OpenBao host
record.

### 8.1 Create the Zone File

``` bash
cat > /var/named/vks.lab22.sunfire.lab.zone <<'EOF'
$TTL 300
@ IN SOA ns1.vks.lab22.sunfire.lab. admin.vks.lab22.sunfire.lab. (
    2026102301
    3600
    900
    604800
    300
)

@       IN NS  ns1.vks.lab22.sunfire.lab.
ns1     IN A   10.22.99.5
EOF
```

### 8.2 Set File Ownership, Permissions, and SELinux Context

``` bash
chown root:named /var/named/vks.lab22.sunfire.lab.zone
chmod 640 /var/named/vks.lab22.sunfire.lab.zone
restorecon -v /var/named/vks.lab22.sunfire.lab.zone
```

------------------------------------------------------------------------

## 9. Validate and Start the DNS Service

### 9.1 Validate the BIND Configuration

Run the configuration checks before starting the service:

``` bash
named-checkconf

named-checkzone \
  vks.lab22.sunfire.lab \
  /var/named/vks.lab22.sunfire.lab.zone
```

### 9.2 Enable and Start BIND

If both validation commands succeed, enable and start the service:

``` bash
systemctl enable --now named
systemctl status named --no-pager
```

### 9.3 Firewall Note

If `firewalld` is enabled, add the required firewall rule. If it is
disabled, ignore this step.

------------------------------------------------------------------------

## 10. Test DNS Resolution

Run the following commands to test DNS resolution against the BIND
server:

``` bash
dig @10.22.99.5 ns1.vks.lab22.sunfirelab.com

```

Both records should return `10.22.99.5` if the queried names are
configured in the DNS zone being served.

### 10.1 Check BIND Logs if the Service Fails

``` bash
journalctl -u named -n 50 --no-pager
```

------------------------------------------------------------------------

## 11. Optional: Configure BIND to Forward Other DNS Queries

The initial BIND configuration uses `recursion no`. If BIND is
configured as the jumpbox's default DNS server, unrelated hostnames may
stop resolving.

If `10.0.254.5` is a reachable and approved upstream DNS server,
configure BIND to forward queries that it cannot answer.

### 11.1 Update the Existing Options Block

Edit `/etc/named.conf`:

``` bash
vi /etc/named.conf
```

Update the existing `options` block with:

``` conf
recursion yes;
allow-recursion { localhost; 10.22.99.0/24; };
allow-query-cache { localhost; 10.22.99.0/24; };

forwarders {
    10.0.254.5;                  #This is used to Forwarded to 10.0.254.5(global DNS) if domain name is not under our zone(lab22.sunfire.lab)
};
forward only;
```

Keep the existing `listen-on` and `allow-query` settings. If the
environment prohibits forwarding, leave recursion disabled and use a
dedicated internal DNS configuration instead.

### 11.2 Validate and Reload BIND

``` bash
named-checkconf
systemctl reload named
```

### 11.3 Test Internal and Upstream DNS Lookups

``` bash
dig @10.22.99.5 ns1.vks.lab22.sunfire.lab
dig @10.22.99.5 example.com                    #Use FQDN which doesn't contain "lab22.sunfire.lab"
```

The first query should return `10.22.99.5`. The second should resolve
only if the upstream DNS server is reachable and permits forwarding.

------------------------------------------------------------------------

## 12. Temporarily Configure the Jumpbox Resolver for Testing

Before changing the DNS resolver configuration, take a backup:

``` bash
# Back up the existing resolver configuration
cp -a /etc/resolv.conf /etc/resolv.conf.backup
```

A backup was taken and `/etc/resolve.conf` was updated with the
following entries to test BIND DNS. This is a temporary change. If
NetworkManager restarts, the file may return to its default contents.

``` conf
# Generated by NetworkManager
search vks.lab22.sunfirelab.com
nameserver 10.22.99.5
```

------------------------------------------------------------------------

## 13. Configure Dynamic DNS Using TSIG

Dynamic DNS allows an authorized client such as External-DNS to add or
update DNS records in the BIND zone. TSIG is used to authenticate update
requests.

### 13.1 Create the Key Directory and Generate a TSIG Key

``` bash
mkdir -p /etc/named/keys
tsig-keygen -a hmac-sha256 externaldns-key
```

Example output recorded during implementation:

``` conf
key "externaldns-key" {
        algorithm hmac-sha256;
        secret "RiF7fBcpYWFR8zzQdFMfPyO7UGUJwmstUDjatZ+g4cs=";
};
```

> **Security:** The example secret is reproduced from the supplied
> notes. Do not reuse this exposed value in a production environment.
> Generate a new key and keep the secret in an approved
> secret-management system.

### 13.2 Save the TSIG Key in a File

Add the generated key content to the following file:

``` bash
vi /etc/named/keys/externaldns.key
```

Set ownership and permissions:

``` bash
chmod 640 /etc/named/keys/externaldns.key
chown root:named /etc/named/keys/externaldns.key
```

The supplied notes also include this path check:

``` bash
ls -l /etc/named/externaldns.key
```

### 13.3 Include the TSIG Key in `named.conf`

Open the BIND configuration:

``` bash
vi /etc/named.conf
```

Add the following include statement:

``` conf
include "/etc/named/keys/externaldns.key";
```

### 13.4 Update the Zone Configuration to Permit Authenticated Updates

Update the existing zone section in `/etc/named.conf`:

``` conf
zone "vks.lab22.sunfire.lab" IN {
        type master;
        file "vks.lab22.sunfire.lab.zone";
        allow-update {
                key "externaldns-key";
        };
        allow-transfer {
                key "externaldns-key";
        };
};
```

### 13.5 Validate and Reload the BIND Configuration

``` bash
named-checkconf
systemctl reload named
systemctl status named
```

------------------------------------------------------------------------

## 14. Test Dynamic DNS Updates

### 14.1 Create a Test Update File

Create the test update file. The following entry uses test values:

``` bash
vi /tmp/test-update.txt
```

Add the following content:

``` text
server 10.22.99.5
zone vks.lab22.sunfire.lab
update add test.vks.lab22.sunfire.lab 300 A 10.11.90.109
send
```

### 14.2 Submit the Dynamic DNS Update

``` bash
nsupdate -k /etc/named/externaldns.key /tmp/test-update.txt
```

### 14.3 Verify the Record Using `nslookup`

``` bash
nslookup test.vks.lab22.sunfire.lab
```

Expected output recorded during implementation:

``` text
Server:         10.22.99.5
Address:        10.22.99.5#53

Name:   test.vks.lab22.sunfire.lab
Address: 10.11.90.109
```

### 14.4 Verify the Record Using `dig`

``` bash
dig @10.22.99.5 test.vks.lab22.sunfire.lab A
```

Expected output recorded during implementation:

``` text
; <<>> DiG 9.11.26-RedHat-9.11.26-6.el8 <<>> @10.22.99.5 test.vks.lab22.sunfire.lab A
; (1 server found)
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 27635
;; flags: qr aa rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 1, ADDITIONAL: 2

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1232
; COOKIE: ca26cbf60fe5e9819c8932256adee920be20c1c947461938 (good)
;; QUESTION SECTION:
;test.vks.lab22.sunfire.lab.                IN      A

;; ANSWER SECTION:
test.vks.lab22.sunfire.lab. 300     IN      A       10.11.90.109

;; AUTHORITY SECTION:
vks.lab22.sunfire.lab.      300     IN      NS      ns1.vks.lab22.sunfire.lab.

;; ADDITIONAL SECTION:
ns1.vks.lab22.sunfire.lab.  300     IN      A       10.22.99.5

;; Query time: 0 msec
;; SERVER: 10.22.99.5#53(10.22.99.5)
;; WHEN: Mon Oct 26 11:16:08 IST 2026
;; MSG SIZE  rcvd: 129
```

A successful response with `status: NOERROR` and the expected A record
confirms that the test record can be queried from the BIND server.

## 15. Configure BIND for cert-manager DNS01 using TSIG

We need addition key for "cert-manager" to Update/Delete TXT record for authorization of OpenBao Certificate Issuer.

This key will be used by ClusterIssuer/Issuer. 

Your existing DNS configuration:

| Parameter | Value | Description |
|---|---|---|
| DNS Server | `10.22.99.5` | BIND DNS server IP address |
| DNS Zone | `vks.lab22.sunfire.lab` | DNS zone where TXT records are created |
| Existing ExternalDNS TSIG Key | `externaldns-key` | Authentication key used by ExternalDNS |
| New cert-manager TSIG Key | `certmanager-acme` | Dedicated authentication key for cert-manager DNS-01 challenges |

We will retain the ExternalDNS key and add a separate TSIG key for cert-manager.

### 15.1 Generate the cert-manager TSIG key

On the BIND server:

tsig-keygen -a hmac-sha256 certmanager-acme
key "certmanager-acme" {
        algorithm hmac-sha256;
        secret "RzBsOraf4CB/NXNAoFG+GTShVMMO8wljV4GlzXCtIug=";
};
Save this output to:
/etc/named/keys/certmanager-acme.key

Protect the file:
chmod 640 /etc/named/keys/certmanager-acme.key
chown root:named /etc/named/keys/certmanager-acme.key

Use the actual group and directory permissions appropriate for your BIND installation.

### 15.2 Include both TSIG keys in named.conf
Make sure both keys are included:
```
include "/etc/named/keys/externaldns.key";
include "/etc/named/keys/certmanager-acme.key";
```
Keep your existing ExternalDNS key definition unchanged.
4.3 Update the zone's dynamic update policy
In your BIND zone configuration, use:
```
zone "lab22.sunfire.lab" IN {
    type master;
    file "vks.lab22.sunfire.lab.zone";

    update-policy {
        grant externaldns-key zonesub ANY;
        grant certmanager-acme zonesub TXT;                                          #Used by 'cert-manager' with only TXT update privilege 
    };

    allow-transfer {
        key "externaldns-key";
    };
};
```
This gives:
- ExternalDNS its existing zone-wide update permissions.
- cert-manager permission to add and remove ACME TXT record.

Important: allow-update and update-policy cannot both be specified in the same zone.

### 15.3 Validate and reload BIND
named-checkconf

named-checkzone vks.lab22.sunfire.lab \
  /var/named/vks.lab22.sunfire.lab.zone

Use the actual zone file path from your configuration.

If validation succeeds:
```
systemctl reload named
```
### 16 Test the BIND update before involving cert-manager

#### 16.1 Create a test update file:
```
cat >/tmp/test-update.txt <<'EOF'
server 10.22.99.5
zone vks.lab22.sunfire.lab
update add _acme-challenge.testing.vks.lab22.sunfire.lab 60 TXT "test"
send
EOF
```
Run:
```
nsupdate -k /etc/named/keys/certmanager-acme.key \
  /tmp/test-update.txt
```

Check the TXT record:
```
dig @10.22.99.5 TXT _acme-challenge.testing.vks.lab22.sunfire.lab

or duplicate the linux session and run below command 

watch -n 1 dig -k /etc/named/keys/externaldns.key @10.22.99.5 vks.lab22.sunfire.lab AXFR
```

#### 16.2 Delete the test record:
cat >/tmp/test-delete.txt <<'EOF'
server 10.22.99.5
zone vks.lab22.sunfire.lab
update delete _acme-challenge.testing.vks.lab22.sunfire.lab TXT
send
EOF
```
nsupdate -k /etc/named/keys/certmanager-acme.key \
  /tmp/test-delete.txt
```
Check the TXT record:
```
dig @10.22.99.5 TXT _acme-challenge.testing.vks.lab22.sunfire.lab

or duplicate the linux session and run below command 

watch -n 1 dig -k /etc/named/keys/externaldns.key @10.22.99.5 vks.vlab22.sunfire.lab AXFR
```
Verify it is gone: SERVFAIL means doesn't exist.

Do not proceed until the TSIG update and delete operations both work. cert-manager uses the same dynamic DNS mechanism. 
cert-manager Documentation

#### 16.2.3 Create the BIND TSIG secret
Extract the secret value from:
cat /etc/named/keys/certmanager-acme.key
Output:
```
key "certmanager-acme" {
        algorithm hmac-sha256;
        secret "RzBsOraf4CB/NXNAoFG+GTShVMMO8wljV4GlzXCtIug=";
};
```
It is the value inside the secret field.

Create the Kubernetes Secret inside Workload Cluster under "cert-manager" namespace :
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

We will refer this secret while creating "ClusterIssuer/Issuer" Objects

Sharing reference only related to ACME DNS integration : Please refer OpenBao Document for entire ClusterIssuer object definition
```yaml
spec:
  acme:
    solvers:
      - dns01:
          rfc2136:
            nameserver: 10.21.99.5:53        #BIND server IP
            tsigKeyName: certmanager-acme
            tsigAlgorithm: HMACSHA256
            tsigSecretSecretRef:
              name: certmanager-bind-tsig     #Refer the "certmanager-bind-tsig" secret used for ACME
              key: tsig-secret
```

------------------------------------------------------------------------

## 17. Integrate BIND with External-DNS on VKS

This section configures BIND to accept DNS queries from the VKS
External-DNS host-network range and shows the External-DNS RFC2136
settings.

### 17.1 Allow the VKS Host-Network Subnet in BIND

Update `/etc/named.conf`:

``` bash
vi /etc/named.conf
```

Add the VKS External-DNS host-network subnet `10.22.92.0/24` to
`allow-query`. This allows the VKS cluster to reach BIND DNS.

``` conf
allow-query { localhost; 10.22.99.0/24; 10.22.92.0/24; };
```

### 17.2 Retrieve the TSIG Key Content made for external DNS

The supplied implementation notes use the key file created earlier:

``` bash
cat /etc/named/keys/externaldns.key
```

Example output recorded in the notes:

``` conf
key "externaldns-key" {
        algorithm hmac-sha256;
        secret "RiF7fBcpYWFR8zzQdFMfPyO7UGUJwmstUDjatZ+g4cs=";
};
```

### 17.3 Create the TSIG Secret in the External-DNS Namespace

Use the TSIG secret value from the BIND key to create a Kubernetes
Secret in the External-DNS namespace:

``` bash
kubectl create secret generic externaldns-tsig \
  -n tanzu-system-service-discovery \
  --from-literal=tsig-secret='RiF7fBcpYWFR8zzQdFMfPyO7UGUJwmstUDjatZ+g4cs='
```

### 17.4 Configure External-DNS Deployment Arguments

Add the following arguments under the External-DNS deployment
configuration:

``` yaml
deployment:
  args:
    - --source=gateway-httproute
    - --provider=rfc2136                              #BIND using RFC2136 provider name
    - --registry=txt
    - --txt-owner-id=vks-external-dns-lab22           #use a different ID for independent instances
    - --domain-filter=vks.lab22.sunfire.lab               #it only updates entry with "vks.lab22.sunfire.lab" domain
    - --policy=upsert-only
    - --rfc2136-host=10.22.99.5                       #bind DNS IP
    - --rfc2136-port=53
    - --rfc2136-zone=vks.lab22.sunfire.lab                #zone name
    - --rfc2136-tsig-keyname=externaldns-key          #In bind DNS server externaldns.key has "externaldns-key" key variable set.
    - --rfc2136-tsig-secret=$(RFC2136_TSIG_SECRET)    #passing env variable
    - --rfc2136-tsig-secret-alg=hmac-sha256
    - --rfc2136-tsig-axfr                             #zone transfer flag
    - --rfc2136-min-ttl=300s                          #ttl set to 300s

  env:
    - name: RFC2136_TSIG_SECRET
      valueFrom:
        secretKeyRef:
          name: externaldns-tsig                      #
          key: tsig-secret
```

The configuration uses:

-   `gateway-httproute` as the source watched by External-DNS.
-   `rfc2136` as the DNS provider.
-   `lab22.sunfire.lab` as the domain filter and zone.
-   `10.22.99.5` as the BIND DNS server.


### BIND DNS final Configuration file

<details>
<summary><b>View named.conf – BIND DNS Server Configuration</b></summary>

```conf
options {
    listen-on port 53 { 127.0.0.1; 10.21.99.5; };
    listen-on-v6 port 53 { ::1; };

    directory "/var/named";

    dump-file "/var/named/data/cache_dump.db";
    statistics-file "/var/named/data/named_stats.txt";
    memstatistics-file "/var/named/data/named_mem_stats.txt";
    secroots-file "/var/named/data/named.secroots";
    recursing-file "/var/named/data/named.recursing";

    allow-query {
        localhost;
        10.21.99.0/24;
        10.21.92.0/24;
        10.0.254.5;
        10.0.0.0/8;
    };

    recursion yes;

    allow-recursion {
        localhost;
        10.21.99.0/24;
    };

    allow-query-cache {
        localhost;
        10.21.99.0/24;
    };

    forwarders {
        10.0.254.5;
    };

    forward only;

    dnssec-enable yes;
    dnssec-validation yes;

    managed-keys-directory "/var/named/dynamic";

    pid-file "/run/named/named.pid";
    session-keyfile "/run/named/session.key";

    include "/etc/crypto-policies/back-ends/bind.config";
};

logging {
    channel default_debug {
        file "data/named.run";
        severity dynamic;
    };
};

zone "." IN {
    type hint;
    file "named.ca";
};

include "/etc/named.rfc1912.zones";
include "/etc/named.root.key";

include "/etc/named/keys/externaldns.key";
include "/etc/named/keys/certmanager-acme.key";

zone "vks.lab21.sunfire.lab" IN {
    type master;
    file "vks.lab21.sunfire.lab.zone";

    update-policy {
        grant externaldns-key zonesub ANY;
        grant certmanager-acme zonesub TXT;
    };

    allow-transfer {
        key "externaldns-key";
    };
};
```

</details>

### Configuration Explanation

| Configuration | Purpose |
|---|---|
| `listen-on` | Defines the IPv4 addresses on which BIND listens for DNS requests. |
| `allow-query` | Controls which clients can query the DNS server. |
| `recursion yes` | Enables recursive DNS resolution. |
| `forwarders` | Forwards queries for external domains to the global DNS server. |
| `forward only` | Sends queries that are not answered locally to the configured forwarder. |
| `zone "vks.lab21.sunfire.lab"` | Defines the authoritative internal DNS zone. |
| `externaldns-key` | TSIG key used by ExternalDNS for dynamic DNS updates. |
| `certmanager-acme` | TSIG key used by cert-manager for DNS-01 TXT record updates. |
| `update-policy` | Restricts ExternalDNS and cert-manager to their authorized DNS update operations. |
| `allow-transfer` | Restricts zone transfers to clients possessing the ExternalDNS TSIG key. |

**Important:** Keep the actual TSIG key contents in separate protected files. Include only the `include` statements in your README, never the private key material.
-   The Kubernetes Secret `externaldns-tsig` to provide the TSIG secret
    through the `RFC2136_TSIG_SECRET` environment variable.
-   `vks-external-dns-lab22` as the TXT registry owner ID.

------------------------------------------------------------------------

