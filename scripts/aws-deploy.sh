#!/bin/bash
# aws-deploy.sh — Manage llm-katan EC2 instances.
#
# Deploys llm-katan on Amazon Linux 2023 (t3a.micro) with:
#   - HTTP on port 8000 (direct)
#   - HTTPS on port 443 (Caddy + Let's Encrypt auto-cert via sslip.io)
#   - All 5 providers: openai, anthropic, vertexai, bedrock, azure_openai
#   - API key validation enabled
#
# Usage:
#   ./scripts/aws-deploy.sh list                    # Show all instances
#   ./scripts/aws-deploy.sh create                  # Launch new instance
#   ./scripts/aws-deploy.sh setup <ip>              # Deploy llm-katan + TLS on existing instance
#   ./scripts/aws-deploy.sh setup-tls <ip>          # Add Let's Encrypt TLS to existing instance
#   ./scripts/aws-deploy.sh delete <ip>             # Terminate instance + release EIP
#   ./scripts/aws-deploy.sh test <ip>               # Run health checks
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - SSH key at ~/.ssh/llm-katan-key.pem
#   - Key pair "llm-katan-key" in us-east-2

set -euo pipefail

# ─── Config ────────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-2}"
AMI="ami-0b0b78dcacbab728f"         # Amazon Linux 2023 (x86_64)
INSTANCE_TYPE="t3a.micro"
KEY_NAME="llm-katan-key"
KEY_FILE="${LLM_KATAN_KEY:-$HOME/.ssh/llm-katan-key.pem}"
SG_NAME="llm-katan-sg"
TAG_KEY="project"
TAG_VALUE="llm-katan"
VOLUME_SIZE=16                       # GB, gp3
LLM_KATAN_VERSION="${LLM_KATAN_VERSION:-}"  # empty = latest from pypi

SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=10"

# ─── Colors ────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1" >&2; }

# ─── Helpers ───────────────────────────────────────────────────────────────

check_prereqs() {
  local missing=()
  for cmd in aws ssh jq curl; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    fail "Missing: ${missing[*]}"
    exit 1
  fi
  if [[ ! -f "$KEY_FILE" ]]; then
    fail "SSH key not found: $KEY_FILE"
    exit 1
  fi
  # Verify AWS credentials
  if ! aws sts get-caller-identity --region "$REGION" &>/dev/null; then
    fail "AWS credentials not configured. Run: aws configure"
    exit 1
  fi
}

get_sg_id() {
  local sg_id
  sg_id=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [[ "$sg_id" == "None" || -z "$sg_id" ]]; then
    echo ""
  else
    echo "$sg_id"
  fi
}

ensure_security_group() {
  local sg_id
  sg_id=$(get_sg_id)
  if [[ -n "$sg_id" ]]; then
    echo "$sg_id"
    # Ensure port 80 is open (needed for Let's Encrypt HTTP-01 challenge)
    if ! aws ec2 describe-security-groups --group-ids "$sg_id" --region "$REGION" \
        --query 'SecurityGroups[0].IpPermissions[?FromPort==`80`]' --output text 2>/dev/null | grep -q "80"; then
      aws ec2 authorize-security-group-ingress --group-id "$sg_id" --region "$REGION" \
        --protocol tcp --port 80 --cidr 0.0.0.0/0 2>/dev/null || true
      ok "Added port 80 to security group (Let's Encrypt)"
    fi
    return
  fi

  # Create new security group
  local vpc_id
  vpc_id=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null)

  sg_id=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$SG_NAME" \
    --description "llm-katan test server" \
    --vpc-id "$vpc_id" \
    --query 'GroupId' --output text)

  for port in 22 80 443 8000; do
    aws ec2 authorize-security-group-ingress --group-id "$sg_id" --region "$REGION" \
      --protocol tcp --port "$port" --cidr 0.0.0.0/0
  done

  aws ec2 create-tags --region "$REGION" --resources "$sg_id" \
    --tags "Key=$TAG_KEY,Value=$TAG_VALUE"

  ok "Created security group $sg_id ($SG_NAME)"
  echo "$sg_id"
}

wait_for_ssh() {
  local ip="$1" max_wait="${2:-120}"
  local elapsed=0
  echo -n "  Waiting for SSH on $ip "
  while [[ $elapsed -lt $max_wait ]]; do
    if ssh $SSH_OPTS "ec2-user@$ip" "echo ok" &>/dev/null; then
      echo ""
      ok "SSH ready"
      return 0
    fi
    echo -n "."
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo ""
  fail "SSH timeout after ${max_wait}s"
  return 1
}

get_fqdn() {
  local ip="$1"
  # Use sslip.io for DNS — resolves X-X-X-X.sslip.io to X.X.X.X.
  # Let's Encrypt issues certs for sslip.io subdomains (unlike *.amazonaws.com
  # which is blocked by LE policy).
  local dashed="${ip//./-}"
  echo "${dashed}.sslip.io"
}

# ─── Remote setup scripts ─────────────────────────────────────────────────

# Runs on the remote instance to install llm-katan + Caddy
remote_setup_script() {
  local fqdn="$1"
  local version_pin="$2"
  cat <<'REMOTE_SCRIPT'
#!/bin/bash
set -euo pipefail

echo ">>> Installing llm-katan..."
if ! command -v llm-katan &>/dev/null; then
  yum install -y python3.11 python3.11-pip
REMOTE_SCRIPT

  # Version pin handling
  if [[ -n "$version_pin" ]]; then
    echo "  pip3.11 install 'llm-katan==$version_pin'"
  else
    echo "  pip3.11 install llm-katan"
  fi

  cat <<'REMOTE_SCRIPT'
else
  echo "  llm-katan already installed: $(llm-katan --version)"
fi

echo ">>> Configuring llm-katan systemd service..."
cat > /etc/systemd/system/llm-katan.service <<'SVC'
[Unit]
Description=LLM Katan Test Server
After=network.target

[Service]
ExecStart=/usr/local/bin/llm-katan --model llm-katan-echo --backend echo --validate-keys --providers openai,anthropic,vertexai,bedrock,azure_openai
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable llm-katan
systemctl restart llm-katan

echo ">>> Configuring journal size cap..."
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/size-limit.conf <<'JRNL'
[Journal]
SystemMaxUse=50M
JRNL
systemctl restart systemd-journald

echo ">>> Installing Caddy..."
if ! command -v caddy &>/dev/null; then
  CADDY_VER="2.9.1"
  curl -sL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VER}/caddy_${CADDY_VER}_linux_amd64.tar.gz" \
    | tar xz -C /usr/local/bin caddy
  chmod +x /usr/local/bin/caddy
  # Create caddy user and dirs
  useradd --system --home /var/lib/caddy --shell /usr/sbin/nologin caddy 2>/dev/null || true
  mkdir -p /etc/caddy /var/lib/caddy/.local/share/caddy /var/log/caddy
  chown -R caddy:caddy /var/lib/caddy /var/log/caddy
  # systemd unit
  cat > /etc/systemd/system/caddy.service <<'CSVC'
[Unit]
Description=Caddy
After=network.target

[Service]
User=caddy
Group=caddy
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
Restart=on-failure
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
CSVC
  systemctl daemon-reload
fi

REMOTE_SCRIPT

  # Caddy config with the actual FQDN
  cat <<REMOTE_SCRIPT
echo ">>> Configuring Caddy for TLS (${fqdn})..."
cat > /etc/caddy/Caddyfile <<CADDY
${fqdn} {
    reverse_proxy localhost:8000
}
CADDY

REMOTE_SCRIPT

  cat <<'REMOTE_SCRIPT'
systemctl enable caddy
systemctl restart caddy

# Wait for Caddy to get the cert (up to 60s)
echo ">>> Waiting for Let's Encrypt certificate..."
for i in $(seq 1 12); do
  if curl -sf --max-time 5 https://localhost:8443/health &>/dev/null; then
    echo "  TLS certificate obtained!"
    break
  fi
  sleep 5
done

echo ">>> Setup complete!"
echo "  HTTP:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/health"
echo "  HTTPS: https://$(curl -s http://169.254.169.254/latest/meta-data/public-hostname)/health"
REMOTE_SCRIPT
}

# ─── Commands ──────────────────────────────────────────────────────────────

cmd_list() {
  echo -e "${BOLD}llm-katan EC2 instances (${REGION})${NC}"
  echo ""

  local instances
  instances=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" "Name=instance-state-name,Values=running,stopped" \
    --query 'Reservations[].Instances[] | sort_by(@, &LaunchTime)' 2>/dev/null)

  if [[ "$(echo "$instances" | jq length)" == "0" ]]; then
    # Fallback: find by security group
    local sg_id
    sg_id=$(get_sg_id)
    if [[ -n "$sg_id" ]]; then
      instances=$(aws ec2 describe-instances --region "$REGION" \
        --filters "Name=instance.group-id,Values=$sg_id" "Name=instance-state-name,Values=running,stopped" \
        --query 'Reservations[].Instances[] | sort_by(@, &LaunchTime)' 2>/dev/null)
    fi
  fi

  local count
  count=$(echo "$instances" | jq length)
  if [[ "$count" == "0" ]]; then
    echo "  No instances found."
    return
  fi

  printf "  %-18s %-22s %-12s %-8s %-40s\n" "IP" "Instance ID" "Type" "State" "FQDN"
  printf "  %-18s %-22s %-12s %-8s %-40s\n" "──────────────" "────────────────────" "──────────" "──────" "────────────────────────────────────"

  echo "$instances" | jq -r '.[] | [.PublicIpAddress // "N/A", .InstanceId, .InstanceType, .State.Name] | @tsv' | \
  while IFS=$'\t' read -r ip iid itype state; do
    local fqdn=""
    if [[ "$ip" != "N/A" ]]; then
      fqdn=$(get_fqdn "$ip")
    fi
    printf "  %-18s %-22s %-12s %-8s %-40s\n" "$ip" "$iid" "$itype" "$state" "$fqdn"
  done

  echo ""

  # Check TLS status for each running instance
  echo -e "  ${BOLD}TLS status:${NC}"
  echo "$instances" | jq -r '.[] | select(.State.Name == "running") | .PublicIpAddress // empty' | \
  while read -r ip; do
    if [[ -z "$ip" ]]; then continue; fi
    local fqdn
    fqdn=$(get_fqdn "$ip")
    local issuer
    issuer=$(echo | openssl s_client -connect "$ip:443" -servername "$fqdn" 2>/dev/null | \
      openssl x509 -noout -issuer 2>/dev/null | sed 's/issuer=//' || echo "unreachable")
    if echo "$issuer" | grep -qi "let's encrypt\|R[0-9]\|E[0-9]\|ISRG"; then
      ok "$ip — Let's Encrypt ✓"
    else
      warn "$ip — $issuer"
    fi
  done
}

cmd_create() {
  echo -e "${BOLD}Creating new llm-katan instance${NC}"
  echo ""

  local sg_id
  sg_id=$(ensure_security_group)
  ok "Security group: $sg_id"

  echo "  Launching EC2 instance..."
  local instance_id
  instance_id=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$sg_id" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llm-katan-test},{Key=$TAG_KEY,Value=$TAG_VALUE},{Key=architecture,Value=x86_64}]" \
    --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled" \
    --query 'Instances[0].InstanceId' --output text)

  ok "Instance: $instance_id"

  echo "  Waiting for instance to be running..."
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$instance_id"
  ok "Instance running"

  # Allocate and associate Elastic IP
  echo "  Allocating Elastic IP..."
  local alloc_id
  alloc_id=$(aws ec2 allocate-address --region "$REGION" \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=$TAG_KEY,Value=$TAG_VALUE},{Key=Name,Value=llm-katan}]" \
    --query 'AllocationId' --output text)

  aws ec2 associate-address --region "$REGION" \
    --instance-id "$instance_id" \
    --allocation-id "$alloc_id" >/dev/null

  local ip
  ip=$(aws ec2 describe-addresses --region "$REGION" \
    --allocation-ids "$alloc_id" \
    --query 'Addresses[0].PublicIp' --output text)

  ok "Elastic IP: $ip"

  # Wait for SSH and run full setup
  wait_for_ssh "$ip"
  cmd_setup "$ip"

  echo ""
  echo -e "${GREEN}${BOLD}Instance ready!${NC}"
  echo "  IP:    $ip"
  echo "  FQDN:  $(get_fqdn "$ip")"
  echo "  SSH:   ssh -i $KEY_FILE ec2-user@$ip"
  echo "  HTTP:  http://$ip:8000/health"
  echo "  HTTPS: https://$(get_fqdn "$ip")/health"
}

cmd_setup() {
  local ip="$1"
  echo -e "${BOLD}Setting up llm-katan on $ip${NC}"
  echo ""

  local fqdn
  fqdn=$(get_fqdn "$ip")
  ok "FQDN: $fqdn"

  # Ensure port 80 is open in security group
  ensure_security_group >/dev/null

  echo "  Running remote setup..."
  remote_setup_script "$fqdn" "$LLM_KATAN_VERSION" | ssh $SSH_OPTS "ec2-user@$ip" "sudo bash -s" 2>&1 | \
    sed 's/^/  /'

  echo ""
  cmd_test "$ip"
}

cmd_setup_tls() {
  local ip="$1"
  echo -e "${BOLD}Setting up Let's Encrypt TLS on $ip${NC}"
  echo ""

  local fqdn
  fqdn=$(get_fqdn "$ip")
  ok "FQDN: $fqdn"

  # Ensure port 80 is open
  ensure_security_group >/dev/null

  # Generate just the Caddy setup portion
  ssh $SSH_OPTS "ec2-user@$ip" "sudo bash -s" <<SETUP_TLS 2>&1 | sed 's/^/  /'
set -euo pipefail

echo ">>> Installing Caddy..."
if ! command -v caddy &>/dev/null; then
  CADDY_VER="2.9.1"
  curl -sL "https://github.com/caddyserver/caddy/releases/download/v\${CADDY_VER}/caddy_\${CADDY_VER}_linux_amd64.tar.gz" \
    | tar xz -C /usr/local/bin caddy
  chmod +x /usr/local/bin/caddy
  useradd --system --home /var/lib/caddy --shell /usr/sbin/nologin caddy 2>/dev/null || true
  mkdir -p /etc/caddy /var/lib/caddy/.local/share/caddy /var/log/caddy
  chown -R caddy:caddy /var/lib/caddy /var/log/caddy
  cat > /etc/systemd/system/caddy.service <<'CSVC'
[Unit]
Description=Caddy
After=network.target

[Service]
User=caddy
Group=caddy
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
Restart=on-failure
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
CSVC
  systemctl daemon-reload
fi

echo ">>> Configuring Caddy for ${fqdn}..."
cat > /etc/caddy/Caddyfile <<'CADDY'
${fqdn} {
    reverse_proxy localhost:8000
}
CADDY

# Stop old self-signed TLS service
systemctl stop llm-katan-tls 2>/dev/null || true
systemctl disable llm-katan-tls 2>/dev/null || true

systemctl enable caddy
systemctl restart caddy

echo ">>> Waiting for Let's Encrypt certificate..."
for i in \$(seq 1 12); do
  if curl -sf --max-time 5 https://localhost:8443/health &>/dev/null; then
    echo "  TLS certificate obtained!"
    break
  fi
  sleep 5
done

echo ">>> Done!"
SETUP_TLS

  echo ""
  cmd_test "$ip"
}

cmd_delete() {
  local ip="$1"
  echo -e "${BOLD}Deleting instance at $ip${NC}"
  echo ""

  # Find instance ID by IP (try direct lookup, then EIP)
  local instance_id
  instance_id=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=public-ip-address,Values=$ip" "Name=instance-state-name,Values=running,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "")

  if [[ -z "$instance_id" || "$instance_id" == "None" ]]; then
    instance_id=$(aws ec2 describe-addresses --region "$REGION" \
      --filters "Name=public-ip,Values=$ip" \
      --query 'Addresses[0].InstanceId' --output text 2>/dev/null || echo "")
  fi

  if [[ -z "$instance_id" || "$instance_id" == "None" ]]; then
    fail "No instance found for IP $ip"
    exit 1
  fi

  # Find and release Elastic IP
  local alloc_id
  alloc_id=$(aws ec2 describe-addresses --region "$REGION" \
    --filters "Name=instance-id,Values=$instance_id" \
    --query 'Addresses[0].AllocationId' --output text 2>/dev/null || echo "")

  if [[ -n "$alloc_id" && "$alloc_id" != "None" ]]; then
    local assoc_id
    assoc_id=$(aws ec2 describe-addresses --region "$REGION" \
      --allocation-ids "$alloc_id" \
      --query 'Addresses[0].AssociationId' --output text 2>/dev/null || echo "")
    if [[ -n "$assoc_id" && "$assoc_id" != "None" ]]; then
      aws ec2 disassociate-address --region "$REGION" --association-id "$assoc_id"
    fi
    aws ec2 release-address --region "$REGION" --allocation-id "$alloc_id"
    ok "Released Elastic IP $ip"
  fi

  # Terminate instance
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$instance_id" >/dev/null
  ok "Terminated instance $instance_id"

  echo "  Waiting for termination..."
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$instance_id"
  ok "Instance terminated"
}

cmd_test() {
  local ip="$1"
  local fqdn
  fqdn=$(get_fqdn "$ip")

  echo -e "${BOLD}Testing $ip ($fqdn)${NC}"
  echo ""

  # HTTP health
  local http_health
  http_health=$(curl -sf --max-time 5 "http://$ip:8000/health" 2>/dev/null || echo "")
  if [[ "$http_health" == *"ok"* ]]; then
    ok "HTTP  :8000/health — $(echo "$http_health" | jq -c '{status,model,providers}' 2>/dev/null || echo "$http_health")"
  else
    fail "HTTP  :8000/health — ${http_health:-no response}"
  fi

  # HTTPS health (Let's Encrypt via sslip.io)
  local https_health
  https_health=$(curl -sf --max-time 10 "https://$fqdn/health" 2>/dev/null || echo "")
  if [[ "$https_health" == *"ok"* ]]; then
    ok "HTTPS :443 (Let's Encrypt) — verified ✓"
  else
    fail "HTTPS :443 — ${https_health:-no response}"
  fi

  # Quick inference test
  local response
  response=$(curl -sf --max-time 10 "http://$ip:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer llm-katan-openai-key" \
    -d '{"model":"llm-katan-echo","messages":[{"role":"user","content":"health check"}],"max_tokens":5}' 2>/dev/null || echo "")
  if [[ "$response" == *"choices"* ]]; then
    ok "Inference: OpenAI chat completions working"
  else
    fail "Inference: ${response:-no response}"
  fi
}

# ─── Main ──────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command> [args]"
  echo ""
  echo "Commands:"
  echo "  list              Show all llm-katan instances"
  echo "  create            Launch new instance (full setup)"
  echo "  setup <ip>        Deploy llm-katan + TLS on existing instance"
  echo "  setup-tls <ip>    Add Let's Encrypt TLS to existing instance"
  echo "  delete <ip>       Terminate instance + release EIP"
  echo "  test <ip>         Run health checks"
  exit 1
fi

check_prereqs

CMD="$1"; shift

case "$CMD" in
  list)       cmd_list ;;
  create)     cmd_create ;;
  setup)      [[ $# -ge 1 ]] || { fail "Usage: $0 setup <ip>"; exit 1; }; cmd_setup "$1" ;;
  setup-tls)  [[ $# -ge 1 ]] || { fail "Usage: $0 setup-tls <ip>"; exit 1; }; cmd_setup_tls "$1" ;;
  delete)     [[ $# -ge 1 ]] || { fail "Usage: $0 delete <ip>"; exit 1; }; cmd_delete "$1" ;;
  test)       [[ $# -ge 1 ]] || { fail "Usage: $0 test <ip>"; exit 1; }; cmd_test "$1" ;;
  *)          fail "Unknown command: $CMD"; exit 1 ;;
esac
