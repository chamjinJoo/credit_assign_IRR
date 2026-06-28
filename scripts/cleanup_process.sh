#!/bin/bash

# =============================================================================
# Generic Training Process Cleaner
# =============================================================================
# Legacy filename is kept for convenience, but this cleaner now supports:
#   - StagHunt
#   - SMAX
#   - SMACv2
#
# Safe mode (default):
#   - protects active training trees
#   - cleans orphaned training roots (PPID=1) and leftover helper processes
#
# All mode:
#   - terminates all selected training trees
#   - then removes leftover helper processes
#
# Usage:
#   bash scripts/kill_smacv2.sh
#   bash scripts/kill_smacv2.sh --env smacv2
#   bash scripts/kill_smacv2.sh --env staghunt,smax --dry-run
#   bash scripts/kill_smacv2.sh --all --env all
# =============================================================================

set -euo pipefail

MODE="safe"
DRY_RUN=false

SUPPORTED_DOMAINS=("staghunt" "smax" "smacv2")

declare -A SELECTED_DOMAINS=()
declare -A PROTECTED_PIDS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/kill_smacv2.sh [--safe|--all] [--env DOMAIN[,DOMAIN...]] [--dry-run]

Options:
  --safe        Protect active training trees and clean orphaned leftovers only.
  --all         Terminate all selected training trees and then clean leftovers.
  --env         Target domain(s): staghunt, smax, smacv2, all
  --dry-run     Show what would be terminated without sending signals.
  -h, --help    Show this help message.
EOF
}

normalize_domain() {
    local raw
    raw=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$raw" in
        staghunt|staghunt)
            printf 'staghunt\n'
            ;;
        smax)
            printf 'smax\n'
            ;;
        smacv2)
            printf 'smacv2\n'
            ;;
        all)
            printf 'all\n'
            ;;
        *)
            return 1
            ;;
    esac
}

add_domain() {
    local domain="$1"
    if [[ "$domain" == "all" ]]; then
        for item in "${SUPPORTED_DOMAINS[@]}"; do
            SELECTED_DOMAINS["$item"]=1
        done
        return
    fi
    SELECTED_DOMAINS["$domain"]=1
}

parse_domain_arg() {
    local value="$1"
    local normalized
    IFS=',' read -r -a items <<< "$value"
    for item in "${items[@]}"; do
        item=$(printf '%s' "$item" | xargs)
        [[ -z "$item" ]] && continue
        if ! normalized=$(normalize_domain "$item"); then
            echo -e "${RED}[ERROR] 지원하지 않는 도메인: ${item}${NC}"
            exit 1
        fi
        add_domain "$normalized"
    done
}

for arg in "$@"; do
    case "$arg" in
        --safe)
            MODE="safe"
            ;;
        --all)
            MODE="all"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
    esac
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --safe|--all|--dry-run)
            shift
            ;;
        --env)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}[ERROR] --env 다음에 도메인을 지정해주세요.${NC}"
                exit 1
            fi
            parse_domain_arg "$2"
            shift 2
            ;;
        -h|--help)
            shift
            ;;
        *)
            echo -e "${RED}[ERROR] 알 수 없는 옵션: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

if [[ ${#SELECTED_DOMAINS[@]} -eq 0 ]]; then
    add_domain "all"
fi

domain_enabled() {
    local domain
    domain=$(normalize_domain "$1") || return 1
    [[ -n "${SELECTED_DOMAINS[$domain]+_}" ]]
}

selected_domains_string() {
    local parts=()
    for domain in "${SUPPORTED_DOMAINS[@]}"; do
        if [[ -n "${SELECTED_DOMAINS[$domain]+_}" ]]; then
            parts+=("$domain")
        fi
    done
    local joined=""
    local item
    for item in "${parts[@]}"; do
        if [[ -z "$joined" ]]; then
            joined="$item"
        else
            joined="${joined}, ${item}"
        fi
    done
    printf '%s\n' "$joined"
}

pid_exists() {
    kill -0 "$1" 2>/dev/null
}

get_ppid() {
    ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '
}

get_cmd() {
    ps -o args= -p "$1" 2>/dev/null
}

extract_env_from_cmd() {
    local cmd="$1"
    if [[ "$cmd" =~ (^|[[:space:]])--env[[:space:]]+(StagHunt|SMAX|SMACv2)([[:space:]]|$) ]]; then
        printf '%s\n' "${BASH_REMATCH[2]}"
        return 0
    fi
    return 1
}

matches_selected_training_pid() {
    local pid="$1"
    local cmd env

    cmd=$(get_cmd "$pid")
    [[ -z "$cmd" ]] && return 1
    [[ "$cmd" != *"main.py"* ]] && return 1
    [[ "$cmd" != *"--env"* ]] && return 1

    if ! env=$(extract_env_from_cmd "$cmd"); then
        return 1
    fi

    case "$env" in
        StagHunt)
            domain_enabled "staghunt"
            ;;
        SMAX)
            domain_enabled "smax"
            ;;
        SMACv2)
            domain_enabled "smacv2"
            ;;
        *)
            return 1
            ;;
    esac
}

collect_tree_pids() {
    local pid="$1"
    local set_name="$2"
    local -n out="$set_name"

    if [[ -z "$pid" ]] || [[ -n "${out[$pid]+_}" ]]; then
        return
    fi
    if ! pid_exists "$pid"; then
        return
    fi

    out["$pid"]=1

    local child
    while read -r child; do
        [[ -z "$child" ]] && continue
        collect_tree_pids "$child" "$set_name"
    done < <(ps --ppid "$pid" -o pid= 2>/dev/null || true)
}

collect_ancestor_pids() {
    local pid="$1"
    local set_name="$2"
    local -n out="$set_name"

    while true; do
        local ppid
        ppid=$(get_ppid "$pid")
        if [[ -z "$ppid" ]] || [[ "$ppid" == "0" ]] || [[ "$ppid" == "1" ]]; then
            break
        fi
        out["$ppid"]=1
        pid="$ppid"
    done
}

declare -a MATCHED_MAIN_PIDS=()

discover_training_pids() {
    MATCHED_MAIN_PIDS=()
    local pid
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if matches_selected_training_pid "$pid"; then
            MATCHED_MAIN_PIDS+=("$pid")
        fi
    done < <(pgrep -f "main.py" 2>/dev/null || true)
}

terminate_pid_set() {
    local label="$1"
    local set_name="$2"
    local -n pid_set="$set_name"
    local -a targets=()
    local pid

    for pid in "${!pid_set[@]}"; do
        if pid_exists "$pid"; then
            targets+=("$pid")
        fi
    done

    if [[ ${#targets[@]} -eq 0 ]]; then
        return 0
    fi

    printf '%b[%s] %s %d개%b\n' "$YELLOW" "$([[ "$DRY_RUN" == true ]] && echo "DRY-RUN" || echo "KILL")" "$label" "${#targets[@]}" "$NC"
    printf '  PIDs: %s\n' "${targets[*]}"

    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    kill -TERM "${targets[@]}" 2>/dev/null || true
    sleep 2

    local -a survivors=()
    for pid in "${targets[@]}"; do
        if pid_exists "$pid"; then
            survivors+=("$pid")
        fi
    done

    if [[ ${#survivors[@]} -gt 0 ]]; then
        printf '%b[KILL] 종료되지 않은 %s %d개에 SIGKILL 전송%b\n' "$RED" "$label" "${#survivors[@]}" "$NC"
        printf '  PIDs: %s\n' "${survivors[*]}"
        kill -KILL "${survivors[@]}" 2>/dev/null || true
    fi
}

kill_orphan_helpers() {
    local pattern="$1"
    local label="$2"
    local -A targets=()
    local -a matches=()
    local pid

    while read -r pid; do
        [[ -z "$pid" ]] && continue
        matches+=("$pid")
    done < <(pgrep -f "$pattern" 2>/dev/null || true)

    if [[ ${#matches[@]} -eq 0 ]]; then
        printf '%b[INFO] %s 프로세스 없음.%b\n' "$GREEN" "$label" "$NC"
        return
    fi

    local protected_count=0
    for pid in "${matches[@]}"; do
        if [[ -n "${PROTECTED_PIDS[$pid]+_}" ]]; then
            protected_count=$((protected_count + 1))
            continue
        fi

        local ppid
        ppid=$(get_ppid "$pid")
        if [[ "$ppid" == "1" ]] || [[ -z "$ppid" ]]; then
            targets["$pid"]=1
        fi
    done

    if [[ ${#targets[@]} -eq 0 ]]; then
        if [[ $protected_count -gt 0 ]]; then
            printf '%b[INFO] %s %d개는 활성 학습 트리에 속해 보호했습니다.%b\n' "$GREEN" "$label" "$protected_count" "$NC"
        else
            printf '%b[INFO] 종료할 orphan %s 프로세스가 없습니다.%b\n' "$GREEN" "$label" "$NC"
        fi
        return
    fi

    if [[ $protected_count -gt 0 ]]; then
        printf '%b[보호] %s %d개는 활성 학습 트리에 속해 유지합니다.%b\n' "$GREEN" "$label" "$protected_count" "$NC"
    fi
    terminate_pid_set "orphan ${label}" targets
}

report_remaining_processes() {
    echo ""
    echo -e "${CYAN}[현황] 남아있는 선택 도메인 관련 프로세스${NC}"

    local found=false
    local pid cmd env
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if matches_selected_training_pid "$pid"; then
            cmd=$(get_cmd "$pid")
            env=$(extract_env_from_cmd "$cmd" || true)
            printf '  PID=%-8s PPID=%-8s ENV=%-8s %s\n' "$pid" "$(get_ppid "$pid")" "$env" "$cmd"
            found=true
        fi
    done < <(pgrep -f "main.py" 2>/dev/null || true)

    if domain_enabled "smacv2"; then
        while read -r pid; do
            [[ -z "$pid" ]] && continue
            if [[ -z "${PROTECTED_PIDS[$pid]+_}" ]] && [[ "$(get_ppid "$pid")" != "1" ]]; then
                continue
            fi
            printf '  PID=%-8s PPID=%-8s AUX=%-8s %s\n' "$pid" "$(get_ppid "$pid")" "SC2_x64" "$(get_cmd "$pid")"
            found=true
        done < <(pgrep -x SC2_x64 2>/dev/null || true)
    fi

    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if [[ -z "${PROTECTED_PIDS[$pid]+_}" ]] && [[ "$(get_ppid "$pid")" != "1" ]]; then
            continue
        fi
        printf '  PID=%-8s PPID=%-8s AUX=%-8s %s\n' "$pid" "$(get_ppid "$pid")" "wandb-core" "$(get_cmd "$pid")"
        found=true
    done < <(pgrep -f "wandb-core" 2>/dev/null || true)

    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if [[ -z "${PROTECTED_PIDS[$pid]+_}" ]] && [[ "$(get_ppid "$pid")" != "1" ]]; then
            continue
        fi
        printf '  PID=%-8s PPID=%-8s AUX=%-8s %s\n' "$pid" "$(get_ppid "$pid")" "gpu_stats" "$(get_cmd "$pid")"
        found=true
    done < <(pgrep -f "gpu_stats" 2>/dev/null || true)

    if [[ "$found" == false ]]; then
        echo "  없음"
    fi
}

echo -e "${CYAN}=== Generic Training Process Cleaner ===${NC}"
echo -e "  target domains: $(selected_domains_string)"
if [[ "$MODE" == "all" ]]; then
    echo -e "  mode: ${RED}all${NC}"
else
    echo -e "  mode: ${GREEN}safe${NC}"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo -e "  dry-run: ${YELLOW}enabled${NC}"
fi
echo ""

discover_training_pids

if [[ "$MODE" == "all" ]]; then
    declare -A ALL_TARGETS=()

    if [[ ${#MATCHED_MAIN_PIDS[@]} -eq 0 ]]; then
        echo -e "${GREEN}[INFO] 선택 도메인에 해당하는 training main.py 프로세스가 없습니다.${NC}"
    else
        for pid in "${MATCHED_MAIN_PIDS[@]}"; do
            collect_tree_pids "$pid" ALL_TARGETS
        done
        terminate_pid_set "selected training tree" ALL_TARGETS
    fi
else
    declare -A ORPHAN_TARGETS=()
    active_count=0
    orphan_count=0

    if [[ ${#MATCHED_MAIN_PIDS[@]} -eq 0 ]]; then
        echo -e "${GREEN}[INFO] 선택 도메인에 해당하는 training main.py 프로세스가 없습니다.${NC}"
    else
        for pid in "${MATCHED_MAIN_PIDS[@]}"; do
            ppid=$(get_ppid "$pid")
            if [[ "$ppid" == "1" ]]; then
                collect_tree_pids "$pid" ORPHAN_TARGETS
                orphan_count=$((orphan_count + 1))
            else
                collect_tree_pids "$pid" PROTECTED_PIDS
                collect_ancestor_pids "$pid" PROTECTED_PIDS
                active_count=$((active_count + 1))
            fi
        done
        if [[ $active_count -gt 0 ]]; then
            echo -e "${GREEN}[보호] 활성 학습 프로세스 ${active_count}개 기반 트리를 보호했습니다.${NC}"
            echo -e "       보호 PID 수: ${#PROTECTED_PIDS[@]}"
        else
            echo -e "${YELLOW}[INFO] 보호할 활성 학습 트리가 없습니다.${NC}"
        fi
        if [[ $orphan_count -gt 0 ]]; then
            terminate_pid_set "orphan training tree" ORPHAN_TARGETS
        else
            echo -e "${GREEN}[INFO] 종료할 orphan training main.py 프로세스가 없습니다.${NC}"
        fi
    fi
fi

kill_orphan_helpers "wandb-core" "wandb-core"
kill_orphan_helpers "gpu_stats" "gpu_stats"

if domain_enabled "smacv2"; then
    kill_orphan_helpers "SC2_x64" "StarCraft II"
    kill_orphan_helpers "xvfb-run" "xvfb-run"
    kill_orphan_helpers "Xvfb" "Xvfb"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${GREEN}dry-run 완료.${NC}"
else
    echo -e "${GREEN}정리 작업이 완료되었습니다.${NC}"
fi

report_remaining_processes
echo ""
