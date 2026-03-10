#!/bin/bash

# =============================================================================
# SMACv2 Orphan Process Cleaner
# =============================================================================
# 학습 중에도 안전하게 사용 가능:
#   - 학습 중인 프로세스 트리(python main.py → SC2_x64 등)는 보호
#   - 고아(orphan) SC2_x64, Xvfb, xvfb-run 프로세스만 종료
#
# 사용법:
#   bash scripts/kill_smacv2.sh          # 고아 프로세스만 종료 (학습 보호)
#   bash scripts/kill_smacv2.sh --all    # 모든 관련 프로세스 강제 종료
# =============================================================================

set -euo pipefail

MODE="${1:-safe}"  # default: safe mode

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== SMACv2 Process Cleaner ===${NC}"
echo ""

# --all 모드: 기존처럼 모든 프로세스 강제 종료
if [[ "$MODE" == "--all" ]]; then
    echo -e "${RED}[MODE] 전체 종료 모드 (--all)${NC}"
    echo -e "${RED}  학습 중인 프로세스를 포함하여 모든 관련 프로세스를 종료합니다.${NC}"
    echo ""

    for pattern in "SC2_x64" "python main.py" "xvfb-run" "Xvfb"; do
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            count=$(pgrep -cf "$pattern" 2>/dev/null || echo 0)
            echo -e "${YELLOW}[KILL] ${pattern} 프로세스 ${count}개 종료 중...${NC}"
            pkill -9 -f "$pattern" 2>/dev/null || true
            echo -e "${GREEN}[OK]   종료 완료.${NC}"
        else
            echo -e "${GREEN}[INFO] ${pattern} 프로세스 없음.${NC}"
        fi
    done

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}전체 종료 완료.${NC}"
    exit 0
fi

# =============================================================================
# Safe 모드: 학습 중인 프로세스를 보호하고 고아 프로세스만 종료
# =============================================================================
echo -e "${GREEN}[MODE] 안전 모드 (학습 보호)${NC}"
echo -e "  학습 중인 프로세스는 보호하고, 고아 프로세스만 종료합니다."
echo ""

# --- Step 1: 학습 중인 프로세스 트리의 PID 수집 ---
# "python main.py"를 실행 중인 메인 프로세스를 찾고,
# 그 조상(xvfb-run 등)과 자손(worker, SC2_x64 등)을 모두 보호 목록에 추가
declare -A PROTECTED_PIDS

collect_tree_pids() {
    local pid=$1
    if [[ -z "$pid" ]] || [[ -n "${PROTECTED_PIDS[$pid]+_}" ]]; then
        return
    fi
    # PID가 실제로 존재하는지 확인
    if ! kill -0 "$pid" 2>/dev/null; then
        return
    fi
    PROTECTED_PIDS[$pid]=1
    # 자식 프로세스 재귀 탐색
    local children
    children=$(ps --ppid "$pid" -o pid= 2>/dev/null || true)
    for child in $children; do
        collect_tree_pids "$child"
    done
}

collect_ancestors() {
    local pid=$1
    while true; do
        local ppid
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [[ -z "$ppid" ]] || [[ "$ppid" == "1" ]] || [[ "$ppid" == "0" ]]; then
            break
        fi
        PROTECTED_PIDS[$ppid]=1
        pid=$ppid
    done
}

# "python main.py"를 커맨드라인에 포함하는 메인 프로세스 찾기
# (메인 프로세스만 찾기 위해, PPID가 xvfb-run/bash 등인 것을 기준으로 함)
training_pids=$(pgrep -f "python main.py" 2>/dev/null || true)

if [[ -n "$training_pids" ]]; then
    training_count=0
    for pid in $training_pids; do
        # 프로세스 트리 전체를 보호 목록에 추가
        collect_tree_pids "$pid"
        # 조상 프로세스도 보호 (xvfb-run, Xvfb 등)
        collect_ancestors "$pid"
        training_count=$((training_count + 1))
    done
    echo -e "${GREEN}[보호] 학습 관련 프로세스 ${#PROTECTED_PIDS[@]}개를 보호 목록에 등록했습니다.${NC}"
    echo -e "  (python main.py 프로세스 ${training_count}개 기반 트리)"
else
    echo -e "${YELLOW}[INFO] 실행 중인 학습 프로세스(python main.py)가 없습니다.${NC}"
fi

echo ""

# --- Step 2: 고아 프로세스 종료 ---
killed_count=0

kill_orphans() {
    local pattern=$1
    local label=$2
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)

    if [[ -z "$pids" ]]; then
        echo -e "${GREEN}[INFO] 고아 ${label} 프로세스 없음.${NC}"
        return
    fi

    local orphan_count=0
    local protected_count=0
    for pid in $pids; do
        if [[ -n "${PROTECTED_PIDS[$pid]+_}" ]]; then
            protected_count=$((protected_count + 1))
        else
            orphan_count=$((orphan_count + 1))
            echo -e "${YELLOW}[KILL] 고아 ${label} (PID: ${pid}) 종료${NC}"
            kill -9 "$pid" 2>/dev/null || true
            killed_count=$((killed_count + 1))
        fi
    done

    if [[ $protected_count -gt 0 ]]; then
        echo -e "${GREEN}[보호] ${label} ${protected_count}개 보호됨 (학습 중)${NC}"
    fi
    if [[ $orphan_count -eq 0 ]] && [[ $protected_count -gt 0 ]]; then
        echo -e "${GREEN}[INFO] 모든 ${label} 프로세스가 학습 중이므로 종료하지 않았습니다.${NC}"
    fi
}

# SC2_x64 고아 프로세스
kill_orphans "SC2_x64" "StarCraft II"

# xvfb-run 고아 프로세스
kill_orphans "xvfb-run" "xvfb-run"

# Xvfb 고아 프로세스
kill_orphans "Xvfb" "Xvfb"

# 부모 없는 python main.py 고아 프로세스 (PPID=1인 경우)
orphan_python_pids=$(pgrep -f "python main.py" 2>/dev/null || true)
if [[ -n "$orphan_python_pids" ]]; then
    for pid in $orphan_python_pids; do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "")
        if [[ "$ppid" == "1" ]]; then
            echo -e "${YELLOW}[KILL] 고아 python main.py (PID: ${pid}, PPID=1) 종료${NC}"
            # 자식도 함께 종료
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
            killed_count=$((killed_count + 1))
        fi
    done
fi

echo ""
echo -e "${CYAN}========================================${NC}"
if [[ $killed_count -gt 0 ]]; then
    echo -e "${GREEN}정리 완료: 고아 프로세스 ${killed_count}개 종료됨.${NC}"
else
    echo -e "${GREEN}정리 완료: 종료할 고아 프로세스가 없습니다.${NC}"
fi

# 남은 프로세스 현황 출력
remaining=$(ps aux | grep -E "SC2_x64|python main.py|xvfb" | grep -v grep || true)
if [[ -n "$remaining" ]]; then
    echo ""
    echo -e "${CYAN}[현황] 남아있는 관련 프로세스:${NC}"
    echo "$remaining" | awk '{printf "  PID=%-8s CPU=%-6s MEM=%-6s %s\n", $2, $3, $4, $11}'
fi
echo ""
