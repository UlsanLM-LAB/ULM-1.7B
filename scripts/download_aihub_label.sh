#!/usr/bin/env bash
set -euo pipefail

# AI Hub 경상도 방언(datasetkey=119) 라벨링데이터 전용 다운로드 도우미 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/private/aihub"
DATASET_KEY="119" # 한국어 방언 발화(경상도)

echo "========================================================="
echo "  ULM-1.7B: AI Hub 경상도 방언 라벨링데이터 다운로더"
echo "========================================================="
echo "※ 사전 필수: AI Hub(aihub.or.kr) 웹사이트에 로그인하여"
echo "  '한국어 방언 발화(경상도)' 데이터셋 페이지에서"
echo "  [다운로드] 신청(약관 동의)을 1회 먼저 완료하셔야 합니다."
echo "  (URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=119)"
echo "---------------------------------------------------------"

# 1. aihubshell 다운로드 및 준비
AIHUBSHELL="${SCRIPT_DIR}/aihubshell"
if ! command -v aihubshell &>/dev/null && [ ! -f "${AIHUBSHELL}" ]; then
    echo "[1/4] aihubshell 도구를 다운로드합니다..."
    curl -sS -o "${AIHUBSHELL}" "https://api.aihub.or.kr/api/aihubshell.do"
    chmod +x "${AIHUBSHELL}"
fi

SHELL_CMD="aihubshell"
if ! command -v aihubshell &>/dev/null; then
    SHELL_CMD="${AIHUBSHELL}"
fi

# 2. API Key 확인 및 입력
if [ -z "${AIHUB_APIKEY:-}" ]; then
    echo ""
    echo "※ AI Hub 최신 aihubshell(v0.6)은 'API Key' 인증 방식을 사용합니다."
    echo "  1) aihub.or.kr 로그인 -> [마이페이지] -> [API Key 발급/관리] 에서 키 복사"
    echo "  2) https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=119 에서 [다운로드] 1회 클릭 (약관 승인)"
    echo ""
    read -r -p "AI Hub API Key (aihubapikey): " AIHUB_APIKEY
    export AIHUB_APIKEY
fi

# 3. 라벨링 데이터 파일 목록 조회
echo ""
echo "[2/4] 경상도 방언(119) 파일 목록을 조회합니다..."
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

LIST_OUTPUT=$("${SHELL_CMD}" -mode l -datasetkey "${DATASET_KEY}")
echo "${LIST_OUTPUT}"

echo "---------------------------------------------------------"
echo "[3/4] 라벨링데이터(JSON) 다운로드 진행"
echo "텍스트 학습에는 음성 원천데이터(수백 GB)가 필요 없으며,"
echo "라벨링데이터(JSON)인 '572701,572713' (총 약 340MB)만 받으시면 됩니다."
echo "---------------------------------------------------------"

read -r -p "다운로드할 filekey [기본값: 572701,572713]: " FILE_KEYS
FILE_KEYS="${FILE_KEYS:-572701,572713}"

if [ -n "${FILE_KEYS}" ]; then
    echo "[4/4] 다운로드를 시작합니다 (대상 디렉토리: ${TARGET_DIR})..."
    "${SHELL_CMD}" -mode d -datasetkey "${DATASET_KEY}" -filekey "${FILE_KEYS}" -aihubapikey "${AIHUB_APIKEY}"
    
    echo ""
    echo "다운로드 파일 확인 및 zip 압축 해제 중..."
    FOUND_ZIP=0
    while IFS= read -r -d '' zipfile; do
        FOUND_ZIP=1
        echo "압축 해제 중: ${zipfile}"
        unzip -q -o "${zipfile}" -d "$(dirname "${zipfile}")/unzipped"
    done < <(find "${TARGET_DIR}" -type f -name "*.zip" -print0)

    if [ "${FOUND_ZIP}" -eq 1 ]; then
        echo "========================================================="
        echo "✓ 라벨링 데이터 배치가 완료되었습니다! (${TARGET_DIR})"
        echo "  다음 단계: 'python scripts/audit_aihub.py --raw-dir data/private/aihub' 실행"
        echo "========================================================="
    else
        echo "경고: 압축 해제할 zip 파일이 생성되지 않았습니다."
        echo "다운로드 오류 메시지(인증/권한 실패 등)를 확인해주세요."
    fi
else
    echo "다운로드가 취소되었습니다."
fi
