# 저장소 동기화·정리 기록

작업일: 2026-09-29. 사용자 요청에 따라 원격 상태 확인, 병합 브랜치 정리, 로컬 변경 보존, 문서 정리와 상세 리뷰를 진행했습니다.

**동기화**

- 시작 main: `717e346074a2b33e6bded9043e8404fd7c0e549b`.
- fetch 후 원격 main: `a6fbfc1cf7ac943baf197810a2bb4e1b04e19217`.
- 로컬은 4개 커밋 뒤처져 있었으며 `git merge --ff-only origin/main`으로 반영했습니다. 강도 제어 API, 모델 비교, README, 메타데이터 정리가 포함되어 있었습니다.
- 이전 작업 메모는 원격 PR #6에서 이미 삭제되어 있었으므로 main에 되살리지 않았습니다. 새 작업 메모는 로컬 Git 제외 대상으로 유지했습니다.

**삭제한 원격 브랜치**

| 브랜치 | 삭제 전 tip | 근거 |
| --- | --- | --- |
| `chore/repo-cleanup-20260929` | `d05aa9a27d34595e6cfb9625b56337ce7be674ef` | PR #6 병합 결과와 tree 동일 |
| `docs/release-readme-references` | `286bbf82e958f99a4cd2527b1a21f4a91f0bd382` | PR #5 병합 결과와 tree 동일 |
| `feat/dialect-strength-api` | `c111eacd0a9765df4f327764314848aae755976a` | PR #4 병합 결과와 tree 동일 |
| `benchmark-comparison-20260928` | `89dc5b9e66feafae6421b1ff73e59a443a4603ad` | PR #3 병합 결과와 tree 동일 |
| `ulm4b-arm-b-plus` | `50f6021afc5510bbed6036b63336ef835e5882e1` | PR #2 병합 결과와 tree 동일 |
| `release/ulm4b-arm-b` | `b5a983c5847b2795b03970af3e876b21daf3583d` | PR #1 병합 결과와 tree 동일 |
| `backup/dialect-alignment-v1` | `cedc6b4b308c819e416c91357459a490c66bf313` | main의 조상 커밋 |
| `backup/recovery-v3-report` | `cedc6b4b308c819e416c91357459a490c66bf313` | main의 조상 커밋 |

PR은 squash merge였으므로 단순 `--merged` 판정만 사용하지 않았습니다. 각 원격 tip과 PR 병합 커밋의 tree가 같은지 확인하고, 병합 커밋이 main에 포함된 것을 검증했습니다. 삭제 시 tip별 lease와 atomic push를 사용했습니다.

원격에 남긴 브랜치는 `main`, `context-repair-v1-runtime`, `instruction-recovery-v2`입니다. 후자의 두 연구 브랜치는 이번에 병합 완료가 입증되지 않아 보존했습니다.

**로컬 브랜치와 worktree**

- `feat/dialect-strength-api`, `benchmark-comparison-20260928`: 로컬 tip이 검증된 원격 tip의 조상이고 worktree에 미커밋 변경이 없음을 확인한 뒤 브랜치와 worktree를 제거했습니다. 남아 있던 Git 제외 파일은 재생성 가능한 Python/pytest/Ruff 캐시였습니다.
- `ulm4b-arm-b-plus`: 병합 완료 브랜치는 삭제하고 worktree는 detached HEAD `50f6021`로 유지했습니다. `outputs/`와 원본 예측·로그 등 실험 산출물이 있으므로 디렉터리를 제거하지 않았습니다.
- `instruction-recovery-v2`: 브랜치와 worktree를 그대로 유지했습니다.
- `wip/local-experiments-20260929`: 기존 미커밋/미추적 파일 11개를 커밋 `460e839`에 보존했습니다. 각 파일의 SHA-256을 백업과 대조했습니다. 이 브랜치는 로컬에만 있으며 데이터·서비스 파일의 공개 여부를 새로 판단하지 않았습니다.
- 보존이 확인된 뒤 이번 작업에서 만든 임시 stash만 제거했습니다.

**백업과 복구**

작업 머신의 백업 디렉터리:

```text
/home/lee/.local/share/ulm-repo-backups/20260929-011243/
  refs.bundle                 # 삭제 전 전체 Git refs와 객체
  refs.txt                    # 삭제 전 ref 목록
  deleted-remote-refs.json     # 삭제한 원격 브랜치와 tip
  worktree-changes.tar.gz      # 기존 미커밋/미추적 파일 11개
  manifest.json               # 보존 파일 SHA-256
  status.txt                  # 시작 Git 상태
```

`git bundle verify`는 통과했습니다. bundle과 실험 파일 백업은 공개 저장소에 올리지 않았습니다. 예를 들어 삭제 브랜치를 복구하려면 해당 머신에서 다음과 같이 별도 브랜치로 가져올 수 있습니다.

```bash
git fetch /home/lee/.local/share/ulm-repo-backups/20260929-011243/refs.bundle \
  refs/remotes/origin/feat/dialect-strength-api:refs/heads/restored/dialect-strength-api
```

로컬 실험을 이어갈 위치는 `/home/lee/ULM-1.7B-local-experiments`입니다. 기존의 대형 `data/private/`, `.venv/`, `outputs/`는 보존했습니다. 디스크 용량만을 이유로 데이터나 가중치를 삭제하지 않았습니다.

**공개 저장소 정리**

- [scripts/README.md](../scripts/README.md): 현재 API·CLI·학습·평가와 과거 EC2 실험 진입점 안내.
- [reports/README.md](README.md): 현재 릴리스 후보, 모델 비교, 후속 실험과 운영 기록 구분.
- [PROJECT_REVIEW.md](PROJECT_REVIEW.md): 최신 코드의 상세 검토, 재현 근거와 수정 우선순위.
- README: 전체 테스트에 필요한 `tts` extra, 핵심 소스 lint 명령, 상세 리뷰 링크 추가.
- `.gitignore`: 모델 비교 원본 예측·검수 대기열과 로컬 `tasks/` 제외. 기존 집계 결과와 연구 보고서는 유지.

학습·추론·평가 소스의 동작은 이번 정리에서 변경하지 않았습니다. 상세 리뷰의 결함 수정과 실제 GPU 재평가는 후속 작업입니다.
