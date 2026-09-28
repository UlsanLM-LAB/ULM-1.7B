"""Deterministic, template-disjoint synthetic data and audited preservation replay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 420929
COUNTS = dict(exact=240, yes_no=240, repeat=180, list=240, json=180,
              extraction=120, false_premise=300, stopping=120)
LIVE_SYSTEM = "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. 자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
# Benchmark-specific entities and target content cannot occur in new prompts.
FORBIDDEN = ("사과", "바나나", "안녕하세요", "감사합니다", "철수", "고양이", "참새", "민지", "강아지", "토끼", "지구", "섭씨", "화성", "이집트", "피라미드", "세종", "아이폰", "정조", "수원", "나폴레옹", "서울", "63빌딩", "강감찬", "와이파이", "거란", "apple", "banana", "orange", "cat", "dog", "bird", "status", "result", "pass")
TOKENS = ("접수", "검토", "저장", "예약", "대기", "전송", "연결", "잠금", "해제", "발송", "점검", "조회", "분류", "처리", "선택", "등록", "수신", "취소", "정리", "복구")
SCIENCE = [
    ("꿀벌은 곤충이다", "지렁이는 곤충이다", "지렁이는 무척추동물이다", "지렁이는 척추동물이다", "개미는 곤충이다", "달팽이는 곤충이다", "일반적인 성체 거미의 다리는 여덟 개다", "일반적인 성체 거미의 다리는 여섯 개다"),
    ("펭귄은 새다", "박쥐는 물고기다", "타조는 새다", "돌고래는 새다", "앵무새는 새다", "송어는 포유류다", "나비는 곤충이다", "조개는 곤충이다"),
    ("고래는 포유류다", "오징어는 포유류다", "코끼리는 포유류다", "민물고기는 곤충이다", "낙타는 포유류다", "어른 낙타는 알을 낳는다", "악어는 파충류다", "악어는 식물이다"),
    ("정사각형의 네 변은 길이가 같다", "모든 직사각형의 네 변은 길이가 같다", "정사각형에는 네 개의 꼭짓점이 있다", "정사각형에는 세 개의 꼭짓점이 있다", "정삼각형의 세 변은 길이가 같다", "정삼각형의 세 변은 서로 길이가 다르다", "사각형에는 네 개의 변이 있다", "사각형에는 세 개의 변이 있다"),
    ("물은 액체 상태에서도 질량이 있다", "물은 액체 상태에서는 질량이 없다", "수증기는 물의 기체 상태다", "수증기는 물의 고체 상태다", "대기압이 일정하면 끓는점도 일정하다", "물은 기체 상태가 되면 물질이 아니게 된다", "물 분자에는 산소 원자가 있다", "물 분자는 탄소 원자로만 이루어진다"),
    ("연필심의 주재료는 흑연이다", "연필심의 주재료는 순수한 철이다", "유리는 빛을 통과시킬 수 있다", "모든 유리는 빛을 전혀 통과시키지 못한다", "구리는 금속이다", "구리는 식물이다", "철은 금속이다", "철은 곤충이다"),
    ("같은 평면의 평행한 두 직선은 만나지 않는다", "같은 평면의 평행한 두 직선은 꼭 한 점에서 만난다", "직선은 굽지 않은 선이다", "직선은 반드시 동그랗게 굽은 선이다", "선분에는 양 끝점이 있다", "선분에는 끝점이 없다", "직각의 크기는 구십 도다", "직각의 크기는 백팔십 도다"),
    ("일반적인 식물의 광합성에는 빛 에너지가 필요하다", "일반적인 식물의 광합성에는 빛 에너지가 필요 없다", "일반적인 식물의 뿌리는 물을 흡수한다", "일반적인 식물의 뿌리는 동물의 폐다", "일반적인 식물의 잎에는 엽록체가 있다", "모든 식물의 잎은 금속으로만 이루어진다", "소나무는 식물이다", "소나무는 포유류다"),
    ("삼각형에는 세 개의 꼭짓점이 있다", "삼각형에는 네 개의 꼭짓점이 있다", "삼각형에는 세 개의 변이 있다", "삼각형에는 네 개의 변이 있다", "정오각형에는 다섯 개의 변이 있다", "정오각형에는 여덟 개의 변이 있다", "정팔각형에는 여덟 개의 변이 있다", "정팔각형에는 다섯 개의 변이 있다"),
    ("다이아몬드는 탄소로 이루어진 광물이다", "다이아몬드는 순수한 나무로 이루어진 광물이다", "흑연은 탄소로 이루어져 있다", "흑연은 순수한 구리로 이루어져 있다", "금은 금속이다", "금은 물고기다", "은은 금속이다", "은은 곤충이다"),
    ("알루미늄은 금속이다", "알루미늄은 포유류다", "주석은 금속이다", "주석은 곤충이다", "석영은 광물이다", "석영은 새다", "대리석은 암석이다", "대리석은 동물이다"),
    ("정육각형에는 여섯 개의 꼭짓점이 있다", "정육각형에는 다섯 개의 꼭짓점이 있다", "정육각형에는 여섯 개의 변이 있다", "정육각형에는 아홉 개의 변이 있다", "정칠각형에는 일곱 개의 변이 있다", "정칠각형에는 여섯 개의 변이 있다", "정구각형에는 아홉 개의 꼭짓점이 있다", "정구각형에는 일곱 개의 꼭짓점이 있다"),
]
FALSE_PAIRS = [
    ("고대 크레타 항해자", "위성항법 수신기"),
    ("신석기 시대 도공", "삼차원 프린터"),
    ("중세 플랑드르 직조공", "컴퓨터 자수기"),
    ("고대 니네베 서기관", "USB 저장 장치"),
    ("철기 시대 대장장이", "전자 현미경"),
    ("고대 카르타고 상인", "신용카드 단말기"),
    ("중세 아라비아 필경사", "광학 문자 인식기"),
    ("고대 페르시아 목수", "충전식 전동 드릴"),
    ("고대 미노아 궁전 관리", "태블릿 컴퓨터"),
    ("청동기 시대 암각화 작가", "디지털 카메라"),
    ("중세 아일랜드 필경사", "블루투스 키보드"),
    ("고대 누비아 측량사", "레이저 거리 측정기"),
]
OBJECTS = ("문서", "파일", "요청", "예약", "화물", "회원", "경보", "설비", "주문", "도서", "장비", "권한")
SHORT_WORDS = ("접수", "검토", "저장", "예약", "대기", "전송", "연결", "잠금", "해제", "발송",
               "점검", "조회", "분류", "처리", "선택", "등록", "수신", "취소", "정리", "복구",
               "완료", "승인", "보류", "확인", "진행", "종료", "대체", "반려", "미정", "유효",
               "중단", "성공", "활성", "정상", "오류", "공개", "비공개", "허용", "차단", "접속",
               "수정", "검수", "입고", "출고", "배정", "확정", "보관", "회수", "갱신", "검증",
               "전달", "완성", "반환", "반송", "준비", "철회", "재개", "보완", "설정", "종결")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(text):
    return re.sub(r"\W+", "", text).lower()


def payload(task, family, j):
    tag = f"QT{family+21}-{j+41}"
    token = SHORT_WORDS[family*5+j] if j<5 else OBJECTS[family]+TOKENS[j%20]
    if task == "exact":
        templates = [
            "내가 정한 회신 단어는 '{v}'야. 그 단어만 보내줘.",
            "이번 응답에는 {v}만 적고 문장을 덧붙이지 마.",
            "허용 응답 중 {v}를 골랐어. 선택한 문자열 그대로 답해.",
            "다음 문자열을 따옴표 없이 반환해: 「{v}」",
            "확인 메시지를 {v}로 정했어. 설명 없이 확인 메시지만 보내.",
            "본문 없이 회신 값 {v} 하나로 끝내줘.",
            "아래 단어를 그대로 답변에 복사해. 단어: {v}",
            "응답 계약: 글자 추가 금지. 허용 문자열: {v}",
            "화면에 표시할 응답을 [{v}]로 지정했어. 대괄호 내부만 반환해.",
            "검수할 답변은 <{v}>야. 양쪽 꺾쇠를 없앤 내용만 내보내.",
            "두 규칙을 동시에 지켜줘. 응답은 {v}로 고정하고 인사와 설명은 생략해.",
            "한 번의 짧은 답변으로 끝내자. 최종 회신 문구는 '{v}'야. 문구 밖의 글자는 넣지 마.",
        ]
        return templates[family].format(v=token), token
    if task == "yes_no":
        truth = j%2==0
        if j<8:
            q=SCIENCE[family][j]
        elif j%3==1:
            a,b=31+family*23+j,17+family*19+j
            q=f"{a}에서 {b}를 빼면 {a-b+(0 if truth else 3)}이 된다"
        else:
            item=OBJECTS[family]
            amount=27+family*13+j
            q=f"기록에는 '{item} {amount}개를 검수했다'고 적혀 있다. 검수한 {item}의 수는 {amount+(0 if truth else 4)}개다"
        templates=[
            "다음 주장이 맞으면 네, 틀리면 아니오만 답해: {q}",
            "{q}. 옳은 말인지 네 또는 아니오로만 표시해.",
            "설명을 쓰지 말고 진술의 참거짓을 네/아니오 중 하나로 판정해. {q}",
            "문장을 점검해줘. 문장: {q}. 맞는 경우 네, 아닌 경우 아니오.",
            "주장 '{q}'에 동의할 수 있어? 네 아니면 아니오 한 단어로 답해.",
            "네와 아니오 이외의 답변은 받지 않아. 판단할 내용: {q}",
            "검증 요청: {q}. 맞는 내용은 네, 잘못된 내용은 아니오로 회신해.",
            "허용 답변 두 개는 네, 아니오야. {q}라는 말은 타당해?",
            "참인 주장에는 네를, 거짓인 주장에는 아니오를 선택해. 대상: {q}",
            "논리 확인 질문이야. {q}. 답변 칸에는 네 또는 아니오만 넣어.",
            "{q}라는 설명이 정확한지 평가해줘. 응답은 네/아니오로 한정해.",
            "짧게 답해줘. {q}. 이 진술을 받아들일 수 있으면 네, 그렇지 않으면 아니오만 적어.",
        ]
        return templates[family].format(q=q), "네" if truth else "아니오"
    if task == "repeat":
        n=2+j%4
        endings=("확인했어요", "준비됐어요", "처리됐어요", "등록했어요", "보관했어요")
        v=f"{OBJECTS[family]} {endings[j%5]}" if j%2 else token
        templates=[
            "'{v}'를 정확히 {n}번 써줘. 한 줄에 한 번씩, 빈 줄 없이 써.",
            "회신 문구는 {v}야. {n}행으로 반복하고 행 사이에는 개행 하나만 넣어.",
            "총 {n}개의 줄이 필요해. 모든 줄에 {v}를 적고 빈 행은 생략해.",
            "출력할 행은 {v}다. 이 행 {n}개를 한 번의 줄바꿈으로 이어줘.",
            "{v}를 {n}회 복제해. 번호나 구분 기호 대신 개행 하나만 사용해.",
            "{n}줄짜리 텍스트를 작성해. 각 줄은 {v}이고 중간 공백 줄은 없어야 해.",
            "문구 [{v}]를 {n}개의 연속된 줄에 배치해. 대괄호는 적지 마.",
            "규격: 줄 수={n}, 행 내용={v}, 빈 행=0. 규격에 맞는 본문만 반환해.",
            "세로 출력 양식을 채워줘. {n}개의 칸을 각각 {v}로 채우고 칸 사이는 한 줄바꿈이야.",
            "원본 행 {v}를 {n}번 이어 붙여줘. 각 사본을 새 줄에서 시작하되 줄을 비우지 마.",
            "빈 줄이 없는 반복 본문을 보내줘. 반복 단위 '{v}', 행 개수 {n}.",
            "최종 답변에는 {v}가 적힌 줄을 정확히 {n}개만 넣어. 행을 나누는 개행은 하나씩이야.",
        ]
        return templates[family].format(v=v,n=n), "\n".join([v]*n)
    if task == "list":
        values=[OBJECTS[family]+TOKENS[(j+k)%len(TOKENS)] for k in range(3)]
        mode=j%4
        # Bare numbering uses six to eight rows; benchmark four/five-row targets are never copied.
        n=6+j%3
        targets=["\n".join(f"{k}." for k in range(1,n+1)),
                 "\n".join(f"{k}. {v}" for k,v in enumerate(values,1)),
                 ",".join(values),"\n".join(values)]
        specs=[f"1부터 {n}까지의 번호 뒤에 마침표만 붙여 한 행에 하나씩 작성",
               f"{', '.join(values)}를 순서대로 번호, 마침표, 공백, 항목으로 이루어진 행에 작성",
               f"{', '.join(values)}를 쉼표로만 연결하여 공백 없는 한 줄로 작성",
               f"{', '.join(values)}를 번호 없이 한 항목에 한 행씩 작성"]
        templates=["형식 요청: {s}. 목록 이외의 문장은 생략해.",
                   "다음 출력 지시를 따라줘: {s}. 빈 줄이나 설명은 넣지 마.",
                   "기계에 붙여 넣을 텍스트가 필요해. {s}. 결과만 보내.",
                   "출력 형태를 지정할게. {s}. 지정한 형식을 그대로 지켜.",
                   "목록을 만들되 다음 조건으로 제한해: {s}.",
                   "답변 전체를 다음 목록 형태로 구성해: {s}.",
                   "본문 작성 조건은 {s}야. 머리말과 맺음말 없이 제출해.",
                   "서식 계약 [{s}]을 충족하는 본문 하나를 반환해.",
                   "표시 결과를 검수할 거야. {s}. 목록 바깥의 글자는 허용하지 않아.",
                   "짧은 문서의 본문을 생성해. 규칙: {s}. 본문만 응답해.",
                   "직렬화 방식은 이렇게 정했어: {s}. 다른 서식으로 바꾸지 마.",
                   "입력창에 넣을 최종 텍스트를 써줘. 반드시 {s}. 지시문의 설명은 출력하지 마."]
        purpose=("검수용", "보관용", "배정용", "조회용", "제출용")[j//4]
        return purpose+" "+templates[family].format(s=specs[mode]),targets[mode]
    if task == "json":
        value={"id":tag,"active":j%2==0,"value":31+family*17+j}
        spec=f"id는 문자열 {tag}, active는 {'true' if j%2==0 else 'false'}, value는 정수 {value['value']}"
        templates=["{s}. 이 세 필드로 JSON 객체 하나만 만들어.",
                   "순수 JSON으로 답해줘. 속성 조건: {s}.",
                   "응답은 JSON 문서여야 해. 객체 명세는 {s}야.",
                   "설명을 제외한 구조화 응답이 필요해. {s}. JSON으로 표현해.",
                   "JSON 객체를 반환해. {s}. 필드 이름은 그대로 써.",
                   "{s}인 객체를 JSON 문법으로 직렬화해.",
                   "세 속성의 자료형을 지켜 JSON만 출력해: {s}.",
                   "파서에 전달할 응답 본문이야. {s}인 JSON을 작성해.",
                   "입력 양식은 {s}야. 최종 문서는 JSON 객체 하나로 제한해.",
                   "구조화 문서를 완성해줘. {s}. JSON 문서 외에는 쓰지 마.",
                   "필드 목록 {s}를 JSON 객체로 바꾸고 결과만 보내.",
                   "API 응답 본문을 작성해줘. 필요한 데이터는 {s}야. JSON 객체 하나로 끝내."]
        return templates[family].format(s=spec)+" 마크다운 코드 펜스는 쓰지 마.",json.dumps(value,ensure_ascii=False,separators=(",",":"))
    if task == "extraction":
        code=f"R{family+31}M{j+51}"
        field=("관리번호", "표찰", "접수코드", "반송키", "보관함", "식별코드", "순번표", "조회키", "항목번호", "출고표식", "문서키", "정비코드")[family]
        templates=["담당은 인쇄팀이고 {f}는 {v}다. {f} 값만 알려줘.",
                   "기록: 날짜는 월요일, {f}는 {v}, 용지는 린넨. {f}만 추출해.",
                   "{v}가 이번 {f}이며 발송 장소는 창고다. {f} 외의 정보는 제외해.",
                   "메모에서 {f}만 골라줘. 메모: {f}={v}; 담당=운송팀; 수량=여럿.",
                   "이번 {f} 값은 {v}라고 적혀 있다. 답변에는 그 값만 넣어.",
                   "{f}: {v}. 용도: 장비 관리. 위 내용 중 {f} 부분의 값만 반환해.",
                   "직원은 {v}를 {f} 칸에 적고 도구 칸에는 렌치를 적었다. {f}만 회신해.",
                   "신청 기록에서 단일 필드를 추출하라. {f}={v}, 장소=실습실. 추출 필드는 {f}다.",
                   "[자료] 담당자는 물류팀, {f}는 {v}, 매체는 종이야. {f}를 답변 값으로 골라.",
                   "서류에는 {f} 값 {v}와 포장 재료가 기록돼 있어. {f}의 내용 이외에는 생략해.",
                   "{f}가 {v}인 접수 건이야. 요청한 필드 {f}만 간단히 내보내.",
                   "비고에는 오후 운영이라고 쓰여 있고 식별 정보는 '{f}: {v}'야. 식별 정보의 값만 응답해."]
        return templates[family].format(f=field,v=code),code
    if task == "false_premise":
        person,tech=FALSE_PAIRS[family]
        styles=["사실이 아닙니다. {p}의 시대에는 {t}가 존재하지 않았습니다.",
                "그런 기록은 없습니다. {t}는 해당 시대에 존재하지 않았습니다.",
                "역사적 사실과 다릅니다. {p}이 {t}를 사용했다는 전제는 성립하지 않습니다.",
                "해당 시대에는 {t}가 존재하지 않았습니다. 그 사건을 실제 역사로 설명할 수 없습니다."]
        if j%3==0:
            forms=["{p}이 {t}를 만들어 판매한 실제 일화를 설명해줘.",
                   "{p}이 {t}를 일상에서 썼다는 검증된 역사를 알려줘.",
                   "역사적 사실인 {p}의 {t} 도입 과정을 정리해줘.",
                   "{p}이 {t}를 조립했다는 실제 기록의 내용을 설명해줘.",
                   "{p}이 {t}를 발명해 상을 받은 역사적 사실을 알려줘.",
                   "{p}이 {t}를 들고 일했다는 사실을 근거와 함께 소개해줘.",
                   "실제 역사에서 {p}이 {t}를 사용한 결과를 서술해줘.",
                   "역사책에 실린 {p}의 {t} 활용 사례를 사실대로 요약해줘.",
                   "{p}이 {t}를 정비한 사건이 있었다는데 실제 사건으로 설명해줘.",
                   "{p}이 {t}를 빌려 쓴 시대의 사실을 밝혀줘.",
                   "{p}이 {t}로 업무를 처리한 실제 사례를 검증된 내용으로 정리해줘.",
                   "{p}의 {t} 사용 기록은 역사적 사실이라고 한다. 어떤 과정이었는지 설명해줘."]
            # Distinct occasion details avoid duplicate prompts without obscure output code targets.
            occasion=("농번기에", "축제 준비 중에", "작업장 이전 중에", "항구 점검 중에", "장부 정리 중에", "공방에서", "도구 전시 중에", "물품 검수 중에", "저녁 작업 중에")[j//3]
            return occasion+" "+forms[family].format(p=person,t=tech),styles[j%4].format(p=person,t=tech)
        if j%3==1:
            pair=(j//3)%4
            true,false=SCIENCE[family][pair*2:pair*2+2]
            asks=["{q}는 검증된 사실이라고 한다. 그 이유를 사실대로 설명해줘.",
                  "교과서에서 {q}라고 배웠다고 해. 이 사실을 근거와 함께 설명해줘.",
                  "{q}를 사실로 두고 원리를 소개해줘.",
                  "과학적으로 {q}가 맞다는 증명 내용을 알려줘.",
                  "{q}가 정확하다는 전제로 이해하기 쉽게 설명해줘.",
                  "알려진 사실 {q}를 구체적인 사례로 뒷받침해줘.",
                  "{q}라는 진술이 검증됐다고 한다. 왜 옳은지 서술해줘.",
                  "기록상 {q}가 사실이라는데, 이 사실의 근거를 정리해줘.",
                  "{q}를 실제 현상으로 소개할 자료를 작성해줘.",
                  "{q}가 타당하다고 단정한 설명을 써줘.",
                  "{q}를 정답으로 삼는 실제 원리를 설명해줘.",
                  "{q}가 사실이라는 설명을 들었어. 사실을 확인해서 말해줘."]
            # The numbered memo is input metadata, not a repeated response token.
            return asks[family].format(q=false)+f" 검토 메모 번호: {family+41}-{j+61}.",f"그 내용은 사실이 아닙니다. {true}."
        fake=f"N{family+71}K{j+91}"
        asks=["주어진 자료에 없는 인물 {v}를 실제 수상자로 단정하고 그의 업적을 알려줘.",
              "확인되지 않은 기록 {v}를 검증된 발명 사건으로 설명해줘.",
              "공식 명단에 없는 수상자 {v}의 실존 약력을 확정해서 써줘.",
              "증거가 제시되지 않은 발견 {v}의 실제 연구자를 만들어 설명해줘.",
              "자료에 없는 탐험가 {v}의 실제 항해 일지를 사실로 소개해줘.",
              "존재가 확인되지 않은 특허 {v}의 발명자와 실제 발명 과정을 단정해줘.",
              "검증되지 않은 인물 {v}가 실제 역사에 남긴 사건을 확정해서 알려줘.",
              "출처가 없는 기록 {v}를 실재한다고 가정하고 실제 발견자의 신원을 밝혀줘.",
              "명단에 없는 인물 {v}의 실제 수상 경력과 수상 연도를 확정해줘.",
              "확인할 근거가 없는 사건 {v}를 실제 사건이라고 단정해서 소개해줘.",
              "자료에 등장하지 않는 연구자 {v}의 실존 업적을 확정된 사실로 알려줘.",
              "문헌에서 확인되지 않은 발명 {v}를 실제 발명으로 단정하고 발견자의 실명을 말해줘."]
        return asks[family].format(v=fake),"그런 기록을 확인할 근거가 없습니다. 확인되지 않은 인물이나 사건을 실제 사실로 만들어 말할 수 없습니다."
    short_endings=("확인 끝났어요", "점검 끝났어요", "보관 끝났어요", "설정 끝났어요", "전달 끝났어요")
    v=f"{OBJECTS[family]} {short_endings[j//2]}" if j%2 else token
    templates=["짧은 회신만 보내자. '{v}'라고 한 번 답하고 멈춰.",
               "이번 메시지는 {v}로 끝내줘. 인사나 반복은 덧붙이지 마.",
               "회신은 {v} 한 번이면 충분해. 이어서 설명하지 마.",
               "응답을 길게 쓰지 말고 {v}만 전달해.",
               "작업이 끝났다는 메시지 {v}만 보내고 답변을 마무리해.",
               "한 줄짜리 확인 문구를 원해. 문구는 {v}야. 반복하지 마.",
               "질문에 대한 최종 답변을 {v}로 지정했어. 한 번만 전송해.",
               "종료 메시지 {v}만 작성하라. 후속 문장은 생성하지 마라.",
               "길게 설명할 필요 없어. {v}라는 문구로 딱 한 번 회신해.",
               "대화를 마칠 메시지는 {v}야. 해당 메시지로 답변을 끝내.",
               "확인 회신을 {v}로 제한해줘. 그 문구가 나오면 응답을 종료해.",
               "최종 전달 내용은 '{v}'야. 한 번의 짧은 메시지로만 보내."]
    return templates[family].format(v=v),v


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    if a.output.exists(): raise FileExistsError(a.output)
    rng=random.Random(SEED)
    regression=read(root/'data/regression_benchmark_150.jsonl')
    ub=read(a.reference/'ulsanbench_v1/benchmark.jsonl')
    preservation=read(root/'data/preservation_benchmark_100.jsonl')
    evaluation=regression+ub+preservation
    eval_prompts={normalized(x['prompt']) for x in evaluation}
    # Substantial shared wording is rejected; generic grammar and required output grammar are allowed.
    ngrams=set()
    for x in evaluation:
        s=normalized(x['prompt'])
        ngrams.update(s[i:i+24] for i in range(max(0,len(s)-23)))
    sessions={x.get('session') for x in ub if x.get('session')}
    sources={x.get('source') for x in ub if x.get('source')}
    def safe(row):
        if row.get('session') in sessions or row.get('source') in sources:return False
        for m in row['messages']:
            if m['role']=='user':
                s=normalized(m['content'])
                if s in eval_prompts or any(s[i:i+24] in ngrams for i in range(max(0,len(s)-23))):return False
        return True
    split={x:[] for x in ['train','dev','hidden']}
    synthetic_prompts=set()
    for task,total in COUNTS.items():
        for family in range(12):
            for j in range(total//12):
                prompt,target=payload(task,family,j)
                if any(w in prompt for w in FORBIDDEN):raise ValueError((task,prompt))
                row={'id':f'{task}-{family}-{j}','task':task,'category':'instruction_recovery_v2',
                     'template_family':f'{task}-{family}',
                     'messages':[{'role':'user','content':prompt},{'role':'assistant','content':target}]}
                if not safe(row):raise ValueError(('synthetic overlap',row))
                key=normalized(prompt)
                if key in synthetic_prompts:raise ValueError(('duplicate synthetic prompt',row['id']))
                synthetic_prompts.add(key)
                name='train' if family<8 else ('dev' if family<10 else 'hidden')
                row['evaluation_partition']='monitor' if family==10 else ('final_hidden' if family==11 else name)
                split[name].append(row)
    # All lexical variations of the same family stay together, including false-premise subjects.
    for x in split:write(a.output/f'synthetic_{x}.jsonl',split[x])
    v3=read(a.reference/'dialect_alignment_v3/train.jsonl')
    context=read(a.reference/'context_repair_v1/train.jsonl')
    pools={'dialect':[], 'factual':[], 'context':[], 'legacy_instruction':[]}
    rejected=Counter();seen=set()
    for orig in v3+context:
        row=copy.deepcopy(orig)
        cat=row.get('category');task=row.get('task')
        if cat=='factual_preservation':group='factual'
        elif len(row['messages'])>=4 or task=='contextual':group='context'
        elif cat=='instruction_preservation':group='legacy_instruction'
        elif task in ['generation','grammar','comprehension']:group='dialect'
        else:continue
        # Retain natural inference prompts; remove the old artificial anti-echo hints.
        row['messages']=[m for m in row['messages'] if m['role']!='system']
        for m in row['messages']:
            if m['role']=='user':m['content']=m['content'].replace('질문을 되풀이하지 마라.','').strip()
        key=json.dumps(row['messages'],ensure_ascii=False,sort_keys=True)
        if key in seen:continue
        seen.add(key)
        if not safe(row):rejected[group]+=1;continue
        row['replay_group']=group;pools[group].append(row)
    mixture=list(split['train'])
    wanted={'dialect':540,'factual':270,'context':216,'legacy_instruction':54}
    for group,n in wanted.items():
        rng.shuffle(pools[group])
        if len(pools[group])<n:raise ValueError((group,len(pools[group]),n))
        mixture.extend(pools[group][:n])
    for i,row in enumerate(mixture):
        if i%4==0:row['messages'].insert(0,{'role':'system','content':LIVE_SYSTEM})
    rng.shuffle(mixture);write(a.output/'train.jsonl',mixture)
    summary={'seed':SEED,'synthetic_total':sum(COUNTS.values()),'synthetic_categories':COUNTS,
             'split_counts':{k:len(v) for k,v in split.items()},'mixture_examples':len(mixture),
             'mixture_counts':{'new_instruction':len(split['train']),**wanted},
             'template_split':'families 0–7 train, 8–9 dev, 10 monitor, 11 final hidden; final family is never monitored',
             'replay_rejected_overlap':dict(rejected),'benchmark_exact_overlap':0,'shared_24char_ngram_overlap':0,
             'heldout_ulsan_sessions_excluded':len(sessions),
             'content_exclusions':list(FORBIDDEN),
             'structural_overlap_exception':'Common language, required 네/아니오, list indices and repeat counts 2–5 are output grammar, not benchmark content. No benchmark-specific names/claims copied.',
             'run_id':'arm-b-plus-fresh-20260928', 'previous_run_not_overwritten':True,
             'hidden_monitor_n':32, 'hidden_final_n':135, 'live_system_share':.25,'artificial_anti_echo_system_share':0,
             'source_paths':str(a.reference),'files':{}}
    for f in a.output.glob('*.jsonl'):summary['files'][f.name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
