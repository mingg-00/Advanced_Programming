# Kiwoom 자동 매매 트레이딩 어시스턴트

강화학습 기반의 자동 매매 시뮬레이터, 데이터 탐색, 챗봇, 주식 계산기를 Streamlit으로 통합한 프로젝트입니다. 키움증권 OpenAPI+를 활용해 실거래 데이터를 수집하고, 모의투자 계좌를 대상으로 매매 전략을 검증할 수 있도록 설계했습니다.

## 주요 기능
- **자동 매매 시뮬레이터**: `stable-baselines3`의 PPO 에이전트로 매수·매도 전략을 학습하고 백테스트합니다.
- **데이터 수집**: Kiwoom OpenAPI+ (Windows) 또는 개발용 yfinance 폴백으로 일봉 데이터를 가져옵니다.
- **백테스트/시각화**: 누적 수익률과 자산曲선을 Streamlit 차트와 `mplfinance` 캔들 차트로 제공합니다.
- **도우미 챗봇**: OpenAI API를 이용한 종목/지표 Q&A.
- **주식 계산기**: 손익, 평단가, 환율 계산 지원.

## 빠른 시작

```bash
conda create -n kiwoom-trader python=3.9
conda activate kiwoom-trader
pip install -r requirements.txt
```

필요 시 yfinance 폴백을 위해 추가 패키지를 설치합니다(이미 requirements에 포함).

### 환경 변수 설정
루트에 `.env` 파일을 생성하고 아래 값을 채웁니다:

```
OPENAI_API_KEY=sk-...
DEFAULT_SYMBOL=005930.KS
```

Windows에서 Kiwoom OpenAPI+를 사용할 경우, 키움증권 개설 계좌로 로그인 가능한 환경(HTS 설치)과 `pykiwoom` 설정이 선행되어야 합니다.

## 실행

```bash
streamlit run app.py
```

### 메뉴 구성
- **대시보드**: 종목 선택, 캔들차트, RL 학습 및 백테스트.
- **자료실**: 종목 코드 테이블 및 시세 조회.
- **챗봇**: OpenAI GPT 기반 투자 도우미.
- **계산기**: 손익/평단가/환율 계산기.

## Kiwoom 환경 준비 (Windows)
1. 키움증권 영웅문 설치 및 계좌 발급.
2. 키움 OpenAPI+ 설치.
3. `pykiwoom` 설치 (`pip install pykiwoom`).
4. 최초 실행 시 HTS 로그인 후, OpenAPI+ 개발자 인증을 마칩니다.

macOS나 Linux 환경에서는 Kiwoom COM 라이브러리를 사용할 수 없으므로 앱이 자동으로 yfinance 폴백 모드로 전환됩니다. 배포 시 반드시 Windows 환경에서 실행해주세요.

## 강화학습/백테스트
- PPO 모델은 `app/services/rl_trading.py`에 정의된 `TradingEnv` 환경에서 학습합니다.
- Streamlit UI에서 학습 스텝과 데이터 기간을 선택할 수 있으며, 학습 완료 후 백테스트를 수행해 누적 수익률, 자산曲선, 샤프 유사 지표를 확인합니다.

## 챗봇
- OpenAI API Key를 사이드바에서 입력하면 `TradingAssistantChatbot`이 활성화됩니다.
- 질문 이력은 세션에 저장되며, 동일 세션에서 반복 조회가 가능합니다.

## 주식 계산기
- **손익 계산**: 수수료/세금을 반영한 순이익과 수익률.
- **평단가 계산**: 총매수금 및 수량 기반의 평단가.
- **환율 계산**: 원화↔외화 변환.

## 개발 노트
- 프로젝트 구조와 서비스 레이어는 `app/services`, `app/components`, `app/utils`로 분리돼 유지보수가 용이합니다.
- Streamlit 세션 상태를 이용해 데이터 공급자, RL 모델, 챗봇 인스턴스를 재사용합니다.
- Windows 이외의 OS에서는 Kiwoom 클라이언트가 `KiwoomUnavailableError`를 발생시키며, UI에 개발용 폴백 메시지가 노출됩니다.

## 라이선스
이 프로젝트는 교육용 예제로 제공됩니다. 실제 투자에 사용하기 전 반드시 충분한 검증을 거치고, 발생 가능한 손실은 사용자 책임입니다.


