Holosoma 야간 감사 — 먼저 읽을 결과
2026-11-02 / 기존 코드 수정 없음 / GPU·학습·물리 시뮬레이션 재실행 없음

1. “버그가 전혀 없다”는 결론은 틀리다.
   기존 empirical normalization의 분산 병합 오류를 CPU에서 재현했다.
   감사용 소스 복사 충돌, 현재 체크포인트 로더 호환성 문제,
   native matched-state 영상 복원 오류도 확인했다. 수정하지 않았다.

2. 그렇다고 현재 MLP의 오른쪽 추종 현상이 가짜라고 확인된 것도 아니다.
   새 vanilla MLP 실행의 실제 policy 입력 122,884,096건,
   샘플된 replay 현재/다음 관측 3,930,718,208건에서 raw vy<0은 0건이다.
   MLP의 실제 배치 준비 함수를 CPU에서 호출해도 반사 augmentation은 없다.
   과거 v2 MLP에는 전수 커맨드 감사 기록이 없어 같은 강도의 역사적 증명은 불가능하다.

3. 오른쪽으로 “잘 걷는다”와 목표 속도를 정확히 따라간다는 것은 다르다.
   새 vanilla MLP 10k: 오른쪽 0.3/0.6 명령의 planar RMSE = 0.2320 / 0.4003 m/s.
   실제 평균 vy는 -0.1474 / -0.2667 m/s이다. 32회씩 모두 생존했지만 성공 기준은 0/32씩이다.
   30k: RMSE = 0.1914 / 0.4012, 평균 vy = -0.2359 / -0.3902.
   0.6에서는 평균 vx가 +0.1690으로 흘러가서 planar 오차가 남는다.

4. 네트워크가 같은 것은 아니다.
   동일 v2 10k 조건의 물리적 반사 잔차 RMS:
   biHAM 1.63e-7 / HAM+symAug 0.0637 / HAM 0.1205 / MLP 0.1002.
   biHAM의 구조적 대칭성은 실제 저장 가중치에서도 작동한다.
   반면 biHAM의 왼쪽 목표 속도 자체를 충분히 학습하지 못해 오른쪽 성능도 제한된다.

5. 비교 조건을 섞으면 안 된다.
   기존 4종 v2: IsaacGym + reflection normalization + Domi phase, 각 10k.
   새 vanilla MLP: IsaacSim + empirical normalization + native phase, 30k까지.
   HAM+symAug는 합성된 오른쪽 학습 샘플을 받으므로 동일한 OOD 대조군이 아니다.
   전후/좌 명령을 배운 실행이지, 순수 왼쪽 이동만 배운 실행도 아니다.
   yaw는 양방향이며, env 0 실제 몸체 vy는 학습 중 약 45.8%의 시점에서 음수였다.

6. 검증 범위와 한계
   체크포인트 150개 전체 로딩·유한성·step 확인.
   1,790개 평가 케이스 / 57,280회 trial 저장값의 수치 일관성 확인: 불일치 0건.
   TensorBoard 188개 파일, 영상 86개 전체 CPU 디코딩 확인.
   활성 코드 Python 689개 + 추가/보존 코드 Python 730개 구문 검사.
   이는 저장 자료 및 CPU 동작 감사이다. 기존 물리 trajectory 전체를 재현한 것은 아니다.
   MLP 일반화의 정확한 인과 메커니즘은 아직 입증되지 않았다.

자세한 설명: FULL_AUDIT_REPORT_KO.txt
그림과 핵심 표: index.html
모든 현재 좌/우 수치: comparison_lateral_current.csv
소스 보존 확인: source_integrity_final.json
재현 방법: reproduce_cpu_audit.py (새 출력 디렉터리만 허용)

원본 코드와 기존 결과는 그대로 두고, 이 감사 디렉터리 안에만 새 산출물을 작성했다.
