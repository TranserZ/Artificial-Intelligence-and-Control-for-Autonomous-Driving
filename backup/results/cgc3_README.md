# 청계천로 3차 학습 (cgc3_ft_s0~s2)

- Kaggle `cgc-ppo-train4`, 2차 best(cgc2_ft_s*)에서 이어 학습, 11.5시간 제한으로 약 73만 스텝에서 종료
- 코드: `feature/cgc-train` e1909af (shield 끄는 판 0→70% / 60만 스텝, 개입 감점 0.1 + 1.0×세기, best는 shield 끈 성적으로 선택)
- `best_model.pt`: 학습 중 shield 끈 평가(20판) 최고 / `model.pt`: 마지막
- `local_eval/`: PC에서 시드 1000~1049, 50판 평가 로그 (`shield_off` = 차량·신호 shield 끔, 보행자 필터 켬)
- 추천 모델: **cgc3_ft_s2/best_model.pt**
