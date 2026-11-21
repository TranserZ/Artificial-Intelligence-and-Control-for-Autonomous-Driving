"""로깅 유틸 — 콘솔 / TensorBoard / CSV 를 한 곳에서 처리.

알고리즘 코드가 로깅 세부사항(파일 경로, 텐서보드 유무 등)을 신경쓰지
않도록 분리했다. 알고리즘은 `logger.log_train(...)` 처럼 부르기만 하면 된다.
"""
import csv
import math
import os


class RunLogger:
    """한 번의 학습 실행(run)에 대한 로그를 모아 기록한다.

    - 콘솔: 사람이 읽을 한 줄 요약
    - TensorBoard: results/<run>/tb/ 에 스칼라 기록 (tensorboard 없으면 자동 생략)
    - CSV: training_log.csv(학습 곡선), eval_log.csv(주행 지표)
           → close() 또는 save_csv() 호출 시 파일로 씀
    """

    def __init__(self, run_dir: str, use_tensorboard: bool = True):
        self.run_dir = run_dir
        self.train_rows = []   # 학습 곡선 기록
        self.eval_rows = []    # 평가 지표 기록
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(os.path.join(run_dir, "tb"))
            except Exception as e:
                # tensorboard 미설치, 또는 경로 문제(한글 경로 등)로 실패해도
                # 학습 자체는 계속되어야 하므로 CSV만 쓰도록 넘어간다.
                print(f"[안내] TensorBoard 로깅 비활성화 ({type(e).__name__}). "
                      f"CSV 로그만 기록합니다.")
                self.writer = None

    # ------------------------------------------------------------------
    def log_train(self, step: int, metrics: dict, verbose: bool = True):
        """업데이트 1회의 학습 지표 기록."""
        row = {"steps": step, **metrics}
        self.train_rows.append(row)

        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", v, step)

        if verbose:
            parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items() if k != "update"]
            upd = metrics.get("update", "")
            print(f"[update {upd:>4}] steps={step:>7d}  " + "  ".join(parts))

    def log_eval(self, step: int, metrics: dict, summary: str = "",
                 n_episodes: int = 0):
        """주기적 평가 결과 기록."""
        self.eval_rows.append({"steps": step, **metrics})

        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"eval/{k}", v, step)

        print(f"    [eval @{step}] {n_episodes}ep  {summary}")

    def log_episode(self, step: int, ep_return: float, ep_length: int):
        """학습 중 개별 에피소드가 끝날 때 (TensorBoard 전용, 콘솔 출력 없음)."""
        if self.writer is not None:
            self.writer.add_scalar("episode/return", ep_return, step)
            self.writer.add_scalar("episode/length", ep_length, step)

    # ------------------------------------------------------------------
    def save_csv(self):
        """모아둔 기록을 CSV 파일로 저장."""
        for name, rows in [("training_log.csv", self.train_rows),
                           ("eval_log.csv", self.eval_rows)]:
            if not rows:
                continue
            # 모든 행의 키를 합집합으로 모아 헤더를 만든다 (키가 달라도 안전)
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(self.run_dir, name), "w", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _numeric_series(rows, key):
        points = []
        for row in rows:
            try:
                x, y = float(row["steps"]), float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        return points

    def save_curves(self):
        """CSV와 같은 값으로 학습 곡선 PNG를 자동 생성한다."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[안내] 학습 곡선 PNG 저장 실패 ({type(e).__name__}: {e})")
            return []

        curves_dir = os.path.join(self.run_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)
        saved = []

        def plot(name, title, rows, keys):
            series = [(label, self._numeric_series(rows, key))
                      for key, label in keys]
            series = [(label, pts) for label, pts in series if pts]
            if not series:
                return
            fig, ax = plt.subplots(figsize=(9, 5))
            for label, pts in series:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        label=label, linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("training steps")
            ax.grid(True, alpha=0.3)
            if len(series) > 1:
                ax.legend()
            fig.tight_layout()
            path = os.path.join(curves_dir, name)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        plot("reward.png", "Training reward", self.train_rows,
             [("ep_rew_mean", "episode reward mean")])
        plot("losses.png", "PPO losses", self.train_rows,
             [("policy_loss", "policy loss"), ("value_loss", "value loss")])
        plot("entropy.png", "Policy exploration", self.train_rows,
             [("entropy", "entropy"), ("p_lane_change", "lane-change probability")])
        plot("training_outcomes.png", "Recent training outcomes", self.train_rows,
             [("official_completion_rate", "official complete"),
              ("perfect_completion_rate", "perfect complete"),
              ("collision_rate", "collision"),
              ("timeout_rate", "timeout"),
              ("stop_success_mean", "stops done / episode"),
              ("lane_changes_per_episode", "lane changes / episode")])
        plot("validation.png", "Validation metrics", self.eval_rows,
             [("completion_rate", "completion rate"),
              ("intervention_rate", "intervention rate"),
              ("collision_rate", "collision rate")])
        return saved

    def close(self):
        self.save_csv()
        self.save_curves()
        if self.writer is not None:
            self.writer.close()
