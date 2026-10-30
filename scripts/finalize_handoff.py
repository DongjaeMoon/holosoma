"""Bundle the latest resume prompt with preserved historical source documents."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / 'HOLOSOMA_RESUME_PROMPT.md'
text = destination.read_text()
marker = '<!-- ARCHIVE_APPEND_POINT -->'
if marker in text:
    archives = []
    for name in ('HANDOFF_PROMPT.md', 'FASTSAC_STATUS.md'):
        archives.append('\n\n---\n\n## 과거 기록 원문: ' + name + '\n\n'
                        '> 과거 시점의 기록이다. 최신 상태는 이 파일 앞쪽 본문을 우선한다.\n\n'
                        + (root / name).read_text())
    destination.write_text(text.replace(marker, ''.join(archives)))

legacy = root / 'HANDOFF_PROMPT.md'
old = legacy.read_text()
notice = ('> **최신 종합 인계는 [HOLOSOMA_RESUME_PROMPT.md](HOLOSOMA_RESUME_PROMPT.md)를 읽으세요.** '
          '2026-10-29: v2 네 모델 10k 학습, 전체 체크포인트 평가, PNG 및 최종 좌우 분할 영상까지 모두 완료됐습니다. '
          '아래는 초기 구현 당시 기록이므로 현재 실행 상태로 해석하지 마세요.\n\n')
if notice not in old:
    lines = old.splitlines(keepends=True)
    if len(lines) > 2 and lines[2].startswith('> 최신 상태'):
        lines[2] = ''
    legacy.write_text(lines[0] + '\n' + notice + ''.join(lines[1:]))

content = destination.read_text()
assert 'training_vs_deploy.mp4' in content
assert 'ham_sa_from_ma_v4_abl_preact.yaml' in content
assert '## 과거 기록 원문: HANDOFF_PROMPT.md' in content
assert '## 과거 기록 원문: FASTSAC_STATUS.md' in content
print(f'{destination}\n{len(content.splitlines())} lines; {destination.stat().st_size} bytes')
