from pathlib import Path
import shutil

from src.run_pipeline import run_pipeline


def test_pipeline_creates_outputs(tmp_path):
    image_path = Path('sample_inputs/mithuna_solutions.jpg')
    if not image_path.exists():
        raise FileNotFoundError('sample input image not found')

    output_dir = tmp_path / 'outputs'
    result = run_pipeline(
        image_path=str(image_path),
        board_width_inches=236,
        output_dir=str(output_dir),
        threshold=140,
        bridge_gap=3,
        min_piece_size=50,
        ignore_margin=0.02,
        upscale=1.0,
        merge_mode='word',
    )

    assert result['piece_count'] >= 1
    assert (output_dir / 'mithuna_solutions_annotated.jpg').exists()
    assert (output_dir / 'mithuna_solutions_results.json').exists()
    shutil.rmtree(output_dir, ignore_errors=True)
