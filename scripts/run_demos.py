# Run the demo cases in demos/ (retrieval & rebirth, prop-free teleoperation, compositional generation)
# and compose the project-page style comparison videos.
'''
Usage (inside the container, after `source shell/export.sh` and `bash shell/download_checkpoints.sh --no-t5`):
    python scripts/run_demos.py                         # all cases + the three figures
    python scripts/run_demos.py --cases retrieval_put_the_marker_in_the_bowl
Outputs: ${ROBODREAM_OUTPUT_DIR}/demos/<case>/{external,wrist}.mp4 and figure_*.mp4
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robodream import demo as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ckpt_dir = os.environ.get('ROBODREAM_CKPT_DIR', 'checkpoints')
    parser = argparse.ArgumentParser(description='RoboDream demos', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--demo_dir', default=os.path.join(ROOT, 'demos'))
    parser.add_argument('--ckpt', default=f'{ckpt_dir}/robodream/model.pt')
    parser.add_argument('--default_text', default=f'{ckpt_dir}/robodream/default_text_emb.pt')
    parser.add_argument('--tokenizer', default=f'{ckpt_dir}/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth')
    parser.add_argument('--output_dir', default=os.path.join(os.environ.get('ROBODREAM_OUTPUT_DIR', 'outputs'), 'demos'))
    parser.add_argument('--cases', nargs='*', default=None, help='Case names from demos/manifest.json (default: all)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=35)
    parser.add_argument('--overwrite', action='store_true', help='Regenerate cases whose videos already exist')
    return parser.parse_args()


def make_figures(demo_dir: str, manifest: dict, results: dict, out_dir: str) -> None:
    cases = {n: D.load_case(demo_dir, manifest, n) for n in manifest['cases']}
    refs = manifest['references']

    retrieval = [c for c in cases.values() if c['mode'] == 'retrieval_rebirth']
    if all(c['name'] in results for c in retrieval):
        references = {c['name']: D.read_video(os.path.join(demo_dir, refs[c['name'].replace('retrieval_', 'retrieved_')]['path'])) for c in retrieval}
        D.write_video(os.path.join(out_dir, 'figure_retrieval_rebirth.mp4'), D.retrieval_figure(retrieval, results, references))

    teleop = D.read_video(os.path.join(demo_dir, refs['propfree_teleop']['path']))
    pf = [cases[n] for n in ['propfree_marker_bowl', 'propfree_novel_instance', 'propfree_novel_task']]
    if all(c['name'] in results for c in pf):
        D.write_video(os.path.join(out_dir, 'figure_prop_free.mp4'), D.prop_free_figure(teleop, pf, results))

    variants = [(cases['propfree_marker_bowl'], 'Generated demo', 'gray', 'object_prior'),
                (cases['propfree_novel_instance'], 'w/ Novel instance', 'blue', 'object_prior'),
                (cases['propfree_novel_scene'], 'w/ Novel scene', 'green', 'scene_prior'),
                (cases['propfree_novel_task'], 'w/ Novel task', 'red', 'object_prior'),
                (cases['propfree_novel_view'], 'w/ Novel view', 'orange', 'scene_prior')]
    if all(v[0]['name'] in results for v in variants):
        D.write_video(os.path.join(out_dir, 'figure_compositional.mp4'),
                      D.compositional_figure(cases['propfree_marker_bowl'], variants, results))


def main():
    args = parse_args()
    manifest = D.load_manifest(args.demo_dir)
    names = args.cases or list(manifest['cases'])
    results = {n: r for n in manifest['cases'] if (r := D.load_result(args.output_dir, n)) is not None}
    todo = [n for n in names if args.overwrite or n not in results]

    if todo:
        import torch
        from robodream.generate import RoboDreamGenerator
        generator = RoboDreamGenerator(
            checkpoint_path=args.ckpt, tokenizer_path=args.tokenizer,
            t5_cache_path=os.path.join(args.demo_dir, 't5_prompts.pkl.gz'),
            default_text_path=args.default_text if os.path.isfile(args.default_text) else None,
            device='cuda' if torch.cuda.is_available() else 'cpu', num_steps=args.num_steps)
        for n in todo:
            case = D.load_case(args.demo_dir, manifest, n)
            print(f'[{n}] {case["prompt"]!r}, {len(case["trajectory"])} frames', flush=True)
            results[n] = D.run_case(generator, case, seed=args.seed)
            D.save_result(args.output_dir, n, results[n])

    make_figures(args.demo_dir, manifest, results, args.output_dir)
    print(f'Results in {args.output_dir}')


if __name__ == '__main__':
    main()
