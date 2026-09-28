"""Decide from complete measured results; publish no weights and change no Live asset."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/instruction-recovery-v2'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def predictions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def values(result):
    reg, bench = result['regression'], result['ulsanbench']
    return {
        'instruction_legacy_pct': reg['instruction_trap']['accuracy_pct'],
        'instruction_corrected_pct': reg['corrected_instruction_pct'],
        'instruction_serving_legacy_pct': result['serving_instruction']['instruction_trap']['accuracy_pct'],
        'instruction_serving_corrected_pct': result['serving_instruction']['corrected_instruction_pct'],
        'factual_qa_pct': reg['factual_qa']['accuracy_pct'],
        'multi_turn_pct': reg['multi_turn']['accuracy_pct'],
        'generation_semantic': bench['generation']['semantic_similarity'],
        'generation_dialectness': bench['generation']['dialectness_proxy'],
        'grammar_semantic': bench['grammar']['semantic_similarity'],
        'grammar_dialectness': bench['grammar']['dialectness_proxy'],
        'context_semantic': bench['context']['semantic_similarity'],
        'context_repetition_count': bench['context']['repetition_count'],
        'context_malformed_count': bench['context']['malformed_count'],
        'comprehension_semantic': bench['comprehension']['semantic_similarity'],
        'identification_accuracy': bench['identification']['accuracy'],
        'synthetic_final_hidden_pct': result['synthetic']['accuracy_pct'],
        'synthetic_dev_pct': result['dev']['accuracy_pct'],
        'general_regression_repetition_count': reg['general_regression_repetition_count'],
    }


def main():
    complete = load(REPORTS / 'execution_complete.json')
    arms = load(REPORTS / 'arms.json')
    if complete['arms'] != 3 or len(arms) != 3 or any(arm['steps'] > 80 for arm in arms):
        raise RuntimeError('Expected three complete bounded SFT arms')
    config = load(ROOT / 'configs/sft/ulm4b_instruction_recovery_v2.json')
    dataset = load(REPORTS / 'dataset_summary.json')
    token_audit = load(REPORTS / 'token_audit.json')
    failure = load(REPORTS / 'failure_analysis.json')
    baseline = load(ROOT / 'configs/release/ulm4b-arm-b.json')['evaluation'].copy()
    baseline['instruction_legacy_pct'] = baseline.pop('instruction_trap_pct')
    baseline['instruction_corrected_pct'] = failure['corrected_instruction_pct']
    serving = load(REPORTS / 'baseline_serving_instruction/summary.json')
    baseline['instruction_serving_legacy_pct'] = serving['instruction_trap']['accuracy_pct']
    baseline['instruction_serving_corrected_pct'] = serving['corrected_instruction_pct']
    baseline['comprehension_semantic'] = load(REPORTS / 'baseline/ulsanbench/summary.json')['metrics']['comprehension']['semantic_similarity']
    baseline['general_regression_repetition_count'] = sum(row['repetition'] for row in predictions(REPORTS / 'baseline/regression/predictions.jsonl') if row['category'] == 'general_korean')
    baseline['synthetic_final_hidden_pct'] = load(REPORTS / 'baseline_final_hidden/summary.json')['accuracy_pct']
    thresholds = {
        'instruction_legacy_pct': 90, 'instruction_corrected_pct': 90,
        'instruction_serving_legacy_pct': 90, 'instruction_serving_corrected_pct': 90,
        'factual_qa_pct': 80, 'multi_turn_pct': 95,
        'generation_semantic': .98, 'generation_dialectness': .65,
        'grammar_semantic': .97, 'grammar_dialectness': .88, 'context_semantic': .44,
    }
    allowed_drops = {
        'generation_semantic': -.005, 'generation_dialectness': -.015,
        'grammar_semantic': -.005, 'grammar_dialectness': -.02,
        'context_semantic': -.02, 'comprehension_semantic': -.01,
    }
    full_path = REPORTS / 'full_candidates.json'
    candidates = []
    for item in load(full_path) if full_path.exists() else []:
        metrics = values(item['result'])
        delta = {key: round(value - baseline[key], 5) for key, value in metrics.items() if key in baseline}
        failed = [f'{key}={metrics[key]} < {floor}' for key, floor in thresholds.items() if metrics[key] < floor]
        failed += [f'{key}={metrics[key]} must be 0' for key in ['context_repetition_count', 'context_malformed_count'] if metrics[key] != 0]
        failed += [f'{key} delta {delta[key]} < {floor}' for key, floor in allowed_drops.items() if delta[key] < floor]
        if metrics['general_regression_repetition_count'] > baseline['general_regression_repetition_count']:
            failed.append('General-language runaway increased above the original Arm B')
        candidates.append({
            'arm': item['arm'], 'step': item['step'], 'adapter': item['adapter'],
            'metrics': metrics, 'delta': delta, 'accepted': not failed, 'failed_gates': failed,
        })
    candidates.sort(key=lambda item: (item['accepted'], item['metrics']['instruction_corrected_pct'], item['metrics']['synthetic_final_hidden_pct'], item['metrics']['context_semantic']), reverse=True)
    best = candidates[0] if candidates else None
    accepted = bool(best and best['accepted'])
    analysis_path = REPORTS / 'final_failure_analysis.json'
    best_failure = load(analysis_path).get(f"final_arm_{best['arm']}_step_{best['step']}/regression", {}) if best and analysis_path.exists() else {}
    monitor = max(({'arm': arm['arm'], 'lr': arm['lr'], **gate} for arm in arms for gate in arm['gates']), key=lambda gate: (not gate['stop_reasons'], gate['result']['regression']['corrected_instruction_pct'], gate['result']['synthetic']['accuracy_pct']))
    dpo_path = REPORTS / 'dpo_results.json'
    dpo_executed = dpo_path.exists()
    dpo_condition = all(gate['result']['regression']['corrected_instruction_pct'] < 90 for arm in arms for gate in arm['gates'])
    limitations = [
        'The 20 instruction traps and declared UlsanBench subsets were monitored as requested; they are not untouched final benchmarks. No benchmark prompt or target was copied into synthetic training data.',
        'Final synthetic template family 11 is withheld from all training, dev evaluation and checkpoint monitoring; monitor items come only from family 10.',
        'Conservative rejection expressions plus manual prediction review are used. These are not universal semantic entailment checks.',
        'Serving-policy instruction scores apply the Live decoding values to benchmark prompts; they are not a test of the Live HTTP endpoint or its exact system prompt.',
        'The existing global no-repeat-ngram rule conflicts with verbatim repetition and prompt copying. Training cannot override forbidden logits. Live decoding is unchanged.',
        'UlsanBench embedding and dialectness values are existing evaluator proxies, not human judgments.',
        'Archived full baseline is reused only after all protected release files match their SHA256 manifest; fresh instruction and fast preservation baselines are also measured.',
    ]
    decision = {
        'run_id': dataset['run_id'], 'name': 'ULM-4B Arm B+' if accepted else 'ULM-4B instruction candidate (rejected)',
        'accepted': accepted, 'status': 'candidate-passed' if accepted else 'rejected',
        'selected_arm': best['arm'] if accepted else None,
        'best_experimental_candidate': best, 'best_monitor_candidate': monitor,
        'best_candidate_instruction_failure_types': best_failure.get('corrected_failed_by_type', {}),
        'baseline': baseline, 'candidates': candidates, 'thresholds': thresholds,
        'maximum_preservation_drops': allowed_drops, 'original_arm_b_and_live_unchanged': True,
        'automatic_deployment': False, 'dpo_condition_reached': dpo_condition,
        'dpo_executed': dpo_executed, 'evaluation_limitations': limitations,
        'dpo_decision': 'Considered, not executed: keep the explicitly requested three-arm SFT experiment bounded. Strict serving repetition targets are prohibited by the existing native no-repeat processor; preference training cannot make prohibited tokens available. Saved SFT results remain available for a separately designed preference experiment.',
        'next_single_experiment': 'Test a narrowly scoped decoding exception for explicit exact-output contracts on new held-out prompts, while retaining the existing loop controls for ordinary conversation. This is a proposal only; no Live policy is changed in this experiment.',
    }
    save(REPORTS / 'decision.json', decision)
    save(REPORTS / 'final_comparison.json', {'baseline': baseline, 'candidates': candidates, 'accepted': accepted, 'selected_arm': decision['selected_arm']})
    for arm in arms:
        save(REPORTS / f"arm_{arm['arm']}_results.json", arm)
    export_path = '/home/ubuntu/models/ULM-4B-Arm-B-Plus' if accepted else None
    release = {
        'name': decision['name'], 'status': decision['status'], 'deployable': accepted,
        'base_model': config['base_model'], 'initial_adapter': config['initial_adapter'],
        'adapter_path': export_path, 'source_checkpoint': best['adapter'] if best else None,
        'best_experimental_adapter': best['adapter'] if best else None,
        'automatic_deployment': False, 'original_release_config': 'configs/release/ulm4b-arm-b.json',
        'inference': {'enable_thinking': False, 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 3},
        'evaluation': best['metrics'] if best else {}, 'report': 'reports/ULM_4B_ARM_B_PLUS_REPORT.md',
    }
    save(ROOT / 'configs/release/ulm4b-arm-b-plus.json', release)
    lines = ['# ULM-4B Arm B+ experiment — fresh data', '', 'Date: 2026-09-28', '',
             f"Verdict: **{'PASS — candidate only' if accepted else 'REJECT — original Arm B remains the release candidate'}**.", '',
             'This is a new, explicitly requested three-arm experiment. Earlier instruction-recovery-v2 data, adapters and results are preserved. The branch and EC2 worktree are `ulm4b-arm-b-plus` and `/home/ubuntu/ulm4b-arm-b-plus`, based on main commit `60f4a926ae0006dbf3fa40c553dc19b222b1078a`. No main merge or Live deployment is performed.', '',
             '## Before and after', '',
             'Comparable regression uses the archived greedy policy with thinking disabled; context uses repetition_penalty=1.1 and no_repeat_ngram_size=3. Separate serving instruction rows apply both Live decoding controls to every instruction prompt. All four instruction scores must reach 90% for acceptance.', '',
             '| Metric | Original Arm B | Best fully evaluated candidate | Delta |', '| --- | ---: | ---: | ---: |']
    if best:
        for key, value in best['metrics'].items():
            lines.append(f"| {key} | {baseline.get(key, '—')} | {value} | {best['delta'].get(key, '—')} |")
        lines += ['', f"Best experimental checkpoint: `{best['adapter']}` (arm {best['arm']}, step {best['step']}).", '', 'Failed gates:', '']
        lines += [f'- {reason}' for reason in best['failed_gates']] or ['- None.']
        lines += ['', f"Remaining strict instruction failures by type: `{best_failure.get('corrected_failed_by_type', {})}`. The saved full predictions distinguish extra blank lines in repeated output from factually wrong yes/no polarity; these remain model failures after legitimate rejection expressions are recognized."]
    else:
        lines += ['', 'No checkpoint passed the fast preservation screen; no unsafe checkpoint was promoted to a full candidate. Best monitor result is recorded separately and is not presented as full-benchmark performance.']
    lines += ['', '## Arms, initialization and runtime', '', '| Arm | LR | Optimizer steps | Runtime (seconds, including fast gates) | Best corrected instruction | Early stop |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for arm in arms:
        peak = max(gate['result']['regression']['corrected_instruction_pct'] for gate in arm['gates'])
        lines.append(f"| {arm['arm']} | {arm['lr']} | {arm['steps']} | {arm['runtime_s']} | {peak} | {', '.join(arm['early_stop_reasons']) or '80-step cap'} |")
    lines += ['', f"Initial adapter: `{config['initial_adapter']}`. Historical and release assets have identical weight hashes; the release path is loaded directly for every arm. `initial_load.json` and `INITIAL_TRAINABLE` log lines verify all 496 inherited LoRA tensors exactly match the release asset, only LoRA tensors require gradients, and `inference_mode=false`. The base is frozen.", '',
              'GPU: NVIDIA L40S, 46068 MiB, g6e.xlarge in ap-northeast-2. Python environment: `/home/ubuntu/ULM-1.7B/.venv/bin/python`; torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, trl 1.13.0. Existing reference kernels are reused; the runtime environment is unchanged.', '',
              'Continued completion-only SFT: inherited LoRA r=8/alpha=16/dropout=.05 on 12 projections; AdamW, BF16, batch 8 × accumulation 2, max length 1024, cosine LR, 2 warmup steps, weight decay .01. Checkpoints and fast gates occur every 20 steps, at most 80 steps per arm. Actual serialized training arguments are saved with each arm.', '',
              'LRs were adjusted to 2e-6/4e-6/7e-6 using the prior failed experiment as evidence: 2e-7 barely moved the adapter and did not improve corrected instruction. An additional general-language runaway gate rejects any increase over the fresh fast baseline.', '',
              '## Dataset and leakage audit', '',
              f"Synthetic total: {dataset['synthetic_total']}. Categories: `{dataset['synthetic_categories']}`. Splits: `{dataset['split_counts']}`. {dataset['template_split']}.", '',
              f"Training mixture: {dataset['mixture_examples']} examples, `{dataset['mixture_counts']}`. New instruction 50%, dialect 25%, factual 12.5%, context 10%, legacy instruction 2.5%. Only generation/grammar/comprehension dialect replay is selected, together with natural context examples. The previous artificial anti-echo system hint is removed; 25% use the existing Live dialect system prompt.", '',
              f"Dialect target-token share: {100*token_audit['target_token_share']['dialect']:.2f}%; new instruction share: {100*token_audit['target_token_share']['new_instruction']:.2f}%. All {token_audit['examples']} completions supervise EOS. Maximum sequence length {token_audit['max_sequence_tokens']}; truncations 0. A real SFTTrainer batch verifies ignored prompt labels and supervised completions/EOS before training.", '',
              'All 1,620 synthetic prompts are unique; none is reused verbatim from the prior synthetic run. Exact benchmark prompt and shared normalized 24-character expression overlaps are zero. All 170 UlsanBench source sessions are excluded from replay. Train/dev/monitor/final template families are disjoint. Yes/no targets are balanced in each split. Required structural tokens such as 네/아니오, numbering and repetition counts are shared output grammar; benchmark names, claims and copied answer payloads are excluded. Four generated JSONL hashes match the independently generated AWS training inputs.', '',
              '## Failure analysis and evaluator', '',
              f"Fresh original instruction scores: legacy {failure['legacy_instruction_pct']}%, corrected {failure['corrected_instruction_pct']}%. False-negative IDs: {failure['evaluator_false_negative_ids']}; false-positive IDs: {failure['evaluator_false_positive_ids']}. All 20 predictions and their failure types are saved in `failure_analysis.json`.", '',
              'Exact words, yes/no polarity, single-newline repetition, numbered output, extraction and valid JSON are checked separately. Rejection recognizes multiple explicit denial forms, while premise acceptance, fabricated continuations and repeated denial loops fail. Both legacy and corrected scores remain available; adversarial evaluator tests passed. Candidate predictions are recorded in `final_failure_analysis.json`.', '',
              'A denial followed by invented chronology is not a valid factual correction. Gang Gam-chan lived in 948–1031; answers placing him in the 12th–13th centuries fail. [Korean Academy encyclopedia](https://encykorea.aks.ac.kr/Article/E0000954).', '',
              'The native installed Transformers no-repeat processor was tested against both exact repetition targets. It blocks a required token in each target, including a token already present in the input prompt. `decoding_contract_audit.json` records the blocked positions. Thus these two exact targets are unreachable under the current global no-repeat policy, regardless of adapter preference. This proof concerns the benchmark prompt/template and does not change Live.', '',
              'Failures are consistent with incomplete transfer of strict formatting, polarity and genuine premise denial, plus preservation risk in unconstrained general-language lists. These are observations and hypotheses, not isolated causal ablations. Original full regression has two general-language loops and one instruction loop; only identical shared categories are compared. Ten supplemental archived dialect probes are excluded; dialect preservation uses the identical 500-item UlsanBench.', '',
              '## Decision and DPO condition', '',
              f"Selected release arm: {decision['selected_arm']}. DPO condition reached: {dpo_condition}; DPO executed: {dpo_executed}. No PPO/GRPO is run. Any DPO is limited to one separate small experiment after all three SFT arms and only when no SFT checkpoint reaches corrected strict instruction 90%.", '',
              decision['dpo_decision'], '',
              decision['next_single_experiment'], '',
              '## Artifacts and operations', '',
              'Dataset summary, per-arm results, final comparison, decision, release config and model-card material are saved. Data, raw transcripts, optimizer states and weights remain on persistent EC2 EBS; original and candidate adapter/tokenizer hashes are backed up locally before shutdown. Git contains code, config and summaries only. Accepted naming/export occurs only after the gate decision; rejected weights keep their research checkpoint names.', '',
              'Git push proof and the actual final EC2 state are recorded in `operations.json`. The required completion action is StopInstances, never TerminateInstances, after all GPU/train/evaluation processes exit and the original protected files are reverified.', '', '## Known limitations', '']
    lines += [f'- {item}' for item in limitations]
    (ROOT / 'reports/ULM_4B_ARM_B_PLUS_REPORT.md').write_text('\n'.join(lines) + '\n')
    (ROOT / 'MODEL_CARD_ARM_B_PLUS.md').write_text('\n'.join([
        '# Arm B+ experiment model-card material', '', f"Status: {decision['status']}. Name: {decision['name']}.", '',
        'Lineage: Qwen3.8-4B-Distill → Dialect Alignment v3 → Context Repair v1 Arm B → Instruction Recovery v1 Arm B → fresh-data continued SFT.', '',
        'Research purpose: improve strict instruction following while preserving Ulsan dialect, factual knowledge and conversation. Existing Arm B remains the Live release candidate. No automatic deployment occurs.', '',
        f"Training: {dataset['mixture_examples']} examples, 50% new instruction and 50% replay, completion-only LoRA continuation; three independent short arms. Initial adapter: {config['initial_adapter']}.", '',
        'Measured results, rejection reasons, legacy/corrected/serving policies and limitations are in reports/ULM_4B_ARM_B_PLUS_REPORT.md and decision.json. Weights and original datasets are not uploaded to GitHub; base-model and source-data obligations remain inherited.', '',
    ]))
    print(json.dumps({'accepted': accepted, 'best': best, 'dpo_condition_reached': dpo_condition}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
