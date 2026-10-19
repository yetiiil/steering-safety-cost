#!/usr/bin/env python3
"""Prepare the fixed paper inputs, or fetch optional public datasets.

    python scripts/00_download_data.py               # fixed main inputs, offline
    python scripts/00_download_data.py --list        # what the items are
    python scripts/00_download_data.py --only xstest # just one item
    python scripts/00_download_data.py --force       # restore recorded input bytes

The default restores data/paper/inputs.zip and checks every file against its
SHA-256 manifest. It includes the recorded CAST and DIM splits, three main
behavior datasets, benign evaluation inputs, and all seven attack input files
(one AutoDAN, three PAIR, three GCG). No model responses or analysis are shipped.

The corresponding --only groups also restore the same fixed inputs. Modified
destinations are protected unless --force is supplied. Other --only groups are
optional source downloads for additional behaviors or datasets. MMLU, TriviaQA,
GSM8K and IFEval are still fetched at run time by run_utility.py.
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import PAPER_GROUPS, paper_paths, prepare_paper_data

from cast.paths import (ATTACK_DATA_DIR, BEHAVIOR_DATA_DIR, BENIGN_DATA_DIR,
                        DATA_DIR, REPO_ROOT, TRAIN_DATA_DIR)


class ManualStep(Exception):
    """Raised by an item that has no public download path.

    Carries the message the user needs: which file is missing, where it comes
    from, and how to regenerate it.  main() reports it and keeps going.
    """


# data/raw/ keeps the upstream file byte for byte; data/processed/ keeps the
# normalised [{"instruction": ..., "category": ...}] form that the split
# construction below reads.  Neither is read by the entry points directly.
raw_data_dir = str(DATA_DIR / 'raw')
processed_data_dir = str(DATA_DIR / 'processed')
splits_data_dir = str(DATA_DIR / 'saladbench_splits')

def download_file(url, file_path):
    response = requests.get(url)
    response.raise_for_status()

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "wb") as file:
        file.write(response.content)

def dump_json(data, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def download_advbench():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    raw_file_path = os.path.join(raw_data_dir, 'advbench.csv')
    processed_file_path = os.path.join(processed_data_dir, 'advbench.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['goal'].to_list()
    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_malicious_instruct():
    url = 'https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt'
    raw_file_path = os.path.join(raw_data_dir, 'malicious_instruct.txt')
    processed_file_path = os.path.join(processed_data_dir, 'malicious_instruct.json')

    download_file(url, raw_file_path)
    with open(raw_file_path, 'r') as f:
        instructions = f.readlines()

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_tdc2023():
    urls = [
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json',
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json'
    ]
    raw_file_paths = [
        os.path.join(raw_data_dir, 'tdc2023_dev_behaviors.json'),
        os.path.join(raw_data_dir, 'tdc2023_test_behaviors.json')
    ]
    processed_file_path = os.path.join(processed_data_dir, 'tdc2023.json')

    for url, raw_file_path in zip(urls, raw_file_paths):
        download_file(url, raw_file_path)

    instructions = []
    for raw_file_path in raw_file_paths:
        with open(raw_file_path, 'r') as f:
            dataset = json.load(f)
        instructions.extend(dataset)

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

# The behaviour table moved out of the jailbreakbench package and now lives only
# in the JBB-Behaviors dataset repository, which is also where run_steering.py
# reads it from.  Same 100 behaviours, same `Goal`/`Category` columns.
JAILBREAKBENCH_CSV = ('https://huggingface.co/datasets/JailbreakBench/'
                      'JBB-Behaviors/resolve/main/data/harmful-behaviors.csv')


def download_jailbreakbench():
    url = JAILBREAKBENCH_CSV
    raw_file_path = os.path.join(raw_data_dir, 'jailbreakbench.csv')
    processed_file_path = os.path.join(processed_data_dir, 'jailbreakbench.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['Goal'].to_list()
    categories = dataset['Category'].to_list()

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_harmbench(split):
    assert split in ['val', 'test']

    url = f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_{split}.csv'
    raw_file_path = os.path.join(raw_data_dir, f'harmbench_{split}.csv')
    processed_file_path = os.path.join(processed_data_dir, f'harmbench_{split}.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = []
    categories = []
    # filter out instructions with Category=copyright or Tags=context
    filtered_dataset = dataset[
        ~dataset['FunctionalCategory'].str.contains('copyright', case=False, na=False) &
        ~dataset['Tags'].str.contains('context', case=False, na=False)
    ]

    instructions.extend(filtered_dataset['Behavior'].to_list())
    categories.extend(filtered_dataset['SemanticCategory'].to_list())

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_strongreject():
    url = 'https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv'
    raw_file_path = os.path.join(raw_data_dir, 'strongreject.csv')
    processed_file_path = os.path.join(processed_data_dir, 'strongreject.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['forbidden_prompt'].to_list()
    categories = dataset['category'].to_list()

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_alpaca():
    from datasets import load_dataset

    hf_path = 'tatsu-lab/alpaca'
    processed_file_path = os.path.join(processed_data_dir, 'alpaca.json')

    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_salad_bench():
    from datasets import load_dataset
    dataset = load_dataset("OpenSafetyLab/Salad-Data", name='base_set', split='train')
    raw_file_path = os.path.join(raw_data_dir, 'salad_bench.json')
    processed_file_path = os.path.join(processed_data_dir, 'salad_bench.json')

    # save dataset to json using huggingface function
    dataset.to_json(raw_file_path)

    # remove source "Multilingual" and "ToxicChat"
    dataset = dataset.filter(lambda x: x['source'] not in ["Multilingual", "ToxicChat"])
    max_examples_per_source = 256
    # reduce each source to max 256 examples
    source_groups = dataset.to_pandas().groupby('source')
    filtered_data = []
    for source, group in source_groups:
        if len(group) > max_examples_per_source:
            filtered_data.append(group.sample(n=max_examples_per_source))
        else:
            filtered_data.append(group)
    dataset = pd.concat(filtered_data)

    print(len(dataset))
    # Convert DataFrame back to a list of dictionaries
    dataset_json = [{'instruction': row["question"].strip(), 'source': row["source"]} for index, row in dataset.iterrows()]
    dump_json(dataset_json, processed_file_path)

def download_xstest():
    url = 'https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv'
    raw_file_path = os.path.join(raw_data_dir, 'xstest.csv')
    processed_file_path = os.path.join(processed_data_dir, 'xstest.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)
    # Filter to only include examples where label is safe
    dataset = dataset[dataset["label"] == "safe"]
    dataset_json = [{'instruction': row["prompt"].strip(), 'category': None} for index, row in dataset.iterrows()]
    print(len(dataset_json))
    dump_json(dataset_json, processed_file_path)


def download_orbench_hard():
    from datasets import load_dataset

    hf_path = "bench-llm/or-bench"
    processed_file_path = os.path.join(processed_data_dir, 'orbench_hard.json')

    dataset = load_dataset(hf_path, name="or-bench-hard-1k")
    dataset_json = [{'instruction': d["prompt"].strip(), 'category': d["category"]} for d in dataset['train']]
    dump_json(dataset_json, processed_file_path)

def construct_harmful_dataset_splits():
    harmful_train_path = os.path.join(splits_data_dir, 'harmful_train.json')
    harmful_val_path = os.path.join(splits_data_dir, 'harmful_val.json')
    harmful_test_path = os.path.join(splits_data_dir, 'harmful_test.json')

    harmful_instructions = []
    for file in ['salad_bench.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmful_instructions.extend(json.load(f))
    
    harmful_test_instructions = []
    for file in ['jailbreakbench.json', 'harmbench_test.json', 'strongreject.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmful_test_instructions.extend(json.load(f))

    # now ensure that there are no duplicates across datasets
    filtered_out = 0
    filtered_harmful_instructions = []
    for instruction in harmful_instructions:
        if instruction['instruction'] not in [x['instruction'] for x in harmful_test_instructions]:
            filtered_harmful_instructions.append(instruction)
        else:
            filtered_out += 1

    print(f"filtered out {filtered_out} duplicate instructions from dataset")
    harmful_instructions = filtered_harmful_instructions

    # Now sample from filtered train set for train and val splits
    harmful_train_instructions = []
    harmful_val_instructions = []
    random.seed(42)
    random.shuffle(harmful_instructions)
    harmful_train_instructions = harmful_instructions[:1024]  # Sample 1024 for train
    harmful_val_instructions = harmful_instructions[1024:1024+128]  # Sample 128 for val

    # print length of the train, val and test sets
    print(f"train set length: {len(harmful_train_instructions)}")
    print(f"val set length: {len(harmful_val_instructions)}")
    print(f"test set length: {len(harmful_test_instructions)}")

    dump_json(harmful_train_instructions, harmful_train_path)
    dump_json(harmful_val_instructions, harmful_val_path)
    dump_json(harmful_test_instructions, harmful_test_path)

def construct_harmless_dataset_splits():
    harmless_train_path = os.path.join(splits_data_dir, 'harmless_train.json')
    harmless_val_path = os.path.join(splits_data_dir, 'harmless_val.json')
    harmless_test_path = os.path.join(splits_data_dir, 'harmless_test.json')

    train_p, val_p, test_p = 0.8, 0.10, 0.10

    harmless_instructions = []
    for file in ['alpaca.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmless_instructions.extend(json.load(f))

    random.seed(42)
    random.shuffle(harmless_instructions)

    total_size = len(harmless_instructions)
    train_size = int(train_p * total_size)
    val_size = int(val_p * total_size)

    harmless_train_instructions = harmless_instructions[:train_size]
    harmless_val_instructions = harmless_instructions[train_size:train_size+val_size]
    harmless_test_instructions = harmless_instructions[train_size+val_size:]

    dump_json(harmless_train_instructions, harmless_train_path)
    dump_json(harmless_val_instructions, harmless_val_path)
    dump_json(harmless_test_instructions, harmless_test_path)


# ---------------------------------------------------------------------------
# Files the research tree kept outside data/processed/, in the exact shape the
# entry points expect.
# ---------------------------------------------------------------------------

def build_xtest_safe():
    """data/benign/xtest_safe.json -- the safe half of XSTest (250 prompts).

    `train_cast.py` uses it for the false-refusal constraint and `run_benign.py
    --benign_source xtest` for the reported FRR.  Both read `instruction`; the
    remaining XSTest columns are carried through so a result row can be traced
    back to the prompt it came from.
    """
    raw_file_path = os.path.join(raw_data_dir, 'xstest.csv')
    if not os.path.exists(raw_file_path):
        download_xstest()

    dataset = pd.read_csv(raw_file_path)
    dataset = dataset[dataset["label"] == "safe"]

    # Upstream calls the prompt-type column "type"; tolerate a rename rather
    # than failing with a bare KeyError.
    if "type" in dataset.columns:
        category_column = "type"
    elif "category" in dataset.columns:
        category_column = "category"
    else:
        raise RuntimeError(
            f"Neither 'type' nor 'category' in {raw_file_path}; "
            f"columns are {list(dataset.columns)}"
        )

    def cell(row, column):
        value = row[column]
        return None if pd.isna(value) else value

    dataset_json = [
        {
            'id_v1': cell(row, 'id_v1'),
            'id_v2': cell(row, 'id_v2'),
            'category': cell(row, category_column),
            'instruction': str(row['prompt']).strip(),
            'focus': cell(row, 'focus'),
            'note': cell(row, 'note'),
        }
        for _, row in dataset.iterrows()
    ]
    print(f"XSTest safe prompts: {len(dataset_json)}")
    dump_json(dataset_json, str(BENIGN_DATA_DIR / 'xtest_safe.json'))


def build_orbench_hard_csv():
    """data/benign/or-bench-hard-1k.csv -- over-refusal prompts for the FRR term.

    `train_cast.py` reads this one as a CSV and requires a `prompt` column, so
    it is written as CSV rather than as the JSON that data/processed/ holds.
    """
    from datasets import load_dataset

    dataset = load_dataset("bench-llm/or-bench", name="or-bench-hard-1k")['train']
    frame = pd.DataFrame(
        {
            'prompt': [str(row['prompt']).strip() for row in dataset],
            'category': [row['category'] for row in dataset],
        }
    )
    output_path = BENIGN_DATA_DIR / 'or-bench-hard-1k.csv'
    os.makedirs(output_path.parent, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"OR-Bench hard prompts: {len(frame)} -> {output_path}")


ALPHASTEER_RAW = ('https://raw.githubusercontent.com/AlphaLab-USTC/AlphaSteer'
                  '/main/data/instructions/train_val')


def download_cast_val_split():
    """data/train/harmful_val.json -- the 1000-prompt validation half.

    This half of the CAST prompt pool is one of the files AlphaSteer publishes
    and is identical to the one the paper's runs used, so it is a real download.
    The training half is not published as a file; build_cast_train_split()
    reconstructs it.
    """
    output_path = TRAIN_DATA_DIR / 'harmful_val.json'
    download_file(f'{ALPHASTEER_RAW}/harmful_val.json', str(output_path))

    # Guard against the upstream file being moved or reshaped: a wrong file here
    # would only surface much later, as a KeyError inside a training run.
    records = json.load(open(output_path))
    if not isinstance(records, list) or not records \
            or not all(isinstance(r, dict) and 'query' in r for r in records):
        raise RuntimeError(
            f"{output_path} is not the expected [{{'query': ...}}] list; "
            f"the upstream file at {ALPHASTEER_RAW}/harmful_val.json has changed."
        )
    print(f"CAST validation prompts: {len(records)} -> {output_path}")


def build_cast_train_split():
    """Restore the recorded 1844-record CAST safety training split."""
    prepare_paper_data("train")


def build_dim_splits():
    """Restore the recorded DIM harmful/harmless train and validation splits."""
    prepare_paper_data("dim-splits")


DIM_PROCESSED_DIR = REPO_ROOT / 'third_party' / 'refusal_direction' / 'dataset' / 'processed'


def build_dim_eval_set():
    """Restore the recorded JailbreakBench inputs in the DIM pipeline layout."""
    prepare_paper_data("dim-eval-set")


# ---------------------------------------------------------------------------
# Behaviour datasets
# ---------------------------------------------------------------------------

CAA_REPO_RAW = 'https://raw.githubusercontent.com/nrimsky/CAA/main'

# Two of the paper's six behaviours are published by nrimsky/CAA and are taken
# from there byte for byte, all three files each.
CAA_BEHAVIORS = (
    'myopic-reward',
    'sycophancy',
)

BEHAVIOR_FILES = (
    ('generate', 'generate_dataset.json'),
    ('test', 'test_dataset_ab.json'),
    ('test', 'test_dataset_open_ended.json'),
)


def caa_behavior_outputs():
    return [
        BEHAVIOR_DATA_DIR / behavior / filename
        for behavior in CAA_BEHAVIORS
        for _, filename in BEHAVIOR_FILES
    ]


def download_caa_behaviors():
    """data/behaviors/<behavior>/ for the two behaviours nrimsky/CAA publishes.

    CAA keeps the A/B generation set and the two test sets in separate trees
    (`datasets/generate/` and `datasets/test/`); this repository puts all three
    under one directory per behaviour, which is what cast.paths.behavior_dir
    resolves to.
    """
    for behavior in CAA_BEHAVIORS:
        for split, filename in BEHAVIOR_FILES:
            url = f'{CAA_REPO_RAW}/datasets/{split}/{behavior}/{filename}'
            download_file(url, str(BEHAVIOR_DATA_DIR / behavior / filename))
        print(f"  {behavior}")


ANTHROPIC_EVALS_RAW = ('https://raw.githubusercontent.com/anthropics/evals/main'
                       '/advanced-ai-risk/lm_generated_evals')

# The three main-table behaviours come from Anthropic's model-written
# advanced-ai-risk evaluations, which already use the two-choice schema CAA
# reads: `question`, `answer_matching_behavior`, `answer_not_matching_behavior`.
ANTHROPIC_BEHAVIORS = (
    'corrigible-more-HHH',
    'power-seeking-inclination',
    'self-awareness-good-text-model',
)

# Those files carry generation-provenance markers spliced into the middle of a
# sentence -- "...better ethical<META_START>pile-stackexchange<META_END> and
# rational outcomes...".  They are not part of the prompt and were stripped
# before the released splits were cut.
META_MARKER = re.compile(r'<META_START>.*?<META_END>', re.S)

# self-awareness-good-text-model was not shuffled: the upstream file was cut
# where it stood, 800 records to the generation half and the rest to the test
# half, and its question text was right-trimmed, which the other two were not.
# Both are properties of the released files, so both are reproduced here rather
# than tidied into line with the others.
SELF_AWARENESS = 'self-awareness-good-text-model'
SELF_AWARENESS_GENERATE_SIZE = 800

# How many A/B test questions carry over to the open-ended test set.
OPEN_ENDED_SIZE = 100


def anthropic_behavior_records(behavior):
    """Download one advanced-ai-risk evaluation into the CAA record shape."""
    raw_file_path = os.path.join(raw_data_dir, 'behaviors', f'{behavior}.jsonl')
    download_file(f'{ANTHROPIC_EVALS_RAW}/{behavior}.jsonl', raw_file_path)

    rstrip_question = behavior == SELF_AWARENESS
    records = []
    with open(raw_file_path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question = META_MARKER.sub('', record['question'])
            records.append({
                'question': question.rstrip() if rstrip_question else question,
                'answer_matching_behavior':
                    META_MARKER.sub('', record['answer_matching_behavior']).strip(),
                'answer_not_matching_behavior':
                    META_MARKER.sub('', record['answer_not_matching_behavior']).strip(),
            })
    return records


def build_open_ended_set(behavior):
    """data/behaviors/<behavior>/test_dataset_open_ended.json.

    The first OPEN_ENDED_SIZE A/B test questions with their two options removed:
    the question is cut at the blank line that separates it from the options.
    A handful of upstream questions run their options on after a single newline
    and are therefore left whole, which is what the released files do.
    """
    behavior_path = BEHAVIOR_DATA_DIR / behavior
    with open(behavior_path / 'test_dataset_ab.json') as handle:
        test_records = json.load(handle)

    prompts = [{'question': record['question'].split('\n\n')[0]}
               for record in test_records[:OPEN_ENDED_SIZE]]
    dump_json(prompts, str(behavior_path / 'test_dataset_open_ended.json'))
    return prompts


def anthropic_behavior_outputs():
    return [
        BEHAVIOR_DATA_DIR / behavior / filename
        for behavior in ANTHROPIC_BEHAVIORS
        for _, filename in BEHAVIOR_FILES
    ]


def download_anthropic_behaviors():
    """data/behaviors/<behavior>/ for the three behaviours the main tables report.

    The 9:1 generation/test split is the authors' own, vendored alongside this
    script as split_behavior_datasets.py so that the seed and the ratio stay
    exactly what they were.
    """
    # The sibling module is on sys.path when this file is run as a script, but
    # not when it is imported; put its directory there either way.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from split_behavior_datasets import split_dataset

    behaviors_raw_dir = os.path.join(raw_data_dir, 'behaviors')
    for behavior in ANTHROPIC_BEHAVIORS:
        records = anthropic_behavior_records(behavior)
        if behavior == SELF_AWARENESS:
            behavior_path = BEHAVIOR_DATA_DIR / behavior
            dump_json(records[:SELF_AWARENESS_GENERATE_SIZE],
                      str(behavior_path / 'generate_dataset.json'))
            dump_json(records[SELF_AWARENESS_GENERATE_SIZE:],
                      str(behavior_path / 'test_dataset_ab.json'))
            print(f"{behavior}: {len(records)} items -> "
                  f"{SELF_AWARENESS_GENERATE_SIZE} generate, "
                  f"{len(records) - SELF_AWARENESS_GENERATE_SIZE} test")
        else:
            dump_json(records, os.path.join(behaviors_raw_dir, f'{behavior}.json'))
            split_dataset(behavior, raw_dir=behaviors_raw_dir,
                          out_dir=BEHAVIOR_DATA_DIR)
        prompts = build_open_ended_set(behavior)
        print(f"  {len(prompts)} open-ended prompts")


LENGTH_CONCISENESS = 'length_conciseness'

# The conciseness behaviour is measured on the first 100 Alpaca instructions.
LENGTH_CONCISENESS_SIZE = 100
LENGTH_CONCISENESS_FIRST_PROMPT = 'Give three tips for staying healthy.'


def build_length_conciseness_open_ended():
    """data/behaviors/length_conciseness/test_dataset_open_ended.json.

    The first 100 Alpaca instructions, an instruction that has an input joined
    to it by a newline.  Conciseness is scored by output length rather than by a
    forced two-choice answer, so this open-ended set is the only test file it
    has; length_effect_stats.py reads the completions generated from it.
    """
    from datasets import load_dataset

    dataset = load_dataset('tatsu-lab/alpaca', split='train')
    prompts = []
    for row in dataset.select(range(LENGTH_CONCISENESS_SIZE)):
        joined = (row['instruction'] + '\n' + row['input']
                  if row['input'].strip() else row['instruction'])
        prompts.append({'question': joined.strip()})

    # Alpaca is served in a fixed order and the released file is a prefix of it,
    # so a changed first prompt means the prefix is no longer the same 100.
    if prompts[0]['question'] != LENGTH_CONCISENESS_FIRST_PROMPT:
        raise RuntimeError(
            f"tatsu-lab/alpaca starts with {prompts[0]['question']!r}, expected "
            f"{LENGTH_CONCISENESS_FIRST_PROMPT!r}; the upstream ordering has "
            f"changed and this is no longer the released prompt set."
        )

    output_path = BEHAVIOR_DATA_DIR / LENGTH_CONCISENESS / 'test_dataset_open_ended.json'
    dump_json(prompts, str(output_path))
    print(f"Conciseness prompts: {len(prompts)} -> {output_path}")


def length_conciseness_generate_set():
    raise ManualStep(
        "Missing: data/behaviors/length_conciseness/generate_dataset.json\n"
        "\n"
        "The contrastive A/B pairs the conciseness steering vector was\n"
        "extracted from.  Unlike the other five behaviours it is not one of\n"
        "Anthropic's model-written evaluations and not part of nrimsky/CAA,\n"
        "and the research tree it comes from keeps only the resulting vector,\n"
        "not the pairs -- so this repository cannot reconstruct it and does\n"
        "not know of a public copy.  Request it from the authors.\n"
        "\n"
        "This blocks only Table 16 and the conciseness row of Table 15:\n"
        "without it, scripts/01_generate_vectors.sh and train_cast.py have no\n"
        "conciseness vector to work from.  The open-ended prompt set that\n"
        "scores that vector is built by --only behaviors-conciseness and the\n"
        "other five behaviours are unaffected."
    )


# ---------------------------------------------------------------------------
# Items with no public download path
# ---------------------------------------------------------------------------

def attacks_autodan():
    """Restore the shared AutoDAN prompt file used in the paper."""
    prepare_paper_data("attacks-autodan")


PAIR_FILES = (
    ('qwen/qwen2.5-7b-instruct', 'jailbreakbench_pair_qwen_7b.json'),
    ('qwen/qwen2.5-14b-instruct', 'jailbreakbench_pair_qwen_14b.json'),
    ('meta-llama/llama-3.1-8b-instruct', 'jailbreakbench_pair_llama_8b.json'),
)


def attacks_pair():
    """Restore the three model-specific PAIR input files."""
    prepare_paper_data("attacks-pair")


def attacks_gcg():
    """Restore the three recorded JailbreakBench GCG suffix files."""
    prepare_paper_data("attacks-gcg")


# ---------------------------------------------------------------------------
# HuggingFace cache warm-up
# ---------------------------------------------------------------------------

def prefetch_hf_datasets():
    """Pull the sets the entry points load from the hub at run time.

    Useful before a GPU allocation on a machine with no outbound network, or
    when several jobs would otherwise race to populate the same cache.  Failures
    are reported per dataset and do not stop the others; TriviaQA in particular
    is a multi-gigabyte download.
    """
    from datasets import load_dataset

    failures = []

    def attempt(label, thunk):
        print(f"  prefetching {label} ...")
        try:
            thunk()
        except Exception as exc:                                # noqa: BLE001
            print(f"  ! {label}: {exc}")
            failures.append(label)

    attempt('JailbreakBench/JBB-Behaviors', lambda: pd.read_csv(
        'hf://datasets/JailbreakBench/JBB-Behaviors/data/harmful-behaviors.csv'))
    attempt('tatsu-lab/alpaca', lambda: pd.read_parquet(
        'hf://datasets/tatsu-lab/alpaca/data/'
        'train-00000-of-00001-a09b74b3ef9c3b56.parquet'))
    attempt('cais/mmlu', lambda: load_dataset('cais/mmlu', 'all')['test'])
    attempt('trivia_qa', lambda: load_dataset('trivia_qa', 'rc.wikipedia')['validation'])
    attempt('gsm8k', lambda: load_dataset('gsm8k', 'main')['test'])
    attempt('google/IFEval', lambda: load_dataset('google/IFEval')['train'])

    if failures:
        raise RuntimeError(f"could not prefetch: {', '.join(failures)}")


# ---------------------------------------------------------------------------
# Item registry
# ---------------------------------------------------------------------------

def processed(*names):
    return [Path(processed_data_dir) / name for name in names]


ITEMS = [
    dict(name='advbench', fetch=download_advbench,
         outputs=processed('advbench.json'),
         desc='AdvBench harmful behaviours (llm-attacks)'),
    dict(name='malicious-instruct', fetch=download_malicious_instruct,
         outputs=processed('malicious_instruct.json'),
         desc='MaliciousInstruct prompts'),
    dict(name='tdc2023', fetch=download_tdc2023,
         outputs=processed('tdc2023.json'),
         desc='TDC 2023 red-teaming behaviours'),
    dict(name='jailbreakbench', fetch=download_jailbreakbench,
         outputs=processed('jailbreakbench.json'),
         desc='JailbreakBench behaviours (also the DIM evaluation set)'),
    dict(name='harmbench', fetch=lambda: (download_harmbench(split='val'),
                                          download_harmbench(split='test')),
         outputs=processed('harmbench_val.json', 'harmbench_test.json'),
         desc='HarmBench val/test behaviours'),
    dict(name='strongreject', fetch=download_strongreject,
         outputs=processed('strongreject.json'),
         desc='StrongREJECT forbidden prompts'),
    dict(name='alpaca', fetch=download_alpaca,
         outputs=processed('alpaca.json'),
         desc='Alpaca instructions without inputs (harmless split source)'),
    dict(name='salad-bench', fetch=download_salad_bench,
         outputs=processed('salad_bench.json'),
         desc='SALAD-Bench base set (harmful split source)'),
    dict(name='xstest', fetch=lambda: (download_xstest(), build_xtest_safe()),
         outputs=processed('xstest.json') + [BENIGN_DATA_DIR / 'xtest_safe.json'],
         desc='XSTest -> data/benign/xtest_safe.json'),
    dict(name='orbench', fetch=lambda: (download_orbench_hard(),
                                        build_orbench_hard_csv()),
         outputs=processed('orbench_hard.json')
                 + [BENIGN_DATA_DIR / 'or-bench-hard-1k.csv'],
         desc='OR-Bench hard-1k -> data/benign/or-bench-hard-1k.csv'),
    dict(name='dim-splits', fetch=build_dim_splits,
         requires=['salad-bench', 'jailbreakbench', 'harmbench', 'strongreject',
                   'alpaca'],
         outputs=[Path(splits_data_dir) / f'{kind}_{split}.json'
                  for kind in ('harmful', 'harmless')
                  for split in ('train', 'val', 'test')],
         desc='harmful/harmless splits for the DIM pipeline'),
    dict(name='train-val', fetch=download_cast_val_split,
         outputs=[TRAIN_DATA_DIR / 'harmful_val.json'],
         desc='CAST validation prompts (AlphaSteer) -> data/train/harmful_val.json'),
    dict(name='train', fetch=build_cast_train_split,
         requires=['dim-splits', 'jailbreakbench'],
         outputs=[TRAIN_DATA_DIR / 'harmful_train.json'],
         desc='CAST training prompts (AlphaSteer + SALAD-Bench, minus '
              'JailbreakBench) -> data/train/harmful_train.json'),
    dict(name='dim-eval-set', fetch=build_dim_eval_set,
         requires=['jailbreakbench'],
         outputs=[DIM_PROCESSED_DIR / 'jailbreakbench.json'],
         desc="JailbreakBench copy the DIM pipeline's evaluation stage reads"),
    dict(name='behaviors', fetch=download_caa_behaviors,
         outputs=caa_behavior_outputs(),
         desc='myopic-reward and sycophancy datasets (nrimsky/CAA)'),
    dict(name='behaviors-main', fetch=download_anthropic_behaviors,
         outputs=anthropic_behavior_outputs(),
         desc='corrigibility / power-seeking / self-awareness datasets '
              "(Anthropic's model-written evals, split 9:1)"),
    dict(name='behaviors-conciseness', fetch=build_length_conciseness_open_ended,
         outputs=[BEHAVIOR_DATA_DIR / LENGTH_CONCISENESS
                  / 'test_dataset_open_ended.json'],
         desc='conciseness open-ended prompts (Alpaca) -> '
              'data/behaviors/length_conciseness/'),
    dict(name='behaviors-conciseness-generate',
         fetch=length_conciseness_generate_set,
         outputs=[BEHAVIOR_DATA_DIR / LENGTH_CONCISENESS / 'generate_dataset.json'],
         desc='[manual] the conciseness A/B pairs, needed only for Tables 15-16'),
    dict(name='attacks-autodan', fetch=attacks_autodan,
         outputs=[ATTACK_DATA_DIR / 'jailbreakbench_autodan.csv'],
         desc='fixed AutoDAN prompts'),
    dict(name='attacks-pair', fetch=attacks_pair,
         outputs=[ATTACK_DATA_DIR / filename for _, filename in PAIR_FILES],
         desc='fixed PAIR prompts, one file per model'),
    dict(name='attacks-gcg', fetch=attacks_gcg, outputs=[],
         desc='fixed GCG suffixes for all three models'),
    dict(name='prefetch', fetch=prefetch_hf_datasets, outputs=[], optional=True,
         desc='[opt-in] warm the HF cache for the sets loaded at run time'),
]

# Main experiments always use the recorded inputs, including on --force.
# Source downloads above remain available for optional datasets only.
for item in ITEMS:
    if item['name'] in PAPER_GROUPS:
        name = item['name']
        item.update(outputs=paper_paths(name),
                    desc=f'[paper] fixed {name} inputs (SHA-256 checked)')
        item.pop('requires', None)
ITEMS.insert(0, dict(name='paper', outputs=paper_paths(),
                     desc='[paper] all fixed main-experiment inputs, offline'))
ITEMS.append(dict(name='alpaca-eval', outputs=paper_paths('alpaca-eval'),
                  desc='[paper] the 100 Alpaca evaluation inputs'))

ITEMS_BY_NAME = {item['name']: item for item in ITEMS}


def run_item(name, force, done, failed):
    """Run one item and everything it depends on.  Returns True on success."""
    if name in done:
        return True
    item = ITEMS_BY_NAME[name]

    if name == 'paper' or name in PAPER_GROUPS:
        try:
            prepare_paper_data(None if name == 'paper' else name, force=force)
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failed.append(name)
            return False
        done.add(name)
        return True

    for dependency in item.get('requires', ()):
        if not run_item(dependency, force, done, failed):
            print(f"[skip] {name}: dependency '{dependency}' is unavailable")
            failed.append(name)
            return False

    outputs = item.get('outputs') or []
    if outputs and not force and all(path.exists() for path in outputs):
        print(f"[have] {name}")
        done.add(name)
        return True

    print(f"[get ] {name}: {item['desc']}")
    try:
        item['fetch']()
    except ManualStep as step:
        print()
        print(f"[MISSING] {name}")
        for line in str(step).split('\n'):
            print(f"  {line}")
        print()
        failed.append(name)
        return False
    except Exception as exc:                                    # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        failed.append(name)
        return False

    done.add(name)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see every item and what it writes.",
    )
    parser.add_argument(
        '--only', action='append', metavar='NAME', default=None,
        help='Prepare just this item (repeatable); defaults to the fixed paper inputs.',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List the items and exit without downloading anything.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Restore recorded paper inputs over modified files; re-fetch optional sources.',
    )
    args = parser.parse_args()

    if args.list:
        width = max(len(item['name']) for item in ITEMS)
        for item in ITEMS:
            print(f"{item['name']:<{width}}  {item['desc']}")
            for path in item.get('outputs') or []:
                marker = 'ok  ' if path.exists() else '--  '
                print(f"{'':<{width}}    {marker}{path.relative_to(REPO_ROOT)}")
        return 0

    if args.only:
        unknown = [name for name in args.only if name not in ITEMS_BY_NAME]
        if unknown:
            parser.error(
                f"unknown item(s): {', '.join(unknown)}. "
                f"Known: {', '.join(ITEMS_BY_NAME)}"
            )
        selected = args.only
    else:
        selected = ['paper']

    done, failed = set(), []
    for name in selected:
        run_item(name, args.force, done, failed)

    print()
    print(f"Fetched or already present: {len(done)}/{len(selected)} requested items")
    if failed:
        print("Unavailable, see the messages above:")
        for name in dict.fromkeys(failed):
            print(f"  - {name}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
