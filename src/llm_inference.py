# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
import argparse
from pathlib import Path

import pandas as pd
import syndisco.model
from tqdm.auto import tqdm

import util.classification


TEXT_COLUMN = "discussion"
OUTPUT_COLUMN = "response"
MAX_LENGTH_CHARS = 2000


def main(
    input_csv_path: Path,
    output_csv_path: Path,
    system_prompt_path: Path,
    hf_model_url: str,
    hf_model_name: str,
):
    SYSTEM_PROMPT = system_prompt_path.read_text().strip()

    if output_csv_path.exists():
        print(f"{output_csv_path} exists, skipping LLM inference.")
        return

    output_csv_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(input_csv_path)
    llm = syndisco.model.TransformersModel(
        model_path=hf_model_url,
        name=hf_model_name,
        max_out_tokens=20,
    )

    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[TEXT_COLUMN])[:util.classification.MAX_LENGTH_CHARS]
        text = text.replace("moderator", "X").replace("Moderator", "X")
        res = llm.prompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=text,
        )
        outputs.append(res)

    df[OUTPUT_COLUMN] = outputs
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a HuggingFace LLM over a CSV dataset"
    )

    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to output CSV",
    )
    parser.add_argument(
        "--system_prompt",
        type=Path,
        required=True,
        help="Path to system prompt text file",
    )
    parser.add_argument(
        "--hf_model_url",
        required=True,
        help="HuggingFace model repo or local path",
    )
    parser.add_argument(
        "--hf_model_name",
        required=True,
        help="Short name/alias for the model",
    )

    args = parser.parse_args()

    main(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        system_prompt_path=args.system_prompt,
        hf_model_url=args.hf_model_url,
        hf_model_name=args.hf_model_name,
    )
