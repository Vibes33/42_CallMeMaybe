import argparse
import json
import sys
from typing import List, Any
from pathlib import Path

from src.models import FunctionDefinition, PromptInput, FunctionCallResult
from src.decoder import ConstrainedDecoder
from llm_sdk import Small_LLM_Model


def load_json(filepath: str) -> Any:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur : {filepath} est introuvable.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Erreur : {filepath} n'est pas un JSON", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")

    args = parser.parse_args()

    raw_functions = load_json(args.functions_definition)
    functions = [FunctionDefinition(**f) for f in raw_functions]

    raw_prompts = load_json(args.input)
    prompts = [PromptInput(**p) for p in raw_prompts]

    print("LLM Download")
    try:
        model = Small_LLM_Model()
        functions_dumped = [f.model_dump() for f in functions]
        decoder = ConstrainedDecoder(model, functions_dumped)
    except KeyboardInterrupt:
        print("\nInterruption", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Erreur lors de l'init du modèle : {e}", file=sys.stderr)
        sys.exit(1)

    results: List[dict] = []

    try:
        for item in prompts:
            print(f"\nTraitement du prompt : '{item.prompt}'")

            raw_json_output = decoder.generate_function_call(item.prompt)
            print(f"Sortie brute générée : {raw_json_output}")

            try:
                parsed_json = json.loads(raw_json_output)
                parsed_json["prompt"] = item.prompt
                result = FunctionCallResult(**parsed_json)

                results.append(result.model_dump())
                print("JSON Valide et conforme au schéma")

            except json.JSONDecodeError:
                print("Le LLM n'a pas généré un JSON valide", file=sys.stderr)
            except Exception as e:
                print(f"Echec Pydantic pour ce prompt : {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur", file=sys.stderr)
        print("sauvegarde des résultats générés", file=sys.stderr)

    except Exception as e:
        print(f"erreur pendant la génération : {e}", file=sys.stderr)
        print("Sauvegarde des résultats partiels", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nRésultats sauvegardés dans {output_path}")
    except Exception as e:
        print(f"Erreur des résultats : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nArrêt", file=sys.stderr)
        sys.exit(130)
