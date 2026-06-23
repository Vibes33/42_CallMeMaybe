import json
import numpy as np
from typing import List, Dict, Any
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model,
                 functions: List[Dict[str, Any]]):
        self.model = model
        self.functions = functions
        raw_vocab = self._load_vocab()
        self.allowed_function_names = [f["name"] for f in functions]
        self.vocab_items = self._build_and_filter_vocab(raw_vocab)

    def _build_and_filter_vocab(
            self, raw_vocab: Dict[str, int]
    ) -> List[tuple[str, int]]:
        print(f"filtrage du vocabulaire ({len(raw_vocab)}")
        filtered_items = []
        for token_str, token_id in raw_vocab.items():
            try:
                decoded_str = self.model.decode([token_id])
            except Exception:
                continue
            if not decoded_str:
                continue
            if all(ord(c) < 128 for c in decoded_str):
                filtered_items.append((decoded_str, token_id))
        print(f"Vocabulaire réduit{len(filtered_items)}")
        return filtered_items

    def _load_vocab(self) -> Dict[str, int]:
        vocab_path = self.model.get_path_to_vocab_file()
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le vocabulaire: {e}")

    def is_valid_prefix(self, prefix: str) -> bool:

        if "  " in prefix:
            return False

        bp1 = '{"name":"'
        if len(prefix) <= len(bp1):
            return bp1.startswith(prefix)

        if not prefix.startswith(bp1):
            return False

        rem = prefix[len(bp1):]

        if '"' not in rem:
            if any(c.isspace() for c in rem):
                return False
            return any(fn.startswith(rem)
                       for fn in self.allowed_function_names)

        parts = rem.split('"', 1)
        fn_name = parts[0]
        if fn_name not in self.allowed_function_names:
            return False

        rem_after_fn = parts[1]

        bp2 = ',"parameters":{'
        if len(rem_after_fn) <= len(bp2):
            return bp2.startswith(rem_after_fn)

        if not rem_after_fn.startswith(bp2):
            return False

        return True

    def generate_function_call(self, prompt: str,
                               max_tokens: int = 150) -> str:
        functions_context = json.dumps(self.functions, indent=2)

        formatted_prompt = (
            f"System: You are an expert AI assistant.\n"
            f"You must output a JSON function call.\n"
            f"Here are the available functions\n"
            f"and their JSON schemas:\n{functions_context}\n\n"
            f"User: {prompt}\n"
            f"Assistant: {{"
        )

        raw_ids = self.model.encode(formatted_prompt).tolist()

        if raw_ids and isinstance(raw_ids[0], list):
            input_ids = raw_ids[0]
        else:
            input_ids = raw_ids

        generated_text = "{"
        print("Génération en cours...", end="", flush=True)

        for _ in range(max_tokens):
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits)

            masked_logits = np.full_like(logits, -float('inf'))

            for decoded_str, token_id in self.vocab_items:
                test_prefix = generated_text + decoded_str

                if self.is_valid_prefix(test_prefix):
                    masked_logits[token_id] = logits[token_id]

            next_token_id = int(np.argmax(masked_logits))

            if masked_logits[next_token_id] == -float('inf'):
                print(" [ERREUR: Impasse, aucun token valide]")
                break

            next_token_str = self.model.decode([next_token_id])
            generated_text += next_token_str
            input_ids.append(next_token_id)

            open_braces = generated_text.count('{')
            close_braces = generated_text.count('}')
            if open_braces > 0 and open_braces == close_braces:
                break

        print(" Terminé.")
        return generated_text
