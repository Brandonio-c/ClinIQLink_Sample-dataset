import json
import os
import random
import numpy as np
import argparse
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import TextGenerationPipeline

# Explicitly set HuggingFace & Torch cache paths for consistency and container safety
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/transformers"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure NLTK uses the provided global path
nltk.data.path.append(os.environ.get("NLTK_DATA", "/usr/local/nltk_data"))
try:
    nltk.data.find("tokenizers/punkt")
    print("NLTK 'punkt' tokenizer available.", flush=True)
except LookupError:
    print("Warning: 'punkt' tokenizer not found. Downloading...", flush=True)
    nltk.download('punkt', quiet=True)

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self, run_mode="container", max_length=1028, sample_sizes=None):
        self.run_mode = run_mode.lower()
        self.max_length = max_length
        self.sample_sizes = sample_sizes or {}
        
        # Set up base directory and SentenceTransformer model based on run mode.
        if self.run_mode == "container":
            print("Running in container mode.", flush=True)
            self.base_dir = "/app"
            self.st_model = SentenceTransformer('/app/models/sentence-transformers_all-MiniLM-L6-v2', local_files_only=True)
        else:
            print("Running in local mode.", flush=True)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Dataset and template directories
        self.dataset_dir = os.getenv("DATA_DIR", os.path.join(self.base_dir, "..", "data"))
        self.template_dir = os.path.join(self.base_dir, "submission_template")
        
        # Set up local NLTK data directory
        self.nltk_data_dir = os.path.join(self.base_dir, "nltk_data")
        nltk.data.path.append(self.nltk_data_dir)
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"NLTK '{resource}' tokenizer found.", flush=True)
            except LookupError:
                print(f"NLTK '{resource}' tokenizer not found. Downloading...", flush=True)
                nltk.download(resource, download_dir=self.nltk_data_dir)
                nltk.data.path.append(self.nltk_data_dir)

        # Load participant model and pipeline from model submissions
        self.model = self.load_participant_model()
        self.pipeline = self.load_participant_pipeline()

        # Load and sample the dataset QA pairs
        self.sampled_qa_pairs = self.load_and_sample_dataset()


    def load_participant_model(self):
        """
        Dynamically load the participant's LLM model from the 'model_submission' folder.
        Supports:
         - Hugging Face Pretrained Model directory (by detecting a config.json)
         - PyTorch checkpoints (.pt or .pth files)
         - Custom Python scripts defining a 'model' callable
        """
        print("Searching for participant's LLM model in 'model_submission'...", flush=True)
        if self.run_mode == "local":
            model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        else:
            model_dir_env = os.getenv("USE_INTERNAL_MODEL", "1").strip().lower()
            if model_dir_env in ["1", "true", "yes"]:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "../model_submission")

        if not os.path.exists(model_submissions_dir):
            print(f"Error: 'model_submission' folder not found at {model_submissions_dir}", flush=True)
            return None

        # Iterate through submissions searching for a model
        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)
            # Case 1: Hugging Face Pretrained Model Directory (by presence of config.json)
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                print(f"Loading Hugging Face model from: {entry_path}", flush=True)
                try:
                    model = AutoModelForCausalLM.from_pretrained(entry_path, trust_remote_code=True, use_safetensors=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(entry_path, trust_remote_code=True)
                    print("Participant's Hugging Face model loaded successfully.", flush=True)
                    return model
                except Exception as e:
                    print(f"Failed to load Hugging Face model: {e}", flush=True)
            # Case 2: PyTorch checkpoint
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint: {entry_path}", flush=True)
                try:
                    model = torch.load(entry_path, map_location=torch.device("cpu"))
                    print("Participant's PyTorch model loaded successfully.", flush=True)
                    return model
                except Exception as e:
                    print(f"Failed to load PyTorch model checkpoint: {e}", flush=True)
            # Case 3: Python script-based model
            elif entry.endswith(".py"):
                print(f"Attempting to execute model script: {entry_path}", flush=True)
                try:
                    model_namespace = {}
                    with open(entry_path, "r") as f:
                        exec(f.read(), model_namespace)
                    model = model_namespace.get("model", None)
                    if model:
                        print("Participant's Python-based model loaded successfully.", flush=True)
                        return model
                    else:
                        print(f"No 'model' object found in {entry_path}.", flush=True)
                except Exception as e:
                    print(f"Failed to execute model script: {e}", flush=True)
        print("Error: No valid model found in 'model_submission'.", flush=True)
        return None


    def load_participant_pipeline(self):
        """
        Dynamically load the LLM inference pipeline.
        If a Hugging Face model is found, it returns a text-generation pipeline.
        For other types, you may need to use the model's 'generate' method directly.
        """
        print("Searching for participant's LLM pipeline in 'model_submission'...", flush=True)
        if self.run_mode == "local":
            model_submissions_dir = os.path.join(self.base_dir, "../model_submission")
        else:
            model_dir_env = os.getenv("USE_INTERNAL_MODEL", "1").strip().lower()
            if model_dir_env in ["1", "true", "yes"]:
                model_submissions_dir = os.path.join(self.base_dir, "model_submission")
            else:
                model_submissions_dir = os.path.join(self.base_dir, "../model_submission")

        if not os.path.exists(model_submissions_dir):
            print(f"Error: 'model_submission' folder not found at {model_submissions_dir}", flush=True)
            return None

        for entry in os.listdir(model_submissions_dir):
            entry_path = os.path.join(model_submissions_dir, entry)
            if os.path.isdir(entry_path) and "config.json" in os.listdir(entry_path):
                print(f"Loading Hugging Face pipeline from: {entry_path}", flush=True)
                try:
                    model = AutoModelForCausalLM.from_pretrained(entry_path, trust_remote_code=True, use_safetensors=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(entry_path, trust_remote_code=True)
                    print("Hugging Face pipeline loaded successfully.", flush=True)
                    return pipeline("text-generation", model=model, tokenizer=self.tokenizer)
                except Exception as e:
                    print(f"Failed to load Hugging Face pipeline: {e}", flush=True)
            elif entry.endswith(".pt") or entry.endswith(".pth"):
                print(f"Loading PyTorch model checkpoint for pipeline: {entry_path}", flush=True)
                try:
                    model = torch.load(entry_path, map_location=torch.device("cpu"))
                    print("PyTorch model loaded successfully for pipeline.", flush=True)
                    return model
                except Exception as e:
                    print(f"Failed to load PyTorch model checkpoint: {e}", flush=True)
            elif entry.endswith(".py"):
                print(f"Attempting to execute model script for pipeline: {entry_path}", flush=True)
                try:
                    model_namespace = {}
                    with open(entry_path, "r") as f:
                        exec(f.read(), model_namespace)
                    model = model_namespace.get("model", None)
                    if model:
                        print("Python-based model pipeline loaded successfully.", flush=True)
                        return model
                    else:
                        print(f"No 'model' object found in {entry_path}.", flush=True)
                except Exception as e:
                    print(f"Failed to execute model script: {e}", flush=True)
        print("Error: No valid pipeline found in 'model_submission'.", flush=True)
        return None


    def load_json(self, filepath):
        """
        Load JSON data from the specified file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}", flush=True)
            return None


    def load_template(self, filename):
        """
        Load the content of the template file.
        """
        filepath = os.path.join(self.template_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
            return None


    def load_and_sample_dataset(self):
        """
        Load QA pairs for each type from the dataset directory and randomly sample a fixed number 
        defined in sample_sizes. The expected filenames for each type are:
         - "TF.json" for True/False
         - "MC.json" for Multiple Choice
         - "list.json" for List questions
         - "short.json" for Short Answer
         - "short_inverse.json" for Short Inverse
         - "multi_hop.json" for Multi-hop
         - "multi_hop_inverse.json" for Multi-hop Inverse
        """
        qa_types = {
            "true_false": ("TF.json", self.sample_sizes.get("num_tf", 200)),
            "multiple_choice": ("MC.json", self.sample_sizes.get("num_mc", 200)),
            "list": ("list.json", self.sample_sizes.get("num_list", 200)),
            "short": ("short.json", self.sample_sizes.get("num_short", 200)),
            "short_inverse": ("short_inverse.json", self.sample_sizes.get("num_short_inv", 200)),
            "multi_hop": ("multi_hop.json", self.sample_sizes.get("num_multi", 200)),
            "multi_hop_inverse": ("multi_hop_inverse.json", self.sample_sizes.get("num_multi_inv", 200)),
        }
        sampled_qa = {}
        for qa_type, (filename, sample_size) in qa_types.items():
            filepath = os.path.join(self.dataset_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Flatten if nested and sample
                    flat_data = [item for sub in data for item in (sub if isinstance(sub, list) else [sub])]
                    sampled_qa[qa_type] = random.sample(flat_data, min(sample_size, len(flat_data)))
            except Exception as e:
                print(f"Error loading {filename}: {e}", flush=True)
                sampled_qa[qa_type] = []
        print(f"Successfully sampled {sum(len(v) for v in sampled_qa.values())} QA pairs.", flush=True)
        return sampled_qa


    def generate_prompt(self, template, qa, qa_type):
        """
        Generate a prompt for a given QA pair using the provided template.
        Supports several QA types.
        """
        try:
            if not isinstance(qa, dict):
                print(f"Error: QA is not a dictionary; got {type(qa)}", flush=True)
                return "Invalid QA input."
            question = qa.get("question", "Unknown Question")
            answer = qa.get("answer", "")
            options = qa.get("options", {})
            reasoning = qa.get("reasoning", "")
            false_answer = qa.get("false_answer", "")
            if qa_type == "true_false":
                return template.format(question=question)
            elif qa_type == "multiple_choice":
                if isinstance(options, list):
                    letter_map = {chr(65 + i): opt for i, opt in enumerate(options)}
                else:
                    raise ValueError("Multiple choice options should be provided as a list.")
                return template.format(
                    question=question,
                    options_A=letter_map.get("A", "Option A missing"),
                    options_B=letter_map.get("B", "Option B missing"),
                    options_C=letter_map.get("C", "Option C missing"),
                    options_D=letter_map.get("D", "Option D missing")
                )
            elif qa_type == "list":
                # For list type, join options with letter labels if provided as list
                if isinstance(options, list):
                    options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
                elif isinstance(options, dict):
                    options_dict = options
                else:
                    options_dict = {"A": str(options)}
                options_joined = "\n".join(f"{k}: {v}" for k, v in options_dict.items())
                return template.format(question=question, options_joined=options_joined)
            elif qa_type == "multi_hop":
                return template.format(question=question)
            elif qa_type == "multi_hop_inverse":
                return template.format(question=question, answer=answer, reasoning=reasoning)
            elif qa_type == "short":
                return template.format(question=question)
            elif qa_type == "short_inverse":
                return template.format(question=question, false_answer=false_answer)
            else:
                print(f"Warning: Unknown QA type '{qa_type}'", flush=True)
                return f"Unsupported QA type: {qa_type}"
        except Exception as e:
            print(f"Exception in generate_prompt: {e}", flush=True)
            return "Error generating prompt."


    def compute_f1_score(self, true_list, pred_list):
        """
        Compute the F1 score (based on precision and recall) for list-type answers.
        """
        try:
            true_set = set(item.strip().lower() for item in true_list)
            pred_set = set(item.strip().lower() for item in pred_list)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing F1 score: {e}", flush=True)
            return 0.0, 0.0, 0.0


    def compute_word_level_similarity(self, expected_text, prediction_text):
        """
        Compute word-level semantic similarity using cosine similarity over word embeddings.
        """
        try:
            expected_words = expected_text.split()
            prediction_words = prediction_text.split()
            if not expected_words or not prediction_words:
                return 0.0
            expected_embeds = self.st_model.encode(expected_words, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_words, convert_to_tensor=True).cpu().numpy()
            sims_expected = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            sims_prediction = [np.max(cosine_similarity([embed], expected_embeds)) for embed in prediction_embeds]
            recall = np.mean(sims_expected)
            precision = np.mean(sims_prediction)
            return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        except Exception as e:
            print(f"Error computing word-level similarity: {e}", flush=True)
            return 0.0


    def compute_sentence_level_similarity(self, expected_text, prediction_text):
        """
        Compute sentence-level semantic similarity by splitting texts into sentences and comparing embeddings.
        """
        try:
            expected_sentences = nltk.sent_tokenize(expected_text)
            prediction_sentences = nltk.sent_tokenize(prediction_text)
            if not expected_sentences or not prediction_sentences:
                return 0.0
            expected_embeds = self.st_model.encode(expected_sentences, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_sentences, convert_to_tensor=True).cpu().numpy()
            sims = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            return np.mean(sims)
        except Exception as e:
            print(f"Error computing sentence-level similarity: {e}", flush=True)
            return 0.0


    def compute_paragraph_level_similarity(self, expected_text, prediction_text):
        """
        Compute paragraph-level semantic similarity using embeddings of full texts.
        """
        try:
            expected_embed = self.st_model.encode(expected_text, convert_to_tensor=True).cpu().numpy()
            prediction_embed = self.st_model.encode(prediction_text, convert_to_tensor=True).cpu().numpy()
            sim = cosine_similarity([expected_embed], [prediction_embed])[0][0]
            return sim
        except Exception as e:
            print(f"Error computing paragraph-level similarity: {e}", flush=True)
            return 0.0


    def evaluate_open_ended(self, expected, prediction):
        """
        Evaluate open-ended answers by a weighted combination of word, sentence, and paragraph-level similarity.
        Returns a score from 0 to 1.
        """
        try:
            if isinstance(expected, list):
                expected = " ".join(expected)
            if isinstance(prediction, list):
                prediction = " ".join(prediction)
            if expected.strip().lower() == prediction.strip().lower():
                return 1.0
            word_sim = self.compute_word_level_similarity(expected, prediction)
            sentence_sim = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)
            semantic_score = 0.3 * word_sim + 0.3 * sentence_sim + 0.4 * paragraph_sim
            if semantic_score >= 0.9:
                return 1.0
            elif semantic_score < 0.4:
                return 0.0
            else:
                return (semantic_score - 0.4) / 0.5
        except Exception as e:
            print(f"Error evaluating open-ended question: {e}", flush=True)
            return 0.0


    def evaluate_open_ended_metrics(self, expected, prediction):
        """
        Calculate BLEU, ROUGE, and METEOR scores for open-ended questions.
        """
        try:
            smoothing_function = SmoothingFunction().method1
            expected_tokens = word_tokenize(expected) if isinstance(expected, str) else expected
            predicted_tokens = word_tokenize(prediction) if isinstance(prediction, str) else prediction
            bleu = sentence_bleu([expected_tokens], predicted_tokens, smoothing_function=smoothing_function)
            meteor = meteor_score([' '.join(expected_tokens)], ' '.join(predicted_tokens))
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(expected, prediction)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            return {"bleu": bleu, "meteor": meteor, "rouge": rouge_avg}
        except Exception as e:
            print(f"Error evaluating open-ended metrics: {e}", flush=True)
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


    def participant_model(self, prompt, qa_type=None):
        """
        Generate a response using the participant's model.
        Uses a Hugging Face text-generation pipeline if available,
        falls back on model.generate if defined, or calls the model if callable.
        Post-processes the output based on the QA type.
        """
        if not self.model:
            print("No participant model loaded. Returning default response.", flush=True)
            return "NO LLM IMPLEMENTED"
        try:
            # If using a Hugging Face pipeline:
            if isinstance(self.pipeline, TextGenerationPipeline):
                response = self.pipeline(prompt, max_length=self.max_length, do_sample=True)[0]['generated_text']
            # If the model has a generate method:
            elif hasattr(self.model, "generate"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output_ids = self.model.generate(input_ids, max_length=self.max_length)
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # If model is a callable script-based model:
            elif callable(self.model):
                response = self.model(prompt)
            else:
                print("Unknown model type. Returning default response.", flush=True)
                response = "NO LLM IMPLEMENTED"
        except Exception as e:
            print(f"Error during inference: {e}", flush=True)
            response = "ERROR DURING INFERENCE"

        if isinstance(response, str):
            response_clean = response.strip().lower()
            if qa_type == "true_false":
                if "true" in response_clean:
                    response = "true"
                elif "false" in response_clean:
                    response = "false"
            elif qa_type == "multiple_choice":
                match = re.search(r"\b[a-d]\b", response_clean)
                response = match.group() if match else response_clean
            elif qa_type == "list":
                response = ", ".join(re.findall(r"[a-zA-Z0-9\s\-]+", response_clean))
            elif qa_type in {"short", "multi_hop", "short_inverse", "multi_hop_inverse"}:
                response = response_clean
        return response


    def evaluate_true_false_questions(self):
        """
        Evaluate all True/False questions.
        """
        tf_data = self.sampled_qa_pairs.get("true_false", [])
        if not tf_data:
            print("No True/False data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("tf_template.prompt")
        results = {}
        scores = []
        for qa in tf_data:
            try:
                prompt = self.generate_prompt(template, qa, "true_false")
                response = self.participant_model(prompt, qa_type="true_false")
                expected = qa.get("answer", "").strip().lower()
                predicted = response.strip().lower()
                score = 1.0 if expected == predicted else 0.0
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected,
                    "predicted": predicted,
                    "score": score
                }
                scores.append(score)
            except Exception as e:
                print(f"Error processing True/False QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Overall True/False F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_multiple_choice_questions(self):
        """
        Evaluate all Multiple Choice questions.
        """
        mc_data = self.sampled_qa_pairs.get("multiple_choice", [])
        if not mc_data:
            print("No Multiple Choice data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("MC_template.prompt")
        results = {}
        scores = []
        for qa in mc_data:
            try:
                prompt = self.generate_prompt(template, qa, "multiple_choice")
                response = self.participant_model(prompt, qa_type="multiple_choice")
                expected_value = qa.get("correct_answer", "").strip().lower()
                options = qa.get("options", [])
                letter_map = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                value_to_letter = {v: k for k, v in letter_map.items()}
                expected_letter = value_to_letter.get(expected_value)
                predicted = response.strip().upper()
                score = 1.0 if predicted == expected_letter else 0.0
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected_letter,
                    "predicted": predicted,
                    "score": score
                }
                scores.append(score)
            except Exception as e:
                print(f"Error processing Multiple Choice QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Overall Multiple Choice F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_list_questions(self):
        """
        Evaluate all List questions using letter mappings and F1 score.
        """
        list_data = self.sampled_qa_pairs.get("list", [])
        if not list_data:
            print("No List data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("list_template.prompt")
        results = {}
        scores = []
        for qa in list_data:
            try:
                prompt = self.generate_prompt(template, qa, "list")
                response = self.participant_model(prompt, qa_type="list")
                options = qa.get("options", [])
                letter_to_option = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
                option_to_letter = {v: k for k, v in letter_to_option.items()}
                expected_values = [ans.strip().lower() for ans in qa.get("answer", [])]
                expected_letters = list({option_to_letter.get(v) for v in expected_values if option_to_letter.get(v)})
                predicted_letters = list({x.strip().upper() for x in response.split(",") if x.strip().upper() in letter_to_option})
                expected_letters.sort()
                predicted_letters.sort()
                _, _, f1 = self.compute_f1_score(expected_letters, predicted_letters)
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected_letters,
                    "predicted": predicted_letters,
                    "score": f1
                }
                scores.append(f1)
            except Exception as e:
                print(f"Error processing List QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Overall List Question F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_short_questions(self):
        """
        Evaluate all Short Answer questions using semantic similarity.
        """
        short_data = self.sampled_qa_pairs.get("short", [])
        if not short_data:
            print("No Short Answer data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("short_template.prompt")
        results = {}
        scores = []
        for qa in short_data:
            try:
                prompt = self.generate_prompt(template, qa, "short")
                response = self.participant_model(prompt, qa_type="short")
                expected = qa.get("answer", "")
                f1_score = self.evaluate_open_ended(expected, response)
                metrics = self.evaluate_open_ended_metrics(expected, response)
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected,
                    "predicted": response,
                    "f1_score": f1_score,
                    "metrics": metrics
                }
                scores.append(f1_score)
            except Exception as e:
                print(f"Error processing Short Answer QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Short Answer F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_short_inverse_questions(self):
        """
        Evaluate all Short Inverse questions.
        """
        short_inv_data = self.sampled_qa_pairs.get("short_inverse", [])
        if not short_inv_data:
            print("No Short Inverse data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("short_inverse_template.prompt")
        results = {}
        scores = []
        for qa in short_inv_data:
            try:
                prompt = self.generate_prompt(template, qa, "short_inverse")
                response = self.participant_model(prompt, qa_type="short_inverse")
                expected = qa.get("incorrect_explanation", "")
                f1_score = self.evaluate_open_ended(expected, response)
                metrics = self.evaluate_open_ended_metrics(expected, response)
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected,
                    "predicted": response,
                    "f1_score": f1_score,
                    "metrics": metrics
                }
                scores.append(f1_score)
            except Exception as e:
                print(f"Error processing Short Inverse QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Short Inverse F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_multi_hop_questions(self):
        """
        Evaluate all Multi-hop questions using semantic similarity.
        """
        mh_data = self.sampled_qa_pairs.get("multi_hop", [])
        if not mh_data:
            print("No Multi-hop data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("multi_hop_template.prompt")
        results = {}
        scores = []
        for qa in mh_data:
            try:
                prompt = self.generate_prompt(template, qa, "multi_hop")
                response = self.participant_model(prompt, qa_type="multi_hop")
                expected = qa.get("answer", "")
                f1_score = self.evaluate_open_ended(expected, response)
                metrics = self.evaluate_open_ended_metrics(expected, response)
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected,
                    "predicted": response,
                    "f1_score": f1_score,
                    "metrics": metrics
                }
                scores.append(f1_score)
            except Exception as e:
                print(f"Error processing Multi-hop QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Multi-hop F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def evaluate_multi_hop_inverse_questions(self):
        """
        Evaluate all Multi-hop Inverse questions.
        """
        mh_inv_data = self.sampled_qa_pairs.get("multi_hop_inverse", [])
        if not mh_inv_data:
            print("No Multi-hop Inverse data loaded.", flush=True)
            return {"average": 0.0, "scores": {}}
        template = self.load_template("multi_hop_inverse_template.prompt")
        results = {}
        scores = []
        for qa in mh_inv_data:
            try:
                prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
                response = self.participant_model(prompt, qa_type="multi_hop_inverse")
                expected = qa.get("incorrect_reasoning_step", "")
                f1_score = self.evaluate_open_ended(expected, response)
                metrics = self.evaluate_open_ended_metrics(expected, response)
                para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                results[para_id] = {
                    "question": qa.get("question", ""),
                    "expected": expected,
                    "predicted": response,
                    "f1_score": f1_score,
                    "metrics": metrics
                }
                scores.append(f1_score)
            except Exception as e:
                print(f"Error processing Multi-hop Inverse QA: {e}", flush=True)
        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Multi-hop Inverse F1 Score: {overall:.2f}", flush=True)
        return {"average": overall, "scores": results}


    def run_all_evaluations(self):
        """
        Run evaluations for all QA types and write the overall results to a JSON file.
        """
        try:
            overall_results = {
                "true_false": self.evaluate_true_false_questions(),
                "multiple_choice": self.evaluate_multiple_choice_questions(),
                "list": self.evaluate_list_questions(),
                "short": self.evaluate_short_questions(),
                "short_inverse": self.evaluate_short_inverse_questions(),
                "multi_hop": self.evaluate_multi_hop_questions(),
                "multi_hop_inverse": self.evaluate_multi_hop_inverse_questions()
            }
            overall_json = json.dumps(overall_results, indent=2)
            output_file = os.environ.get("RESULTS_PATH", "/tmp/overall_evaluation_results.json")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(overall_json)
            print(f"Saved overall evaluation results to {output_file}", flush=True)
        except Exception as e:
            print(f"Error running overall evaluations: {e}", flush=True)

            
def parse_args():
    parser = argparse.ArgumentParser(description="ClinIQLink Evaluation Script")
    parser.add_argument(
        "--mode",
        choices=["local", "container"],
        default="container",
        help="Run mode: 'local' for development, 'container' for Docker-based deployment (default: container)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1028,
        help="Maximum token length for generated responses (default: 1028)"
    )
    parser.add_argument("--num_tf", type=int, default=200, help="Number of True/False questions to evaluate")
    parser.add_argument("--num_mc", type=int, default=200, help="Number of Multiple Choice questions to evaluate")
    parser.add_argument("--num_list", type=int, default=200, help="Number of List questions to evaluate")
    parser.add_argument("--num_short", type=int, default=200, help="Number of Short Answer questions to evaluate")
    parser.add_argument("--num_short_inv", type=int, default=200, help="Number of Short Inverse questions to evaluate")
    parser.add_argument("--num_multi", type=int, default=200, help="Number of Multi-hop questions to evaluate")
    parser.add_argument("--num_multi_inv", type=int, default=200, help="Number of Multi-hop Inverse questions to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_sizes = {
        "num_tf": args.num_tf,
        "num_mc": args.num_mc,
        "num_list": args.num_list,
        "num_short": args.num_short,
        "num_short_inv": args.num_short_inv,
        "num_multi": args.num_multi,
        "num_multi_inv": args.num_multi_inv
    }
    evaluator = ClinIQLinkSampleDatasetSubmit(run_mode=args.mode, max_length=args.max_length, sample_sizes=sample_sizes)
    evaluator.run_all_evaluations()
