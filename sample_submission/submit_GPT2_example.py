import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.qa_dir = os.path.join(self.base_dir, "..", "sample_QA_pairs")
        self.template_dir = os.path.join(self.base_dir, "submission_template")
        # Load a pre-trained SentenceTransformer model for semantic similarity calculations.
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        nltk.download('punkt')
        # Placeholder: load the participant's LLM model and inference pipeline.
        self.model = self.load_participant_model()
        self.pipeline = self.load_participant_pipeline()

    def load_participant_model(self):
        """
        Load the GPT-2 model.
        """
        print("Loading GPT-2 model...", flush=True)
        try:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            print("GPT-2 model loaded successfully.", flush=True)
            return model
        except Exception as e:
            print(f"Error loading GPT-2 model: {e}", flush=True)
            return None

    def load_participant_pipeline(self):
        """
        Initialize the text generation pipeline with the GPT-2 model.
        """
        print("Loading GPT-2 pipeline...", flush=True)
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # Ensure the tokenizer has a padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            text_generator = pipeline("text-generation", model=self.model, tokenizer=tokenizer)
            print("GPT-2 pipeline loaded successfully.", flush=True)
            return text_generator
        except Exception as e:
            print(f"Error loading GPT-2 pipeline: {e}", flush=True)
            return None
        

    def load_json(self, filepath):
        """
        Load JSON data from the specified file.
        """
        try:
            with open(filepath, "r") as f:
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
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
            return None


    def compute_f1_score(self, true_list, pred_list):
        """
        Compute precision, recall, and F1 score for list-type answers.
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


    def evaluate_true_false(self, expected, prediction):
        """
        Evaluate True/False questions: returns 1 if answers match, else 0.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating True/False question: {e}", flush=True)
            return 0.0

    def evaluate_multiple_choice(self, expected, prediction):
        """
        Evaluate Multiple Choice questions: returns 1 if the selected option matches the expected answer.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating Multiple Choice question: {e}", flush=True)
            return 0.0

    def evaluate_list(self, expected, prediction):
        """
        Evaluate List questions using the F1 score.
        'expected' should be a list of strings and 'prediction' can be a comma-separated string or list.
        """
        try:
            # Convert prediction to a list if it's a string
            if isinstance(prediction, str):
                pred_list = [item.strip().lower() for item in prediction.split(",")]
            else:
                pred_list = [item.strip().lower() for item in prediction]
            exp_list = [item.strip().lower() for item in expected]
            _, _, f1 = self.compute_f1_score(exp_list, pred_list)
            return f1
        except Exception as e:
            print(f"Error evaluating List question: {e}", flush=True)
            return 0.0


    def compute_word_level_similarity(self, expected_text, prediction_text):
        """
        Compute a word-level similarity score using token embeddings.
        For each word in expected_text, find the maximum cosine similarity with any word in prediction_text,
        and vice versa, then compute the harmonic mean of the averaged precision and recall.
        Returns a float score between 0 and 1.
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
            if (precision + recall) == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            print(f"Error computing word-level similarity: {e}", flush=True)
            return 0.0


    def compute_sentence_level_similarity(self, expected_text, prediction_text):
        """
        Compute sentence-level similarity by splitting texts into sentences,
        encoding them, and averaging the maximum cosine similarity for each expected sentence.
        Returns a float score between 0 and 1.
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
        Compute paragraph-level similarity using embeddings for the full texts.
        Returns a similarity score between 0 and 1.
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
        Evaluate open-ended questions by first checking for an exact match.
        If the response exactly matches the expected answer, return 1.0.
        Otherwise, compute a weighted semantic similarity using:
            - Word-level similarity (weight 0.3)
            - Sentence-level similarity (weight 0.3)
            - Paragraph-level similarity (weight 0.4)
        Full points are given if the final semantic score is >= 0.9,
        0 points if below 0.4, and linear interpolation is used between.
        """
        try:
            if expected.strip().lower() == prediction.strip().lower():
                return 1.0

            word_sim = self.compute_word_level_similarity(expected, prediction)
            sentence_sim = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

            # Weights that sum to 1
            w_word = 0.3
            w_sentence = 0.3
            w_paragraph = 0.4
            semantic_score = w_word * word_sim + w_sentence * sentence_sim + w_paragraph * paragraph_sim

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
        Calculate BLEU, ROUGE, and METEOR scores for the given expected and predicted answers.
        Returns a dictionary with the scores.
        """
        try:
            smoothing_function = SmoothingFunction().method1
            bleu = sentence_bleu([expected.split()], prediction.split(), smoothing_function=smoothing_function)
            meteor = meteor_score([expected], prediction)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(expected, prediction)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            return {"bleu": bleu, "meteor": meteor, "rouge": rouge_avg}
        except Exception as e:
            print(f"Error evaluating open-ended metrics: {e}", flush=True)
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


    def call_GPT2(self, prompt):
        """
        Generate text using the GPT-2 pipeline.
        """
        try:
            # Generate text with a maximum length of 100 tokens
            response = self.pipeline(prompt, max_length=1000, num_return_sequences=1)
            # Extract and return the generated text
            generated_text = response[0]['generated_text']
            return generated_text
        except Exception as e:
            print(f"Error generating text with GPT-2: {e}", flush=True)
            return "Error generating text with GPT-2."

    def evaluate_true_false_questions(self):
        """
        Evaluate all True/False questions and compute the overall F1 score.
        Returns a dictionary containing the average score and a mapping of paragraph_id to individual QA scores.
        """
        try:
            tf_path = os.path.join(self.qa_dir, "TF.json")
            tf_data = self.load_json(tf_path)
            if tf_data is None:
                print("No True/False data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("tf_template.prompt")
            results = {}
            scores = []
            for qa in tf_data:
                try:
                    prompt = self.generate_prompt(template, qa, "true_false")
                    response = self.call_GPT2(prompt)
                    expected = qa.get("answer", "").strip().lower()
                    predicted = response.strip().lower()
                    score = self.evaluate_true_false(expected, predicted)
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
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall True/False F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating True/False questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multiple_choice_questions(self):
        """
        Evaluate all Multiple Choice questions and compute the overall F1 score.
        Returns a dictionary containing the average score and a mapping of paragraph_id to individual QA scores.
        """
        try:
            mc_path = os.path.join(self.qa_dir, "MC.json")
            mc_data = self.load_json(mc_path)
            if mc_data is None:
                print("No Multiple Choice data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("MC_template.prompt")
            results = {}
            scores = []
            for qa in mc_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multiple_choice")
                    response = self.call_GPT2(prompt)
                    expected = qa.get("correct_answer", "").strip().lower()
                    predicted = response.strip().lower()
                    score = self.evaluate_multiple_choice(expected, predicted)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "score": score
                    }
                    scores.append(score)
                except Exception as inner_e:
                    print(f"Error processing Multiple Choice QA: {inner_e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall Multiple Choice F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multiple Choice questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_list_questions(self):
        """
        Evaluate all List questions using the F1 score.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            list_path = os.path.join(self.qa_dir, "list.json")
            list_data = self.load_json(list_path)
            if list_data is None:
                print("No List data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("list_template.prompt")
            results = {}
            scores = []
            for qa in list_data:
                try:
                    prompt = self.generate_prompt(template, qa, "list")
                    response = self.call_GPT2(prompt)
                    expected_items = [item.strip().lower() for item in qa.get("answer", [])]
                    predicted_items = [item.strip().lower() for item in response.split(",")]
                    _, _, f1 = self.compute_f1_score(expected_items, predicted_items)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected_items,
                        "predicted": predicted_items,
                        "score": f1
                    }
                    scores.append(f1)
                except Exception as inner_e:
                    print(f"Error processing List QA: {inner_e}", flush=True)
            overall_f1 = sum(scores) / len(scores) if scores else 0.0
            print(f"Overall List Question F1 Score: {overall_f1:.2f}", flush=True)
            return {"average": overall_f1, "scores": results}
        except Exception as e:
            print(f"Error evaluating List questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_short_questions(self):
        """
        Evaluate all Short Answer questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_path = os.path.join(self.qa_dir, "short.json")
            short_data = self.load_json(short_path)
            if short_data is None:
                print("No Short Answer data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_template.prompt")
            results = {}
            scores = []
            for qa in short_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short")
                    response = self.call_GPT2(prompt)
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
                except Exception as inner_e:
                    print(f"Error processing Short Answer QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Answer questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}

    
    def evaluate_short_inverse_questions(self):
        """
        Evaluate Short Inverse questions by comparing the LLM's response to the provided incorrect explanation.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_inverse_path = os.path.join(self.qa_dir, "short_inverse.json")
            short_inverse_data = self.load_json(short_inverse_path)
            if short_inverse_data is None:
                print("No Short Inverse data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_inverse_template.prompt")
            results = {}
            scores = []
            for qa in short_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short_inverse")
                    response = self.call_GPT2(prompt)
                    print("Short Inverse Response:", response, flush=True)
                    # Use the provided incorrect explanation as the expected text.
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
                except Exception as inner_e:
                    print(f"Error processing Short Inverse QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Inverse F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Inverse questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_questions(self):
        """
        Evaluate all Multi-hop questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id)
        of individual QA scores.
        """
        try:
            mh_path = os.path.join(self.qa_dir, "multi_hop.json")
            mh_data = self.load_json(mh_path)
            if mh_data is None:
                print("No Multi-hop data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_template.prompt")
            results = {}
            scores = []
            for qa in mh_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop")
                    response = self.call_GPT2(prompt)
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
                except Exception as inner_e:
                    print(f"Error processing Multi-hop QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multi-hop questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_inverse_questions(self):
        """
        Evaluate all Multi-hop Inverse questions by comparing the LLM's response with the provided
        incorrect reasoning step. Returns a dictionary containing the average F1 score and a mapping
        (by paragraph_id) of individual QA scores.
        """
        try:
            mh_inverse_path = os.path.join(self.qa_dir, "multi_hop_inverse.json")
            mh_inverse_data = self.load_json(mh_inverse_path)
            if mh_inverse_data is None:
                print("No Multi-hop Inverse data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_inverse_template.prompt")
            results = {}
            scores = []
            for qa in mh_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
                    response = self.call_GPT2(prompt)
                    print("Multi-hop Inverse Response:", response, flush=True)
                    # Use the provided incorrect reasoning step as the expected text.
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
                except Exception as inner_e:
                    print(f"Error processing Multi-hop Inverse QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop Inverse F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multi-hop Inverse questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}



    def run_all_evaluations(self):
        """
        Run evaluations for all QA types and save the overall results to a JSON file.
        """
        try:
            overall_results = {}
            overall_results["true_false"] = self.evaluate_true_false_questions()
            overall_results["multiple_choice"] = self.evaluate_multiple_choice_questions()
            overall_results["list"] = self.evaluate_list_questions()
            overall_results["short"] = self.evaluate_short_questions()
            overall_results["multi_hop"] = self.evaluate_multi_hop_questions()
            overall_results["short_inverse"] = self.evaluate_short_inverse_questions()
            overall_results["multi_hop_inverse"] = self.evaluate_multi_hop_inverse_questions()
            
            overall_json = json.dumps(overall_results, indent=2)
            print("Overall Evaluation Results:", overall_json, flush=True)
            
            output_file = os.path.join(self.base_dir, "overall_evaluation_results.json")
            with open(output_file, "w") as f:
                json.dump(overall_results, f, indent=2)
            print(f"Saved overall evaluation results to {output_file}", flush=True)
        except Exception as e:
            print(f"Error running overall evaluations: {e}", flush=True)



if __name__ == "__main__":
    evaluator = ClinIQLinkSampleDatasetSubmit()
    evaluator.run_all_evaluations()
