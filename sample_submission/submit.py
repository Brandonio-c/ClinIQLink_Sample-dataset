import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

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
        Placeholder function to load the participant's LLM model.
        Replace this function with your actual model loading code.
        
        Example:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained("your-model-name")
        
        This function attempts to load the model and returns it.
        If any error occurs during model loading, the error is caught, logged, and None is returned.
        """
        print("Loading participant's LLM model... (placeholder)", flush=True)
        try:
            # Replace the following line with your actual model loading code.
            model = None  # e.g., model = AutoModelForCausalLM.from_pretrained("your-model-name")
            print("Participant's LLM model loaded successfully.", flush=True)
            return model
        except Exception as e:
            print(f"Error loading participant's LLM model: {e}", flush=True)
            return None


    def load_participant_pipeline(self):
        """
        Placeholder function to load the participant's LLM inference pipeline.
        Replace this function with your actual pipeline initialization code.
        
        Example:
            from transformers import pipeline
            pipeline_instance = pipeline("text-generation", model=self.model)
        
        This function attempts to initialize the LLM inference pipeline and returns it.
        If any error occurs during initialization, the error is caught, logged, and None is returned.
        """
        print("Loading participant's LLM pipeline... (placeholder)", flush=True)
        try:
            # Replace the following line with your actual pipeline initialization code.
            pipeline_instance = None  # e.g., pipeline_instance = pipeline("text-generation", model=self.model)
            print("Participant's LLM pipeline loaded successfully.", flush=True)
            return pipeline_instance
        except Exception as e:
            print(f"Error loading participant's LLM pipeline: {e}", flush=True)
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


    def generate_prompt(self, template, qa, qa_type):
        """
        Generates a prompt for the given QA pair using the specified template.

        Args:
            template (str): The prompt template.
            qa (dict): A dictionary containing question and options (if applicable).
            qa_type (str): The type of question (e.g., "true_false", "multiple_choice", "list", etc.).

        Returns:
            str: A formatted prompt.
        """
        try:
            # Extract common fields
            question = qa.get("question", "Unknown Question")
            answer = qa.get("answer", "")
            options = qa.get("options", {})
            reasoning = qa.get("reasoning", "")
            false_answer = qa.get("false_answer", "")

            if qa_type == "true_false":
                return template.format(question=question)

            elif qa_type == "multiple_choice":
                # Ensure the placeholders match your MC template
                return template.format(
                    question=question,
                    options_A=options.get("A", "Option A missing"),
                    options_B=options.get("B", "Option B missing"),
                    options_C=options.get("C", "Option C missing"),
                    options_D=options.get("D", "Option D missing")
                )

            elif qa_type == "list":
                # Convert list to a joined string for {options_joined}
                options_joined = "\n".join(options) if isinstance(options, list) else str(options)
                return template.format(
                    question=question,
                    options_joined=options_joined
                )

            elif qa_type == "multi_hop":
                return template.format(question=question)

            elif qa_type == "multi_hop_inverse":
                return template.format(
                    question=question,
                    answer=answer,
                    reasoning=reasoning
                )

            elif qa_type == "short":
                return template.format(question=question)

            elif qa_type == "short_inverse":
                return template.format(
                    question=question,
                    false_answer=false_answer
                )

            else:
                print(f"Warning: Unknown QA type '{qa_type}'", flush=True)
                return "Invalid QA type."

        except Exception as e:
            print(f"Error generating prompt: {e}", flush=True)
            return "Error generating prompt."



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



    def YOUR_LLM_PLACEHOLDER(self, prompt):
        """
        This function is a placeholder for interfacing with your actual LLM.
        Replace the contents of this function with your LLM integration as needed.
        For now, it simply returns "NO LLM IMPLEMENTED" to indicate that no LLM is connected.
        """
        # Here, you would normally call your LLM (You must call a local model to do inference, no API requests will be accepted)
        # and return its response. For now, we return a fixed message.
        return "NO LLM IMPLEMENTED"

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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
                    response = self.YOUR_LLM_PLACEHOLDER(prompt)
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
