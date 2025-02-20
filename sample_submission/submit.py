import json
import os

class ClinIQLinkSampleDatasetSubmit:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.qa_dir = os.path.join(self.base_dir, "..", "sample_QA_pairs")
        self.template_dir = os.path.join(self.base_dir, "submission_template")

    def load_json(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_template(self, filename):
        filepath = os.path.join(self.template_dir, filename)
        with open(filepath, "r") as f:
            return f.read()

    def compute_f1_score(self, true_list, pred_list):
        """
        Compute precision, recall, and F1 score for list-type answers.
        Both inputs should be lists of strings.
        """
        true_set = set(item.strip().lower() for item in true_list)
        pred_set = set(item.strip().lower() for item in pred_list)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def evaluate_true_false(self, expected, prediction):
        """
        Evaluate True/False questions: returns 1 if answers match, else 0.
        """
        return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0

    def evaluate_multiple_choice(self, expected, prediction):
        """
        Evaluate Multiple Choice questions: returns 1 if the selected option matches expected answer.
        """
        return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0

    def evaluate_open_ended(self, expected, prediction):
        """
        Evaluate open-ended questions (short answer, multi-hop) using a placeholder metric.
        Detailed metric implementation is obscured to reduce likelihood of benchmark gamification.
        Here we use a simple exact-match check as a placeholder.
        """
        return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0

    def generate_prompt(self, template_str, qa, qa_type):
        """
        Generate a prompt by replacing placeholders in the template with fields from the QA pair.
        """
        mapping = {}
        mapping["question"] = qa.get("question", "")
        
        if qa_type == "multiple_choice":
            options = qa.get("options", {})
            mapping["options.A"] = options.get("A", "")
            mapping["options.B"] = options.get("B", "")
            mapping["options.C"] = options.get("C", "")
            mapping["options.D"] = options.get("D", "")
        elif qa_type == "list":
            options_list = qa.get("options", [])
            mapping["options_joined"] = ", ".join(options_list)
        # For other types (short, multi_hop, etc.), no extra mapping is needed.
        prompt = template_str.format(**mapping)
        return prompt

    def simulate_llm_response(self, prompt):
        """
        Simulate an LLM response.
        Replace this with actual LLM API calls as needed.
        For demonstration, this function returns a placeholder.
        """
        print("Prompt sent to LLM:")
        print(prompt)
        # For demonstration purposes, return placeholder responses based on prompt content.
        if "Multiple Choice" in prompt:
            return "B"  # placeholder response
        elif "True/False" in prompt:
            return "False"  # placeholder response
        elif "List Question" in prompt:
            # Assume perfect prediction: return the options from the prompt (this is just a simulation)
            # In practice, this should be replaced with your model's output.
            options_line = prompt.split("Options:")[-1].strip()
            return options_line
        elif "Short Answer" in prompt and "inverse" not in prompt:
            return "Their articulation with ribs."  # placeholder
        elif "Multi-hop Question" in prompt and "inverse" not in prompt:
            return "Final Answer: It causes flexion of the head towards the opposite side.\n\nReasoning: Detailed reasoning steps here."
        elif "inverse" in prompt:
            # For inverse questions, simulation is not critical here.
            return "Placeholder inverse response"
        else:
            return ""

    def evaluate_multiple_choice_questions(self):
        mc_path = os.path.join(self.qa_dir, "MC.json")
        mc_data = self.load_json(mc_path)
        template = self.load_template("MC_template.prompt")
        scores = []
        for qa in mc_data:
            prompt = self.generate_prompt(template, qa, "multiple_choice")
            response = self.simulate_llm_response(prompt)
            expected = qa.get("correct_answer", "")
            score = self.evaluate_multiple_choice(expected, response)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Multiple Choice Score: {avg:.2f}")

    def evaluate_true_false_questions(self):
        tf_path = os.path.join(self.qa_dir, "TF.json")
        tf_data = self.load_json(tf_path)
        template = self.load_template("tf_template.prompt")
        scores = []
        for qa in tf_data:
            prompt = self.generate_prompt(template, qa, "true_false")
            response = self.simulate_llm_response(prompt)
            expected = qa.get("answer", "")
            score = self.evaluate_true_false(expected, response)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average True/False Score: {avg:.2f}")

    def evaluate_list_questions(self):
        list_path = os.path.join(self.qa_dir, "list.json")
        list_data = self.load_json(list_path)
        template = self.load_template("list_template.prompt")
        scores = []
        for qa in list_data:
            prompt = self.generate_prompt(template, qa, "list")
            response = self.simulate_llm_response(prompt)
            # Assume LLM returns a comma-separated list.
            expected = qa.get("answer", [])
            pred_list = [item.strip() for item in response.split(",")]
            _, _, f1 = self.compute_f1_score(expected, pred_list)
            scores.append(f1)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average List Question F1 Score: {avg:.2f}")

    def evaluate_short_questions(self):
        short_path = os.path.join(self.qa_dir, "short.json")
        short_data = self.load_json(short_path)
        template = self.load_template("short_template.prompt")
        scores = []
        for qa in short_data:
            prompt = self.generate_prompt(template, qa, "short")
            response = self.simulate_llm_response(prompt)
            expected = qa.get("answer", "")
            score = self.evaluate_open_ended(expected, response)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Short Answer Score: {avg:.2f}")

    def evaluate_multi_hop_questions(self):
        mh_path = os.path.join(self.qa_dir, "multi_hop.json")
        mh_data = self.load_json(mh_path)
        template = self.load_template("multi_hop_template.prompt")
        scores = []
        for qa in mh_data:
            prompt = self.generate_prompt(template, qa, "multi_hop")
            response = self.simulate_llm_response(prompt)
            expected = qa.get("answer", "")
            score = self.evaluate_open_ended(expected, response)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average Multi-hop Score: {avg:.2f}")

    def run_all_evaluations(self):
        self.evaluate_multiple_choice_questions()
        self.evaluate_true_false_questions()
        self.evaluate_list_questions()
        self.evaluate_short_questions()
        self.evaluate_multi_hop_questions()

if __name__ == "__main__":
    evaluator = ClinIQLinkSampleDatasetSubmit()
    evaluator.run_all_evaluations()
