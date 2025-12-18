"""
Question Generator for Interview Coach
Loads fine-tuned model and generates interview questions.
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class InterviewGenerator:
    """
    Generate interview questions using fine-tuned model.
    """

    # Your model on HuggingFace
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    LORA_MODEL = "shubhampareek/interview-coach-lora"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize generator.

        Args:
            device: "cuda", "mps" (Mac), or "cpu"
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {device}")

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the fine-tuned model."""
        print(f"Loading base model: {self.BASE_MODEL}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.LORA_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        if self.device == "cuda":
            # GPU - use float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU/MPS - load smaller
            print("Warning: Loading on CPU/MPS will be slow. Consider using Colab with GPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Load LoRA adapters
        print(f"Loading LoRA adapters: {self.LORA_MODEL}")
        self.model = PeftModel.from_pretrained(self.model, self.LORA_MODEL)

        if self.device in ["cpu", "mps"]:
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully!")

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 400,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> str:
        """
        Generate interview questions from prompt.

        Args:
            prompt: Formatted prompt with resume context
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated interview questions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format as chat
        messages = [
            {
                "role": "system",
                "content": "You are an expert interviewer. Generate relevant interview questions based on the candidate's resume."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract generated part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response


def main():
    """Test generator (requires GPU or patience on CPU)."""

    print("=" * 60)
    print("INTERVIEW GENERATOR TEST")
    print("=" * 60)

    generator = InterviewGenerator()

    print("\nNote: Loading 7B model requires ~14GB RAM")
    print("On CPU this will be SLOW. Use Colab for real inference.\n")

    proceed = input("Load model? (y/n): ")
    if proceed.lower() != 'y':
        print("Skipped. Use this in Colab with GPU for better performance.")
        return

    generator.load_model()

    # Test prompt
    test_prompt = """
    Here is the candidate's resume information:

    [SKILLS]
    Technical Skills: Python, TensorFlow, PyTorch, SQL, AWS, Docker

    [EXPERIENCE]
    Senior Data Scientist at TechCorp
    Period: 2021 - Present
    - Built recommendation system serving 5M users
    - Reduced model inference time by 60%
    - Led team of 3 ML engineers

    Generate 5 interview questions based on this resume.
    """

    print("\nGenerating questions...")
    response = generator.generate(test_prompt)

    print("\n" + "=" * 60)
    print("GENERATED QUESTIONS:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()